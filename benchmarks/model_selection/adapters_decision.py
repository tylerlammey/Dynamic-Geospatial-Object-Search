"""Single-forward discriminative decision adapters, never autoregressive generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
STATE = "An aerial image tile."


def question(query):
    return f"Does this aerial image contain {query}?"


def reviewed_import(name, path, expected_hash=None):
    if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Reviewed source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def linear_patch(module, pixels):
    weight = module.proj.weight
    return torch.nn.functional.linear(
        pixels.reshape(-1, weight[0].numel()).to(weight.dtype),
        weight.reshape(weight.shape[0], -1),
        module.proj.bias,
    ).view(-1, module.embed_dim)


class InternAdapter:
    def __init__(self, model_path, dtype="bfloat16", linear_patches=True):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.dtype = dtype
        # Reviewed inference.py is used only for the exact trained prompt compiler.
        pins = json.loads((ROOT / "model_pins.json").read_text())
        self.compiler = reviewed_import(
            "benchmark_intern",
            ROOT / "vendor/intern/inference.py",
            pins["intern"]["source_sha256"]["inference.py"],
        )
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.tokenizer.padding_side = "right"
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
            use_kernels=False,
            disable_mmap=True,
        ).eval()
        if linear_patches:
            patch = self.model.model.visual.patch_embed
            patch.forward = types.MethodType(linear_patch, patch)
        self.marker = self.processor.tokenizer.convert_tokens_to_ids("<decision>")
        self.ids = [
            self.processor.tokenizer.encode(x, add_special_tokens=False)[0] for x in ["A", "B"]
        ]
        self.temperature = self.compiler.DEFAULT_TEMPERATURE
        self.info = {
            "kind": "decision",
            "dtype": dtype,
            "temperature": self.temperature,
            "source_calibration": "Published non-aerial temperature, not NYS calibration",
            "linear_patch_projection": linear_patches,
            "generation": False,
            "state": STATE,
            "processor": self.processor.image_processor.to_dict(),
        }

    def prepare(self, images, queries):
        texts = []
        for q in queries:
            row = {
                "state": STATE,
                "questions": {"present": {"type": "noul", "instructions": question(q)}},
                "images": ["unused"],
            }
            compiled = self.compiler.compile_row(row)
            messages = compiled.messages
            messages[1]["content"] = [{"type": "image"}, messages[1]["content"][-1]]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                    add_vision_id=True,
                )
            )
        batch = dict(self.processor(text=texts, images=images, return_tensors="pt", padding=True))
        positions = (batch["input_ids"] == self.marker).nonzero()
        if positions.shape[0] != len(images) or not torch.equal(
            positions[:, 0], torch.arange(len(images))
        ):
            raise ValueError("Require one trained decision marker per image/query")
        batch["decision_positions"] = positions[:, 1] - 1
        return batch

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k != "decision_positions"}
        # Bypass the full vocabulary projection: use the exact two trained LM rows
        # at the preceding decision positions. All visual/LM computation is retained.
        hidden = self.model.model(**inputs, use_cache=False).last_hidden_state
        positions = batch["decision_positions"]
        pooled = hidden[torch.arange(len(positions), device=hidden.device), positions]
        weights = self.model.lm_head.weight[self.ids].float()
        logits = torch.nn.functional.linear(pooled.float(), weights)
        return (logits / self.temperature).softmax(-1)

    def postprocess(self, raw, batch):
        return [
            {"probability": p, "score_kind": "published_temperature_binary_decision"}
            for p in raw[:, 1].float().cpu().tolist()
        ]


class BlipAdapter:
    def __init__(self, model_path, dtype="float16"):
        from transformers import AutoProcessor, BlipForImageTextRetrieval

        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=getattr(torch, dtype),
            weights_only=True,
            use_kernels=False,
        ).eval()
        self.info = {
            "kind": "matching_control",
            "dtype": dtype,
            "generation": False,
            "score_kind": "caption_match_softmax_not_calibrated_object_presence",
            "processor": self.processor.image_processor.to_dict(),
        }

    def prepare(self, images, queries):
        return dict(
            self.processor(
                images=images,
                text=[f"An aerial image of {q}." for q in queries],
                return_tensors="pt",
                padding=True,
            )
        )

    def forward(self, batch):
        return self.model(**batch, use_itm_head=True).itm_score.float().softmax(-1)

    def postprocess(self, raw, batch):
        return [
            {"probability": p, "score_kind": self.info["score_kind"]}
            for p in raw[:, 1].cpu().tolist()
        ]


class PeekAdapter:
    def __init__(self, model_path, base_path, dtype="bfloat16", deduplicate=True):
        self.dtype = dtype
        self.deduplicate = deduplicate
        source = ROOT / "vendor/peekaboolean_source"
        pins = json.loads((ROOT / "model_pins.json").read_text())
        for name in [
            "src/peekaboolean/model.py",
            "src/peekaboolean/data.py",
            "src/peekaboolean/__init__.py",
        ]:
            path = source / name
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != pins["peekaboolean_source"]["sha256"][name]
            ):
                raise ValueError("Reviewed Peekaboolean source changed")
        sys.path.insert(0, str(source / "src"))
        from peekaboolean.data import candidate_prompts, to_example
        from peekaboolean.model import CandidateScorer, configure_image_size

        self.to_example, self.prompts = to_example, candidate_prompts
        from unittest.mock import patch

        from peft import PeftModel
        from transformers import AutoModelForImageTextToText

        original_loader = PeftModel.from_pretrained

        def cpu_loader(*args, **kwargs):
            kwargs["torch_device"] = "cpu"
            return original_loader(*args, **kwargs)

        # PEFT otherwise infers cuda and safetensors loads the adapter directly
        # there, invoking the unstable Windows ROCm pinned allocator.
        base_loader = AutoModelForImageTextToText.from_pretrained

        def ordinary_cpu_loader(*args, **kwargs):
            kwargs["disable_mmap"] = True
            return base_loader(*args, **kwargs)

        with (
            patch.object(PeftModel, "from_pretrained", cpu_loader),
            patch.object(AutoModelForImageTextToText, "from_pretrained", ordinary_cpu_loader),
        ):
            self.model = CandidateScorer(
                base_path,
                adapter=model_path,
                gradient_checkpointing=False,
                dtype=getattr(torch, dtype),
            ).eval()
        self.model.backbone = self.model.backbone.merge_and_unload()
        self.processor = CandidateScorer.load_processor(base_path, image_size=256, prompt="yesno")
        configure_image_size(self.processor, 256)
        calibration = json.loads((Path(model_path) / "calibration-256.json").read_text())
        self.temperature = calibration["temperatures"]["noul:3"]
        self.info = {
            "kind": "decision",
            "dtype": dtype,
            "temperature": self.temperature,
            "deduplicate_binary_rows": deduplicate,
            "merged_lora": True,
            "adapter_load_device": "cpu",
            "generation": False,
            "unknown_policy": "shipped noul=Pyes+0.5*Punknown; not literal object-presence probability",
            "state": STATE,
            "processor": self.processor.image_processor.to_dict(),
        }

    def prepare(self, images, queries):
        texts, pictures = [], []
        k = 2 if self.deduplicate else 3
        for image, q in zip(images, queries):
            ex = self.to_example(
                {
                    "image": "unused",
                    "type": "noul",
                    "state": STATE,
                    "instructions": question(q),
                    "value": 0.5,
                },
                augment=False,
                unknown=True,
            )
            prompts = self.prompts(ex, "yesno")
            if self.deduplicate:
                prompts = [prompts[0], prompts[2]]
            for prompt in prompts:
                texts.append(
                    self.processor.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                pictures.append([image])
        batch = dict(self.processor(text=texts, images=pictures, return_tensors="pt", padding=True))
        # Official model.hidden shares each image across its candidate sequences.
        for key in ["pixel_values", "pixel_attention_mask", "spatial_shapes"]:
            if key in batch:
                batch[key] = batch[key][::k]
        batch["image_counts"] = torch.full((len(images),), k, dtype=torch.long)
        return batch

    def forward(self, batch):
        scores = self.model(batch).float().reshape(-1, 2 if self.deduplicate else 3)
        logits = (
            torch.stack([-0.5 * scores[:, 0], 0.5 * scores[:, 0], scores[:, 1]], -1)
            if self.deduplicate
            else scores * scores.new_tensor([-0.5, 0.5, 1.0])
        )
        return (logits / self.temperature).softmax(-1)

    def postprocess(self, raw, batch):
        return [
            {
                "probability": yes + 0.5 * unknown,
                "raw_yes_probability": yes,
                "unknown_probability": unknown,
                "conditional_yes_probability": yes / max(1 - unknown, 1e-10),
                "score_kind": "shipped_noul_unknown_midpoint_convention",
            }
            for no, yes, unknown in raw.cpu().tolist()
        ]
