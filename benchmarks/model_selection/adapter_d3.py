"""Standalone d3-lite adapter; no generation, remote code, or network access.

The pinned release uses noncausal softmax attention and causal Gated DeltaNet
layers. Running the stock causal backbone would change its trained behavior.
The layer loop, image-feature insertion, linear patch projection and readout
below follow the reviewed vendor/d3/d3_runtime.py (Apache-2.0), from
vllm-sr/d3-lite@03560c26f5199c3c04129a2ccb38960c913635e0.

Required local snapshot: native Qwen3_5Model weights/index, processor/tokenizer
assets, decision_config.json and readout.safetensors. This adapter loads native
Transformers classes only. CPU preparation and host-to-device transfer remain
outside forward so that the parent runner can measure them independently.
GPU synchronization, warm-up and timing belong to the runner.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from PIL import Image

FORMAT_SHA256 = "3badfbc70561bb3f5b8ec16a11d2c455b96ddb32caace0a8a54ca102eae1f3fa"
STATE = "An aerial image tile."
IMAGE_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}


def _format_module():
    """Import only the completely reviewed pure prompt-format module."""
    path = Path(__file__).parent / "vendor" / "d3" / "d3_format.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != FORMAT_SHA256:
        raise ValueError("Vendored d3_format.py does not match the reviewed source")
    spec = importlib.util.spec_from_file_location("benchmark_d3_format", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load reviewed d3 prompt formatter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linear_patch_forward(module, hidden_states):
    """Official equivalent to Conv3d for already flattened, disjoint patches."""
    weight = module.proj.weight
    flat = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
    output = torch.nn.functional.linear(flat, weight.reshape(weight.shape[0], -1), module.proj.bias)
    return output.view(-1, module.embed_dim)


def _noncausal_hidden(language_model, embeds, present, mrope):
    """Official PromptEncoder behavior; padded keys never enter full attention."""
    from transformers.masking_utils import create_recurrent_attention_mask

    rope = language_model.rotary_emb(embeds, mrope)
    keys = present.bool()[:, None, None, :]
    scan = create_recurrent_attention_mask(
        config=language_model.config,
        inputs_embeds=embeds,
        attention_mask=present,
        past_key_values=None,
        position_ids=None,
    )
    hidden = embeds
    depth = language_model.config.num_hidden_layers
    for layer, kind in zip(
        language_model.layers[:depth], language_model.config.layer_types[:depth]
    ):
        hidden = layer(
            hidden,
            position_embeddings=rope,
            attention_mask=keys if kind == "full_attention" else scan,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
        )
    return language_model.norm(hidden)


class Adapter:
    """One queried object per tile, batched across independent image/query rows."""

    def __init__(self, model_path: str, dtype: str = "bfloat16"):
        from safetensors.torch import load_file
        from transformers import AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        root = Path(model_path).resolve()
        if not root.is_dir():
            raise ValueError("model_path must be a complete local d3-lite snapshot")
        if dtype not in ("bfloat16", torch.bfloat16):
            raise ValueError("d3 release numerics require a BF16 backbone and FP32 head")
        config = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
        if (
            config.get("format_id") != "d3-code-readout-v1"
            or config.get("prompt") != "d3"
            or config.get("pooling") != "last"
            or config.get("attention_mode") != "noncausal_full_attention"
            or config.get("readout_dtype") != "float32"
        ):
            raise ValueError("Unsupported d3 decision configuration")
        self.temperature = float(config["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Decision temperature must be finite and positive")
        self.max_length = config.get("max_length")
        self.format = _format_module()
        self.processor = AutoProcessor.from_pretrained(str(root), local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.size = {
            "shortest_edge": 65_536,
            "longest_edge": 1_638_400,
        }
        codes, token_ids = self.format.answer_codes(self.processor.tokenizer)
        if codes != config["codes"] or token_ids != config["token_ids"]:
            raise ValueError("Tokenizer does not match the trained decision readout")
        self.codes = codes
        self.model = (
            Qwen3_5Model.from_pretrained(
                str(root),
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                disable_mmap=True,
            )
            .eval()
            .requires_grad_(False)
        )
        readout = load_file(str(root / "readout.safetensors"))["weight"]
        expected = (255, self.model.config.text_config.hidden_size)
        if tuple(readout.shape) != expected or not torch.isfinite(readout).all():
            raise ValueError("Invalid trained d3 readout")
        # Only A=false and B=true can contribute; this is the release's masked
        # 255-way readout with the 253 impossible outputs omitted from the GEMM.
        self.model.register_buffer("benchmark_readout", readout[:2].float().contiguous())
        patched = 0
        for module in self.model.modules():
            projection = getattr(module, "proj", None)
            if (
                isinstance(projection, torch.nn.Conv3d)
                and tuple(projection.kernel_size) == tuple(projection.stride)
                and hasattr(module, "embed_dim")
            ):
                module.forward = types.MethodType(_linear_patch_forward, module)
                patched += 1
        if patched != 1:
            raise ValueError(f"Expected one visual patch embedding, found {patched}")
        self.metadata = {
            "attention_mode": "noncausal_full_attention",
            "temperature": self.temperature,
            "readout": "trained FP32 A=false/B=true rows",
            "patch_embedding": "official equivalent linear projection",
            "runtime": "native Transformers layers; no d3_fast/custom kernels",
            "state": STATE,
            "question_template": "Does this aerial image contain {query}?",
        }
        self.dtype = "bfloat16"
        self.info = self.metadata

    def prepare(self, images: list[Image.Image], queries: list[str]) -> dict[str, Any]:
        if not images or len(images) != len(queries):
            raise ValueError("Pass a nonempty, aligned image/query batch")
        texts = []
        for image, query in zip(images, queries):
            if not isinstance(image, Image.Image) or image.mode != "RGB":
                raise ValueError("Each input must be the original RGB PIL tile")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Each image needs a nonempty object query")
            if "<|image_pad|>" in query or "<|video_pad|>" in query:
                raise ValueError("Object query contains a reserved media token")
            question = {
                "type": "noul",
                "instructions": f"Does this aerial image contain {query}?",
            }
            messages = [
                {"role": "system", "content": self.format.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": self.format.user_prompt(STATE, question, self.codes),
                        },
                    ],
                },
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            )
        encoded = dict(self.processor(text=texts, images=images, padding=True, return_tensors="pt"))
        if set(encoded) != IMAGE_KEYS:
            raise ValueError(f"Unexpected d3 processor keys: {sorted(encoded)}")
        limit = self.max_length or self.model.config.text_config.max_position_embeddings
        if encoded["input_ids"].shape[1] > limit:
            raise ValueError("Prompt exceeds model context; no truncation is permitted")
        # Left padding ensures every row's final position is a real answer slot.
        if not bool(encoded["attention_mask"][:, -1].all()):
            raise ValueError("Final answer position is padded")
        return encoded

    @torch.inference_mode()
    def forward(self, prepared: dict[str, Any]):
        ids = prepared["input_ids"]
        if ids.device.type != "cuda":
            raise RuntimeError("d3 benchmark requires the requested GPU; no CPU fallback")
        backbone = self.model
        embeds = backbone.get_input_embeddings()(ids)
        pooled = backbone.get_image_features(
            prepared["pixel_values"].to(dtype=backbone.dtype),
            prepared["image_grid_thw"],
            return_dict=True,
        ).pooler_output
        pooled = torch.cat(pooled, dim=0).to(embeds.device, embeds.dtype)
        slots, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=pooled)
        embeds = embeds.masked_scatter(slots, pooled)
        positions = backbone.compute_3d_position_ids(
            input_ids=ids,
            image_grid_thw=prepared["image_grid_thw"],
            video_grid_thw=None,
            inputs_embeds=embeds,
            attention_mask=prepared["attention_mask"],
            past_key_values=None,
            mm_token_type_ids=prepared["mm_token_type_ids"],
        )
        hidden = _noncausal_hidden(
            backbone.language_model, embeds, prepared["attention_mask"], positions
        )[:, -1]
        logits = hidden.float() @ backbone.benchmark_readout.T
        return (logits / self.temperature).softmax(-1)

    def postprocess(self, raw, prepared: dict[str, Any]) -> list[dict[str, float]]:
        if raw.shape != (prepared["input_ids"].shape[0], 2):
            raise ValueError("Unexpected binary decision output shape")
        probabilities = raw.detach().float().cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite d3 probabilities")
        return [{"probability": float(row[1])} for row in probabilities]
