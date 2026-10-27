"""Offline native-Transformers detector adapters for the standalone benchmark.

The source images remain RGB 256 x 256 native embedding windows. Each processor
applies its own checkpoint's required resize and normalization. Box confidence
and the maximum-box proxy below are not calibrated tile-presence probabilities.
Timing, synchronization, device ownership, and batch transfers belong to the
benchmark runner; importing this module does not construct or load a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image


class Adapter:
    """One query per tile, using a previously downloaded local model snapshot.

    ``prepare`` returns CPU tensors only. The runner transfers floating tensors
    using ``dtype`` while preserving integer token and boolean mask dtypes.
    Call ``postprocess`` after the matching ``prepare``/``forward`` pair; this
    object keeps that batch's prompt strings for audit labels. It is not a
    concurrent service or a production pipeline component.
    """

    _MODEL_TYPES = {
        "gdino": "grounding-dino",
        "llmdet": "mm-grounding-dino",
        "owlv2": "owlv2",
        "omdet": "omdet-turbo",
    }

    def __init__(
        self, key: str, model_path: str, dtype: str = "float16", restore_decoder_heads=True
    ):
        from transformers import (
            AutoConfig,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            OmDetTurboForObjectDetection,
        )

        if key not in self._MODEL_TYPES:
            raise ValueError(f"Unknown detection adapter: {key!r}")
        allowed_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in allowed_dtypes:
            raise ValueError(f"Unsupported dtype: {dtype!r}")
        snapshot = Path(model_path).resolve()
        if not snapshot.is_dir():
            raise ValueError("model_path must be an existing local snapshot directory")
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU-enabled PyTorch runtime is required; no CPU fallback")

        self.key = key
        self.dtype = allowed_dtypes[dtype]
        self.threshold = 0.01
        self.nms_threshold = 0.5
        self.text_threshold = 0.25
        self._queries: list[str] = []
        offline = {"local_files_only": True, "trust_remote_code": False}
        if key == "omdet":
            import json

            from transformers import OmDetTurboConfig, TimmBackboneConfig

            raw_config = json.loads((snapshot / "config.json").read_text())
            # Transformers 5.17 otherwise queries Hub to decide whether the
            # checkpoint's legacy timm name is a repository, even offline.
            # Materialize the exact declared timm config; do not fetch weights.
            raw_config["backbone_config"] = TimmBackboneConfig(
                backbone=raw_config["backbone"], **raw_config["backbone_kwargs"]
            )
            config = OmDetTurboConfig.from_dict(raw_config)
        else:
            config = AutoConfig.from_pretrained(str(snapshot), **offline)
        expected_type = self._MODEL_TYPES[key]
        if config.model_type != expected_type:
            raise ValueError(f"{key} needs model_type={expected_type!r}, got {config.model_type!r}")

        # DINO, MM DINO, and OmDet all advertise this config field. Keeping the
        # native PyTorch attention path avoids NVIDIA-only extension builds.
        if hasattr(config, "disable_custom_kernels"):
            config.disable_custom_kernels = True
        self.processor = AutoProcessor.from_pretrained(str(snapshot), **offline)
        model_class = (
            OmDetTurboForObjectDetection if key == "omdet" else AutoModelForZeroShotObjectDetection
        )
        self.model = model_class.from_pretrained(
            str(snapshot),
            config=config,
            dtype=self.dtype,
            use_kernels=False,
            disable_mmap=True,
            **offline,
        ).eval()
        head_audit = None
        if key == "llmdet" and restore_decoder_heads:
            from llmdet_heads import restore_heads

            head_audit = restore_heads(self.model, snapshot)
        self.model.to(device="cuda").eval()
        self.info = {
            "key": key,
            "decoder_head_restore_audit": head_audit,
            "model_type": config.model_type,
            "architecture": type(self.model).__name__,
            "dtype": dtype,
            "source_image": "RGB uint8; original 256 x 256 embedding window",
            "prompt_policy": (
                "one lowercased period-terminated query per image"
                if key in {"gdino", "llmdet"}
                else "one unmodified object-query label per image"
            ),
            "processor_class": type(self.processor).__name__,
            "image_processor_class": type(self.processor.image_processor).__name__,
            "image_processor": self.processor.image_processor.to_dict(),
            "model_config": config.to_dict(),
            "disable_custom_kernels": getattr(config, "disable_custom_kernels", None),
            "use_hub_kernels": False,
            "box_score_threshold": self.threshold,
            "nms_threshold": self.nms_threshold if key == "omdet" else None,
            "dino_text_threshold": self.text_threshold if key in {"gdino", "llmdet"} else None,
            "language_cache": {
                "enabled": key == "omdet",
                "capacity": getattr(config, "cache_size", None),
                "policy": "native OmDet class/prompt LRU caches; warm trials include cache hits"
                if key == "omdet"
                else "native full forward; no adapter-level text cache",
            },
            "score_kind": "detector_box_confidence_not_calibrated_presence_probability",
        }

    def prepare(self, images: list[Image.Image], queries: list[str]) -> dict[str, torch.Tensor]:
        """Prepare a matching batch on CPU without resizing source tiles first."""
        if not images or len(images) != len(queries):
            raise ValueError("Provide one nonempty text query per image")
        if any(image.mode != "RGB" or image.size != (256, 256) for image in images):
            raise ValueError("Benchmark inputs must be unchanged RGB 256 x 256 source windows")
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("Every query must be a nonempty string")
        self._queries = list(queries)

        # A flat list is ambiguous for GroundingDinoProcessor: it may merge it
        # into one multi-class prompt. Period-terminated strings make it an
        # unambiguous batch. Nested labels preserve one class per image for
        # OWLv2/OmDet rather than applying all batch queries to every image.
        if self.key in {"gdino", "llmdet"}:
            text = [query.strip().lower().rstrip(".") + "." for query in queries]
        else:
            text = [[query] for query in queries]
        encoded = self.processor(
            images=images,
            text=text,
            padding=True,
            return_tensors="pt",
        )
        if any(not isinstance(value, torch.Tensor) for value in encoded.values()):
            raise TypeError("Detector processor returned a non-tensor model input")
        if any(value.device.type != "cpu" for value in encoded.values()):
            raise RuntimeError("Preprocessing must return CPU tensors for separate transfer timing")
        return dict(encoded)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Run inference on tensors already transferred by the benchmark runner."""
        with torch.inference_mode():
            return self.model(**batch, return_dict=True)

    def postprocess(self, raw: Any, batch: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """Return native xyxy boxes scaled to original 256-pixel coordinates.

        OmDet uses its native NMS; the two DINO variants and OWLv2 use their
        native postprocessors without adding an unrequested NMS policy. The
        low audit threshold retains candidates for later threshold sweeps.
        """
        count = int(batch["pixel_values"].shape[0])
        if count != len(self._queries):
            raise ValueError("Postprocessing batch does not match the most recent prepared prompts")
        kwargs = {
            "threshold": self.threshold,
            "target_sizes": [(256, 256)] * count,
        }
        if self.key in {"gdino", "llmdet"}:
            kwargs.update(input_ids=batch["input_ids"], text_threshold=self.text_threshold)
        else:
            kwargs["text_labels"] = [[query] for query in self._queries]
        if self.key == "omdet":
            kwargs["nms_threshold"] = self.nms_threshold
        with torch.inference_mode():
            native = self.processor.post_process_grounded_object_detection(raw, **kwargs)

        results = []
        for query, output in zip(self._queries, native, strict=True):
            boxes = output["boxes"].detach().float().cpu().tolist()
            scores = output["scores"].detach().float().cpu().tolist()
            text_labels = output.get("text_labels")
            if text_labels is None:
                text_labels = [query] * len(scores)
            predictions = []
            for box, score, label in zip(boxes, scores, text_labels, strict=True):
                clipped = [max(0.0, min(256.0, coordinate)) for coordinate in box]
                if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                    predictions.append({"box": clipped, "score": score, "text_label": label})
            results.append(
                {
                    "predictions": predictions,
                    "detector_proxy": max(scores, default=0.0),
                    "score_kind": "maximum_retained_box_score_not_presence_probability",
                    "query": query,
                }
            )
        return results
