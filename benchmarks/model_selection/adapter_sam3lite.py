"""Native SAM3-LiteText detector and presence-head benchmark adapter.

This baseline retains the full native forward, including mask computation.
The presence sigmoid is exposed separately for evaluation; it has not been
calibrated on the NYS tile/query distribution. No generation is performed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image


class Adapter:
    """Load a local snapshot and score one text query for each original tile."""

    def __init__(self, model_path: str, dtype: str = "float16"):
        from transformers import AutoConfig, AutoProcessor, Sam3LiteTextModel

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
            raise RuntimeError("GPU-enabled PyTorch is required; no CPU inference fallback")

        self.dtype = allowed_dtypes[dtype]
        self.threshold = 0.01
        self._queries: list[str] = []
        offline = {"local_files_only": True, "trust_remote_code": False}
        config = AutoConfig.from_pretrained(str(snapshot), **offline)
        if config.model_type != "sam3_lite_text":
            raise ValueError(f"Expected SAM3-LiteText, got {config.model_type!r}")
        self.processor = AutoProcessor.from_pretrained(str(snapshot), **offline)
        self.model = (
            Sam3LiteTextModel.from_pretrained(
                str(snapshot),
                config=config,
                dtype=self.dtype,
                attn_implementation="sdpa",
                use_kernels=False,
                disable_mmap=True,
                **offline,
            )
            .to(device="cuda")
            .eval()
        )
        self.info = {
            "key": "sam3lite",
            "kind": "detector_with_separate_presence_head",
            "architecture": type(self.model).__name__,
            "dtype": dtype,
            "generation": False,
            "source_image": "RGB uint8; original 256 x 256 embedding window",
            "prompt_policy": "one unmodified text query per image; native SAM3 tokenization",
            "processor_class": type(self.processor).__name__,
            "image_processor_class": type(self.processor.image_processor).__name__,
            "image_processor": self.processor.image_processor.to_dict(),
            "native_input_size": self.processor.image_processor.size,
            "model_config": config.to_dict(),
            "attn_implementation": "sdpa",
            "use_hub_kernels": False,
            "disable_mmap": True,
            "box_score_threshold": self.threshold,
            "box_score_rule": "sigmoid(pred_logits) * sigmoid(presence_logits)",
            "presence_score_kind": "query_conditioned_presence_sigmoid_not_NYS_calibrated",
            "mask_decoder_executed": True,
            "mask_decoder_policy": (
                "Native Transformers forward always executes the mask decoder; "
                "no public skip flag exists in installed 5.17.0. Masks are not postprocessed."
            ),
            "caching": "Full native forward per tile/query; no adapter-level feature cache",
        }

    def prepare(self, images: list[Image.Image], queries: list[str]) -> dict[str, torch.Tensor]:
        """Return CPU model inputs with checkpoint resize/normalization intact."""
        if not images or len(images) != len(queries):
            raise ValueError("Provide one nonempty text query per image")
        if any(image.mode != "RGB" or image.size != (256, 256) for image in images):
            raise ValueError("Inputs must be unchanged RGB 256 x 256 embedding windows")
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("Every query must be a nonempty string")
        self._queries = list(queries)
        encoded = self.processor(images=images, text=list(queries), return_tensors="pt")
        # original_sizes is processor metadata. The detector target is always
        # 256 x 256 here, so it need not be transferred or sent to the model.
        batch = {key: value for key, value in encoded.items() if key != "original_sizes"}
        if any(not isinstance(value, torch.Tensor) for value in batch.values()):
            raise TypeError("SAM3 processor returned a non-tensor model input")
        if any(value.device.type != "cpu" for value in batch.values()):
            raise RuntimeError("Preprocessing must return CPU tensors for transfer timing")
        return batch

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Run the complete native detector/presence/mask forward on the GPU."""
        with torch.inference_mode():
            return self.model(**batch, return_dict=True)

    def postprocess(self, raw: Any, batch: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """Use native box scoring and preserve the separate presence sigmoid."""
        count = int(batch["pixel_values"].shape[0])
        if count != len(self._queries):
            raise ValueError("Postprocessing does not match the latest prepared batch")
        if raw.presence_logits is None or raw.presence_logits.numel() != count:
            raise ValueError("Expected one presence logit per image/query")
        with torch.inference_mode():
            native = self.processor.post_process_object_detection(
                raw, threshold=self.threshold, target_sizes=[(256, 256)] * count
            )
            # Sigmoid before conversion matches the native processor's scoring
            # precision, including when the forward output is float16.
            presence = raw.presence_logits.sigmoid().reshape(-1).float().cpu().tolist()
            presence_logits = raw.presence_logits.reshape(-1).float().cpu().tolist()

        results = []
        for query, output, probability, logit in zip(
            self._queries, native, presence, presence_logits, strict=True
        ):
            boxes = output["boxes"].detach().float().cpu().tolist()
            scores = output["scores"].detach().float().cpu().tolist()
            predictions = [
                {"box": box, "score": score, "text_label": query}
                for box, score in zip(boxes, scores, strict=True)
            ]
            results.append(
                {
                    "predictions": predictions,
                    "detector_proxy": max(scores, default=0.0),
                    "presence_probability": probability,
                    "presence_logit": logit,
                    "score_kind": "box_confidence_with_separate_uncalibrated_presence_sigmoid",
                    "query": query,
                }
            )
        return results
