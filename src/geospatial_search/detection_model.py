"""Local LLMDet Large inference on native embedding chips, under caller-owned GPU locks."""

from __future__ import annotations

import argparse
import hashlib
import math
from copy import deepcopy
from pathlib import Path

MODEL_ID = "iSEE-Laboratory/llmdet_large"
MODEL_REVISION = "bec37f296f05b22f6c6b39bc05a6c611239f4e31"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "model-cache" / "llmdet"
DEFAULT_BOX_THRESHOLD = 0.30
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_NMS_IOU = 0.50
MAX_BATCH_SIZE = 2
MAX_BOXES = 50
CHIP_SIZE = 256
EDGE_TOLERANCE = 2
CONTAINMENT_COVERAGE = 0.90
CONTAINMENT_AREA_RATIO = 0.25
SCORE_KIND = "maximum_retained_box_score_not_presence_probability"

_FILES = {
    "config.json": "7669f371f9ae2d642881ecd7b1769095384439c8a9e01b5f958eaed298919cb2",
    "model.safetensors": "6799bea5a4fe67ddda9ba55cf46cff06ae82653f3415e2d7d08428e02f3a0766",
    "preprocessor_config.json": "c17fd68afb1f124bfb87a494d409925eec4201b3487c8a694dde64d9ce7109a3",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer_config.json": "03070b03f452696da3d89489fa8a4e53730ab74ed2ae466d53923ab5376ce7b5",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def resolve_weights(model_dir=None, *, offline=True, cache_dir=None):
    """Resolve and verify the pinned local snapshot without importing torch."""
    if model_dir is None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            model_dir = snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=list(_FILES),
                cache_dir=DEFAULT_CACHE if cache_dir is None else cache_dir,
                local_files_only=offline,
            )
        except LocalEntryNotFoundError as exc:
            raise FileNotFoundError(
                "LLMDet weights are not cached. Run python -m geospatial_search.detection_model "
                "--download from the project's GPU environment."
            ) from exc
    path = Path(model_dir).resolve()
    for name, expected in _FILES.items():
        try:
            with (path / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Pinned LLMDet snapshot is incomplete: missing {name}"
            ) from exc
        if actual != expected:
            raise ValueError(f"Pinned LLMDet checkpoint checksum differs: {name}")
    return path


def _restore_heads(model, checkpoint):
    """Restore independent decoder heads shadowed by Transformers 5.17 weight tying.

    Issue 49529 documents the mismatch with the checkpoint's nonshared heads.
    This restores checkpoint values before transferring the model to the GPU.
    """
    import torch
    from safetensors import safe_open
    from torch import nn

    config = model.config
    flags = ("decoder_bbox_embed_share", "decoder_cls_embed_share")
    layers = config.decoder_layers
    if (
        config.model_type != "mm-grounding-dino"
        or any(getattr(config, flag, None) is not False for flag in flags)
        or layers != 6
    ):
        raise ValueError("Expected LLMDet Large's six nonshared decoder heads")
    decoder = model.model.decoder
    restored = {}
    count = 0
    with safe_open(str(checkpoint), framework="pt", device="cpu", backend="mmap") as weights:
        keys = set(weights.keys())
        for family in ("bbox_embed", "class_embed"):
            originals = getattr(model, family)
            if len(originals) != layers or len(getattr(decoder, family)) != layers:
                raise ValueError(f"Unexpected decoder head count in {family}")
            heads = nn.ModuleList()
            storages = set()
            for layer, original in enumerate(originals):
                # Copy individually: copying a tied ModuleList would preserve its aliases.
                head = deepcopy(original)
                state = {}
                for suffix, target in head.state_dict().items():
                    key = f"{family}.{layer}.{suffix}"
                    aliases = [alias for alias in (key, f"model.decoder.{key}") if alias in keys]
                    if not aliases:
                        raise ValueError(f"Missing decoder checkpoint tensor: {key}")
                    source = weights.get_tensor(aliases[0])
                    if len(aliases) == 2 and not torch.equal(
                        source, weights.get_tensor(aliases[1])
                    ):
                        raise ValueError(f"Conflicting decoder checkpoint aliases: {key}")
                    if source.shape != target.shape:
                        raise ValueError(f"Unexpected decoder checkpoint tensor shape: {key}")
                    state[suffix] = source.to(dtype=target.dtype)
                head.load_state_dict(state, strict=True)
                for suffix, actual in head.state_dict().items():
                    storage = actual.untyped_storage().data_ptr()
                    if storage in storages or not torch.equal(actual, state[suffix]):
                        raise ValueError(f"Decoder head restoration failed: {family}.{layer}")
                    storages.add(storage)
                heads.append(head)
                count += len(state)
            restored[family] = heads.train(originals.training)
    if count != 42:
        raise ValueError("Expected 42 tensors for the twelve LLMDet Large decoder heads")
    # No partial restoration is visible if any checkpoint validation fails.
    for family, heads in restored.items():
        setattr(model, family, heads)
        setattr(decoder, family, heads)
    return count


def _threshold(value):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("Detection thresholds must be numbers between 0 and 1")
    return float(value)


def _duplicate_box(first, second, iou_threshold):
    x1, y1, x2, y2 = first
    xx1, yy1, xx2, yy2 = second
    intersection = max(0.0, min(x2, xx2) - max(x1, xx1)) * max(0.0, min(y2, yy2) - max(y1, yy1))
    area_first = (x2 - x1) * (y2 - y1)
    area_second = (xx2 - xx1) * (yy2 - yy1)
    if intersection > iou_threshold * (area_first + area_second - intersection):
        return True
    smaller, larger = sorted((area_first, area_second))
    # Preserve distinct nested small objects inside much larger regions.
    return (
        intersection >= CONTAINMENT_COVERAGE * smaller
        and smaller >= CONTAINMENT_AREA_RATIO * larger
    )


def suppress_overlapping_boxes(predictions, nms_iou=DEFAULT_NMS_IOU):
    """Suppress overlapping and nearly contained duplicates; 1 disables both rules."""
    threshold = _threshold(nms_iou)
    ordered = sorted(predictions, key=lambda prediction: prediction["score"], reverse=True)
    if threshold == 1:
        return ordered[:MAX_BOXES]
    kept = []
    for prediction in ordered:
        if any(_duplicate_box(prediction["box"], prior["box"], threshold) for prior in kept):
            continue
        kept.append(prediction)
        if len(kept) == MAX_BOXES:
            break
    return kept


def _serialize_boxes(output, query, nms_iou=DEFAULT_NMS_IOU):
    boxes = output["boxes"].detach().float().cpu().tolist()
    scores = output["scores"].detach().float().cpu().tolist()
    if len(boxes) != len(scores):
        raise RuntimeError("LLMDet returned mismatched boxes and scores")
    # The native tokenizer can decode an empty batch as one blank string.
    # An empty box tensor is still a valid negative tile.
    labels = output.get("text_labels") if boxes else []
    if labels is None:
        labels = [query] * len(scores)
    predictions = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        if not (math.isfinite(score) and 0 <= score <= 1 and all(map(math.isfinite, box))):
            raise RuntimeError("LLMDet returned a nonfinite or invalid detection")
        x1, y1, x2, y2 = (max(0.0, min(float(CHIP_SIZE), value)) for value in box)
        if x2 > x1 and y2 > y1:
            predictions.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": score,
                    "text_label": label,
                    "touches_tile_edge": min(x1, y1, CHIP_SIZE - x2, CHIP_SIZE - y2)
                    <= EDGE_TOLERANCE,
                }
            )
    predictions = suppress_overlapping_boxes(predictions, nms_iou)
    return {
        "predictions": predictions,
        "maximum_box_score": predictions[0]["score"] if predictions else 0.0,
        "score_kind": SCORE_KIND,
    }


class LLMDetector:
    """Bounded FP16 LLMDet inference; workers must own collection and GPU locks."""

    def __init__(
        self,
        model_dir=None,
        *,
        offline=True,
        batch_size=MAX_BATCH_SIZE,
        cache_dir=None,
        box_threshold=DEFAULT_BOX_THRESHOLD,
        text_threshold=DEFAULT_TEXT_THRESHOLD,
        nms_iou=DEFAULT_NMS_IOU,
    ):
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"LLMDet batch size must be 1 to {MAX_BATCH_SIZE}")
        self.batch_size = batch_size
        self.box_threshold = _threshold(box_threshold)
        self.text_threshold = _threshold(text_threshold)
        self.nms_iou = _threshold(nms_iou)

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPU-enabled PyTorch is required for LLMDet; no CPU fallback")
        try:
            import safetensors  # noqa: F401 - checked before allocating model weights
            from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install requirements-detection.txt in the GPU environment") from exc

        snapshot = resolve_weights(model_dir, offline=offline, cache_dir=cache_dir)
        local = {"local_files_only": True, "trust_remote_code": False}
        config = AutoConfig.from_pretrained(str(snapshot), **local)
        config.disable_custom_kernels = True
        processor = AutoProcessor.from_pretrained(str(snapshot), **local)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(snapshot),
            config=config,
            dtype=torch.float16,
            use_kernels=False,
            use_safetensors=True,
            disable_mmap=True,
            **local,
        ).eval()
        restored_tensors = _restore_heads(model, snapshot / "model.safetensors")
        self._model = model.to(device="cuda")
        self._processor = processor
        self.info = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dtype": "float16",
            "restored_decoder_tensors": restored_tensors,
            "max_boxes_per_chip": MAX_BOXES,
            "score_kind": SCORE_KIND,
        }

    def close(self):
        """Release this detector's references; the worker owns GPU cache disposal."""
        self._model = None
        self._processor = None

    def predict(self, images, query, *, box_threshold=None, text_threshold=None, nms_iou=None):
        """Detect one arbitrary text query, returning native chip-relative xyxy boxes."""
        import torch

        if self._model is None:
            raise RuntimeError("LLMDet is closed")
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise ValueError("Enter a detection phrase of 1 to 200 characters")
        if not 1 <= len(images) <= self.batch_size:
            raise ValueError(f"Provide 1 to {self.batch_size} detector images")
        if any(image.mode != "RGB" or image.size != (CHIP_SIZE, CHIP_SIZE) for image in images):
            raise ValueError("LLMDet requires native 256 x 256 RGB chips")
        box_threshold = self.box_threshold if box_threshold is None else _threshold(box_threshold)
        text_threshold = (
            self.text_threshold if text_threshold is None else _threshold(text_threshold)
        )
        nms_iou = self.nms_iou if nms_iou is None else _threshold(nms_iou)
        # A period-terminated string per image preserves per-image prompts in the DINO processor.
        prompt = query.strip().lower().rstrip(".") + "."
        with torch.inference_mode():
            encoded = self._processor(
                images=images, text=[prompt] * len(images), padding=True, return_tensors="pt"
            )
            # Keep integer tokens and masks intact. No pinned transfers on Windows ROCm.
            inputs = {
                key: tensor.to(
                    "cuda", dtype=torch.float16 if tensor.is_floating_point() else tensor.dtype
                )
                for key, tensor in encoded.items()
            }
            raw = self._model(**inputs, return_dict=True)
            outputs = self._processor.post_process_grounded_object_detection(
                raw,
                input_ids=inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(CHIP_SIZE, CHIP_SIZE)] * len(images),
            )
        if len(outputs) != len(images):
            raise RuntimeError("LLMDet returned the wrong number of images")
        return [_serialize_boxes(output, query, nms_iou) for output in outputs]


def main():
    parser = argparse.ArgumentParser(description="Download and verify pinned LLMDet weights")
    parser.add_argument("--download", action="store_true", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    print(resolve_weights(offline=False, cache_dir=args.cache_dir))


if __name__ == "__main__":
    main()
