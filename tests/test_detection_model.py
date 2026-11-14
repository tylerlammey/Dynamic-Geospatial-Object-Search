"""Exercise the pinned detector's CPU-side boundaries without real models or GPU."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

from geospatial_search import detection_model as detection


class _TiedHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="mm-grounding-dino",
            decoder_layers=6,
            decoder_bbox_embed_share=False,
            decoder_cls_embed_share=False,
        )
        bbox = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 4))
        classification = nn.Module()
        classification.bias = nn.Parameter(torch.zeros(()))
        self.bbox_embed = nn.ModuleList([bbox] * 6)
        self.class_embed = nn.ModuleList([classification] * 6)
        self.model = nn.Module()
        self.model.decoder = nn.Module()
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed
        self.half()


def _checkpoint(model):
    return {
        f"{prefix}.{family}.{index}.{suffix}".lstrip("."): torch.full(
            target.shape, index + 0.125, dtype=torch.float32
        )
        for family in ("bbox_embed", "class_embed")
        for index, head in enumerate(getattr(model, family))
        for prefix in ("model.decoder" if index else "",)
        for suffix, target in head.state_dict().items()
    }


class RestoreHeadsTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "temp" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(prefix="nys-detector-", dir=root)
        self.addCleanup(directory.cleanup)
        self.snapshot = Path(directory.name)
        self.model = _TiedHeads()
        self.weights = self.snapshot / "model.safetensors"

    def test_unties_layers_and_preserves_checkpoint_values(self):
        tensors = _checkpoint(self.model)
        save_file(tensors, str(self.weights))

        self.assertEqual(detection._restore_heads(self.model, self.weights), 42)

        bbox = self.model.bbox_embed
        classification = self.model.class_embed
        self.assertIs(bbox, self.model.model.decoder.bbox_embed)
        self.assertIs(classification, self.model.model.decoder.class_embed)
        self.assertEqual(classification[0].bias.item(), 0.125)
        self.assertEqual(classification[5].bias.item(), 5.125)
        torch.testing.assert_close(
            bbox[3][2].weight, tensors["model.decoder.bbox_embed.3.2.weight"].half()
        )
        with torch.no_grad():
            bbox[0][2].weight.fill_(9)
        self.assertEqual(bbox[1][2].weight[0, 0].item(), 1.125)

    def test_missing_late_tensor_does_not_partially_repair_model(self):
        tensors = _checkpoint(self.model)
        del tensors["model.decoder.class_embed.5.bias"]
        save_file(tensors, str(self.weights))
        tied_box, tied_class = self.model.bbox_embed, self.model.class_embed

        with self.assertRaisesRegex(ValueError, "Missing decoder checkpoint tensor"):
            detection._restore_heads(self.model, self.weights)

        self.assertIs(self.model.bbox_embed, tied_box)
        self.assertIs(self.model.class_embed, tied_class)
        self.assertIs(tied_box[0], tied_box[5])
        self.assertEqual(tied_class[5].bias.item(), 0)

    def test_explicit_snapshot_requires_exact_checkpoint(self):
        payload = b"known synthetic checkpoint"
        with patch.object(
            detection, "_FILES", {self.weights.name: hashlib.sha256(payload).hexdigest()}
        ):
            with self.assertRaisesRegex(FileNotFoundError, "snapshot is incomplete"):
                detection.resolve_weights(self.snapshot)
            self.weights.write_bytes(payload)
            self.assertEqual(detection.resolve_weights(self.snapshot), self.snapshot.resolve())
            self.weights.write_bytes(payload[::-1])
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                detection.resolve_weights(self.snapshot)


class DetectorOutputTests(unittest.TestCase):
    def test_clips_native_box_bounds_and_bounds_sorted_results(self):
        low = [[1, 1, 5, 5]] * 55
        output = {
            "boxes": torch.tensor(low + [[300, 0, 500, 10], [-1, 12, 300, 30]]),
            "scores": torch.tensor([0.25] * len(low) + [1.0, 0.875]),
            "text_labels": [str(i) for i in range(len(low))] + ["invalid", ""],
        }

        result = detection._serialize_boxes(output, "boats", nms_iou=1)

        boxes = result["predictions"]
        self.assertEqual(len(boxes), 50)
        self.assertEqual(result["maximum_box_score"], 0.875)
        self.assertEqual(
            boxes[0],
            {"box": [0, 12, 256, 30], "score": 0.875, "text_label": "", "touches_tile_edge": True},
        )
        self.assertEqual([box["text_label"] for box in boxes[1:]], [str(i) for i in range(49)])

    def test_nms_uses_clipped_bounds_and_keeps_stable_score_ties(self):
        output = {
            "boxes": torch.tensor([[-0.5, 0, 10, 10], [-12, 0, 10, 10], [12, 0, 22, 10]]),
            "scores": torch.tensor([0.75, 0.75, 0.875]),
            "text_labels": ["first", "second", "disjoint"],
        }

        result = detection._serialize_boxes(output, "boats")

        self.assertEqual(result["maximum_box_score"], 0.875)
        self.assertEqual(
            [box["text_label"] for box in result["predictions"]], ["disjoint", "first"]
        )
        self.assertEqual(result["predictions"][1]["box"], [0, 0, 10, 10])

    def test_nms_retains_iou_equal_to_cutoff_and_limits_after_suppression(self):
        boxes = [
            {"box": [0, 0, 3, 10], "score": 0.75},
            {"box": [1, 0, 4, 10], "score": 0.50},
        ]
        self.assertEqual(detection.suppress_overlapping_boxes(boxes, 0.50), boxes)
        self.assertEqual(detection.suppress_overlapping_boxes(boxes, 0.49), boxes[:1])
        duplicates = [boxes[0]] * 60 + boxes[1:]
        self.assertEqual(detection.suppress_overlapping_boxes(duplicates), boxes)
        self.assertEqual(detection.suppress_overlapping_boxes(duplicates, 1), duplicates[:50])

    def test_containment_suppresses_duplicate_but_keeps_smaller_nested_object(self):
        tighter_track = {"box": [30, 35, 210, 125], "score": 0.465}
        oversized_track = {"box": [1, 1, 249, 235], "score": 0.379}
        small_object = {"box": [37, 40, 57, 65], "score": 0.350}

        predictions = [oversized_track, small_object, tighter_track]
        self.assertEqual(
            detection.suppress_overlapping_boxes(predictions), [tighter_track, small_object]
        )
        self.assertEqual(
            detection.suppress_overlapping_boxes(predictions, 1),
            [tighter_track, oversized_track, small_object],
        )

    def test_edge_metadata_marks_near_boundary_and_is_false_for_interior_boxes(self):
        output = {
            "boxes": torch.tensor([[2, 10, 10, 20], [30, 30, 254, 250], [30, 8, 35, 25]]),
            "scores": torch.tensor([0.875, 0.75, 0.5]),
        }
        result = detection._serialize_boxes(output, "boat")
        self.assertEqual(
            [box["touches_tile_edge"] for box in result["predictions"]], [True, True, False]
        )

    def test_rejects_invalid_nms_iou_before_model_load(self):
        for value in (-0.01, 1.01, True, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "thresholds"):
                detection.LLMDetector(nms_iou=value)

    def test_empty_tile_ignores_tokenizer_placeholder_label(self):
        output = {"boxes": torch.empty(0, 4), "scores": torch.empty(0), "text_labels": [""]}
        self.assertEqual(detection._serialize_boxes(output, "airplane")["predictions"], [])

    def test_empty_and_nonfinite_predictions(self):
        output = {"boxes": torch.tensor([[300, 1, 310, 10]]), "scores": torch.tensor([0.75])}
        result = detection._serialize_boxes(output, "boat")
        self.assertEqual(result["predictions"], [])
        self.assertEqual(result["maximum_box_score"], 0.0)
        output["boxes"] = torch.tensor([[float("nan"), 1, 10, 10]])
        with self.assertRaisesRegex(RuntimeError, "invalid detection"):
            detection._serialize_boxes(output, "boat")

    def test_rejects_missing_gpu_and_unmeasured_batch_size_before_model_load(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "batch size"):
                detection.LLMDetector(batch_size=4)
            with patch.object(detection, "resolve_weights") as resolve:
                with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                    detection.LLMDetector()
                resolve.assert_not_called()

    def test_input_and_override_validation_before_inference(self):
        detector = detection.LLMDetector.__new__(detection.LLMDetector)
        detector.batch_size = 2
        detector.box_threshold = 0.10
        detector.text_threshold = 0.25
        detector.nms_iou = 0.50
        detector._model = object()
        with self.assertRaisesRegex(ValueError, "native 256 x 256 RGB"):
            detector.predict([Image.new("RGB", (64, 64))], "boat")
        with self.assertRaisesRegex(ValueError, "thresholds"):
            detector.predict([Image.new("RGB", (256, 256))], "boat", box_threshold=float("nan"))
        with self.assertRaisesRegex(ValueError, "thresholds"):
            detector.predict([Image.new("RGB", (256, 256))], "boat", nms_iou=1.1)
        with self.assertRaisesRegex(ValueError, "1 to 2 detector images"):
            detector.predict([], "boat")


if __name__ == "__main__":
    unittest.main()
