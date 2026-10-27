"""Synthetic metrics regressions; no image data, model, or GPU required."""

import unittest

import numpy as np
from metrics import (
    apply_platt,
    box_iou,
    break_even,
    clip_prediction_boxes,
    evaluate_decisions,
    evaluate_detections,
    fit_platt,
    threshold_sweep,
    wilson_lower,
)


def decision(probability, label, query="car", **kwargs):
    return dict(probability=probability, label=label, query=query, **kwargs)


def detection(sample_id, boxes, predictions=(), query="car", exhaustive=True):
    return dict(
        sample_id=sample_id,
        query=query,
        boxes=boxes,
        predictions=[dict(box=box, score=score) for box, score in predictions],
        exhaustive=exhaustive,
    )


class DecisionMetricsTests(unittest.TestCase):
    def test_false_negatives_threshold_boundary_and_unlabeled(self):
        rows = [
            decision(0.8, 1),
            decision(0.2, 1),
            decision(0.5, 0),
            decision(0.1, 0),
            decision(0.99, None),
        ]
        result = evaluate_decisions(rows)["overall"]
        self.assertEqual((result["tp"], result["fn"], result["fp"], result["tn"]), (1, 1, 1, 1))
        self.assertEqual(
            (result["precision"], result["recall"], result["false_negative_rate"]), (0.5, 0.5, 0.5)
        )
        self.assertEqual((result["labeled"], result["unlabeled"]), (4, 1))
        self.assertAlmostEqual(result["brier"], 0.235)

    def test_per_query_and_no_positives(self):
        result = evaluate_decisions([decision(0.8, 1), decision(0.2, 0, "pool")])
        self.assertEqual(result["by_query"]["car"]["recall"], 1)
        self.assertIsNone(result["by_query"]["pool"]["recall"])
        self.assertIsNone(result["by_query"]["pool"]["precision"])
        self.assertIsNone(evaluate_decisions([decision(0.2, None)])["overall"]["brier"])

    def test_calibration_bins_and_extreme_probabilities(self):
        result = evaluate_decisions([decision(0, 0), decision(1, 1)], bins=2)["overall"]
        self.assertEqual(result["brier"], 0)
        self.assertEqual(result["ece"], 0)
        self.assertTrue(np.isfinite(result["nll"]))
        result = evaluate_decisions([decision(0.5, 1), decision(0.5, 0)], bins=2)["overall"]
        self.assertEqual(result["ece"], 0)
        self.assertAlmostEqual(result["nll"], np.log(2))

    def test_invalid_probabilities_and_thresholds(self):
        for value in (np.nan, np.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                evaluate_decisions([decision(value, 1)])
        with self.assertRaises(ValueError):
            evaluate_decisions([decision(0.5, 2)])
        with self.assertRaises(ValueError):
            evaluate_decisions([], bins=0)

    def test_sweep_includes_false_negatives(self):
        results = threshold_sweep([decision(0.2, 1), decision(0.1, 0)], [0, 0.15, 0.5])
        self.assertEqual([row["recall"] for row in results], [1, 1, 0])
        self.assertEqual([row["fp"] for row in results], [1, 0, 0])

    def test_wilson_is_not_perfect_recall_claim(self):
        self.assertLess(wilson_lower(10, 10), 0.73)
        self.assertGreater(wilson_lower(1000, 1000), 0.996)
        self.assertIsNone(wilson_lower(0, 0))

    def calibration_rows(self):
        rows = [
            decision(
                0.9,
                index % 2,
                split="calibration",
                sample_id=f"c{index}",
                group_id="calibration-source",
            )
            for index in range(40)
        ]
        rows += [decision(0.9, 1, split="test", sample_id="test", group_id="test-source")]
        return rows

    def test_platt_corrects_overconfidence_without_using_test_labels(self):
        rows = self.calibration_rows()
        calibration = fit_platt(rows)
        self.assertTrue(calibration["converged"])
        self.assertAlmostEqual(apply_platt([0.9], calibration)[0], 0.5, places=3)
        rows[-1]["label"] = 0
        self.assertEqual(calibration, fit_platt(rows))

    def test_platt_rejects_image_and_spatial_leakage_and_missing_groups(self):
        for key in ("sample_id", "group_id"):
            rows = self.calibration_rows()
            rows[-1][key] = rows[0][key]
            with self.assertRaisesRegex(ValueError, "leakage"):
                fit_platt(rows)
        rows = self.calibration_rows()
        del rows[-1]["group_id"]
        with self.assertRaisesRegex(ValueError, "group_id"):
            fit_platt(rows)
        with self.assertRaisesRegex(ValueError, "both required"):
            fit_platt(rows[:-1])

    def test_platt_rejects_same_tile_with_different_queries_and_sample_ids(self):
        rows = self.calibration_rows()
        rows[0].update(tile_id="same-native-tile", query="car")
        rows[-1].update(tile_id="same-native-tile", query="boat")
        self.assertNotEqual(rows[0]["sample_id"], rows[-1]["sample_id"])
        self.assertNotEqual(rows[0]["group_id"], rows[-1]["group_id"])
        with self.assertRaisesRegex(ValueError, "overlapping tile_id/image identity"):
            fit_platt(rows)

    def test_break_even_is_strict_and_rejects_nonfinite(self):
        self.assertFalse(break_even(10, 100, 0.9)["faster"])
        self.assertTrue(break_even(10, 100, 0.8)["faster"])
        self.assertFalse(break_even(200, 100, 0)["faster"])
        self.assertAlmostEqual(break_even(10, 100, 0.8)["speedup"], 100 / 90)
        with self.assertRaises(ValueError):
            break_even(float("nan"), 100, 0.5)


class DetectionMetricsTests(unittest.TestCase):
    def test_clip_predictions_preserves_raw_ground_truth_metadata_and_native_dimensions(self):
        row = detection(
            "edge",
            [[0, 5, 248, 244]],
            [
                ([-0.6875, 5.75, 248, 243.625], 0.9),
                ([260, 1, 280, 20], 0.8),
            ],
        )
        row["predictions"][0]["text_label"] = "edge object"
        clipped = clip_prediction_boxes([row])
        self.assertEqual(
            clipped[0]["predictions"],
            [{"box": [0, 5.75, 248, 243.625], "score": 0.9, "text_label": "edge object"}],
        )
        self.assertEqual(clipped[0]["boxes"], row["boxes"])
        self.assertEqual(row["predictions"][0]["box"][0], -0.6875)
        self.assertEqual(len(row["predictions"]), 2)
        self.assertEqual(evaluate_detections(clipped)["tp_at_score_threshold"], 1)
        clipped[0]["boxes"][0][0] = 99
        self.assertEqual(row["boxes"][0][0], 0)
        other = detection("custom", [], [([-5, -5, 100, 100], 0.7)])
        other.update(width=80, height=40)
        self.assertEqual(clip_prediction_boxes([other])[0]["predictions"][0]["box"], [0, 0, 80, 40])

    def test_clip_predictions_rejects_invalid_inputs_and_discards_empty_boundary_boxes(self):
        for box in ([0, 1, 2], [0, 0, np.nan, 1], [0, 0, np.inf, 1], [10, 0, 5, 1]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                clip_prediction_boxes([detection("invalid", [], [(box, 0.5)])])
        for score in (-0.1, 1.1, np.nan):
            with self.subTest(score=score), self.assertRaises(ValueError):
                clip_prediction_boxes([detection("invalid", [], [([0, 0, 1, 1], score)])])
        row = detection("empty", [], [([-10, 0, 0, 10], 0.5), ([1, 1, 1, 2], 0.5)])
        self.assertEqual(clip_prediction_boxes([row])[0]["predictions"], [])
        row["width"] = float("nan")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            clip_prediction_boxes([row])

    def test_iou_pixel_edges_and_empty(self):
        overlaps = box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10]])
        np.testing.assert_allclose(overlaps, [[1, 1 / 3]])
        self.assertEqual(box_iou([], [[0, 0, 1, 1]]).shape, (0, 1))

    def test_perfect_detection_and_small_area_boundary(self):
        small, large = [0, 0, 4, 8], [40, 40, 72, 72]
        result = evaluate_detections([detection("x", [small, large], [(small, 0.9), (large, 0.8)])])
        self.assertAlmostEqual(result["ap50"], 1)
        self.assertAlmostEqual(result["ap50_95"], 1)
        self.assertEqual(result["small_objects"], 1)
        self.assertEqual(result["small_recall50_at_score_threshold"], 1)

    def test_duplicate_predictions_and_false_positive_rank_penalty(self):
        box = [0, 0, 8, 8]
        false_box = [100, 100, 108, 108]
        result = evaluate_detections(
            [detection("x", [box], [(false_box, 0.99), (box, 0.9), (box, 0.8)])]
        )
        self.assertAlmostEqual(result["ap50"], 0.5)
        self.assertEqual(result["tp_at_score_threshold"], 1)
        self.assertEqual(result["fp_at_score_threshold"], 2)
        self.assertEqual(result["recall50_at_score_threshold"], 1)
        self.assertAlmostEqual(result["precision50_at_score_threshold"], 1 / 3)

    def test_missing_small_object_is_false_negative(self):
        small, large = [0, 0, 4, 8], [40, 40, 100, 100]
        result = evaluate_detections([detection("x", [small, large], [(large, 0.9)])])
        self.assertEqual(result["recall50_at_score_threshold"], 0.5)
        self.assertEqual(result["small_recall50_at_score_threshold"], 0)
        self.assertAlmostEqual(result["ap50"], 51 / 101)

    def test_empty_and_nonexhaustive_do_not_invent_accuracy(self):
        box = [0, 0, 8, 8]
        result = evaluate_detections([detection("unknown", [box], [(box, 0.9)], exhaustive=False)])
        self.assertIsNone(result["ap50"])
        self.assertEqual(result["excluded_nonexhaustive_records"], 1)
        result = evaluate_detections([detection("negative", [], [(box, 0.9)])])
        self.assertIsNone(result["ap50"])
        self.assertEqual(result["fp_at_score_threshold"], 1)
        self.assertEqual(result["precision50_at_score_threshold"], 0)
        result = evaluate_detections([detection("positive", [box])])
        self.assertEqual(result["ap50"], 0)
        self.assertEqual(result["recall50_at_score_threshold"], 0)

    def test_query_macro_average_does_not_weight_frequent_objects(self):
        box = [0, 0, 8, 8]
        rows = [
            detection("a", [box], [(box, 0.9)], query="pool"),
            detection("b", [box], query="car"),
            detection("c", [box], query="car"),
        ]
        result = evaluate_detections(rows)
        self.assertEqual(result["ap50"], 0.5)
        self.assertAlmostEqual(result["recall50_at_score_threshold"], 1 / 3)

    def test_invalid_native_coordinates_and_duplicate_identity(self):
        with self.assertRaisesRegex(ValueError, "native tile bounds"):
            evaluate_detections([detection("x", [[0, 0, 300, 10]])])
        row = detection("x", [[0, 0, 10, 10]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_detections([row, row])


if __name__ == "__main__":
    unittest.main()
