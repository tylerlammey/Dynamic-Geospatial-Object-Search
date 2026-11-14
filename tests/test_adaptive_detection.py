"""Synthetic coverage for the detector's bounded retrieval feedback."""

import unittest
from dataclasses import asdict
from types import SimpleNamespace

from geospatial_search.adaptive_detection import (
    DetectionOptions,
    adaptive_search,
    match_identity,
    validate_options,
)

BASE = SimpleNamespace(dataset_id="base")
SUPPLEMENT = SimpleNamespace(dataset_id="supplement")


def match(row, score=0.5, dataset=BASE):
    return (dataset, "shared-shard", row, score)


def prediction(positive, score=0.55):
    return {
        "predictions": (
            [{"box": [0.0, 4.0, 12.0, 14.0], "score": score, "text_label": "boat"}]
            if positive
            else []
        ),
        "maximum_box_score": score if positive else 0.0,
        "score_kind": "test_box_score",
    }


class CandidatePool:
    def __init__(self, candidates, accepted):
        self.candidates = candidates
        self.accepted = accepted
        self.fetches = []
        self.batches = []
        self.details = {"backend": "federated", "search_seconds": 0.5, "approximate": True}

    def fetch(self, count, seen):
        self.fetches.append((count, set(seen)))
        return [m for m in self.candidates if match_identity(m) not in seen][:count], self.details

    def evaluate(self, matches):
        keys = [match_identity(m) for m in matches]
        self.batches.append(keys)
        return [prediction(key in self.accepted) for key in keys]

    def run(self, top_k, options=DetectionOptions(), **kwargs):
        return adaptive_search(self.fetch, self.evaluate, top_k=top_k, options=options, **kwargs)


class AdaptiveDetectionTests(unittest.TestCase):
    def test_fetches_tail_after_negative_tiles_and_uses_qualified_exclusions(self):
        candidates = [
            match(0, 0.9),
            match(1, 0.8),
            match(0, 0.7, SUPPLEMENT),
            match(2, 0.6),
            match(3, 0.5),
        ]
        accepted = {match_identity(candidates[index]) for index in (1, 2, 3)}
        pool = CandidatePool(candidates, accepted)
        updates = []
        results, detections, search, stats = pool.run(
            1, DetectionOptions(fetch_size=3), progress=updates.append
        )
        self.assertEqual(results, [candidates[index] for index in (1, 2, 3)])
        self.assertEqual(set(detections), accepted)
        self.assertEqual(pool.fetches[1], (3, set(map(match_identity, candidates[:3]))))
        self.assertEqual(search["search_seconds"], 1.5)
        self.assertTrue(search["approximate"])
        self.assertEqual(stats["recent_detection_percent"], 60)
        self.assertEqual(stats["stop_reason"], "no_new_candidates")
        self.assertEqual(stats["box_count"], 3)
        self.assertEqual(updates[0]["evaluated_candidates"], 0)
        self.assertIsNone(updates[0]["recent_detection_rate"])
        self.assertEqual(updates[-1]["evaluated_candidates"], 5)
        self.assertEqual(updates[-1]["accepted_candidates"], 3)
        self.assertEqual(updates[-1]["max_candidates"], 256)

    def test_all_negatives_end_after_minimum_depth_with_a_full_window(self):
        candidates = [match(row) for row in range(100)]
        pool = CandidatePool(candidates, set())
        results, _, _, stats = pool.run(1)
        self.assertEqual(results, [])
        self.assertEqual([n for n, _ in pool.fetches], [32, 32])
        self.assertEqual(stats["recent_evaluated"], 32)
        self.assertEqual(stats["recent_detection_rate"], 0.0)
        self.assertEqual(stats["stop_reason"], "low_detection_yield")
        self.assertEqual(stats["evaluated"], 64)

    def test_feedback_uses_only_last_n_evaluations_and_stops_inside_page(self):
        candidates = [match(row) for row in range(100)]
        pool = CandidatePool(candidates, {match_identity(m) for m in candidates[:12]})
        matches, _, _, stats = pool.run(
            4,
            DetectionOptions(fetch_size=8, window_size=4, min_candidates=8, max_candidates=32),
        )
        # After 16 checked tiles, 75% overall are positive, but the last four
        # are empty. The controller stops immediately with all twelve matches.
        self.assertEqual(matches, candidates[:12])
        self.assertEqual([n for n, _ in pool.fetches], [8, 8])
        self.assertEqual(stats["recent_detection_rate"], 0.0)
        self.assertEqual(stats["stop_reason"], "low_detection_yield")
        self.assertEqual(stats["evaluated"], 16)

    def test_equality_at_rate_threshold_continues_until_it_drops(self):
        candidates = [match(row) for row in range(32)]
        pool = CandidatePool(candidates, {match_identity(candidates[7])})
        matches, _, _, stats = pool.run(
            1,
            DetectionOptions(
                fetch_size=8, window_size=4, min_candidates=8, min_detection_rate=0.25
            ),
        )
        self.assertEqual(matches, [candidates[7]])
        self.assertEqual([n for n, _ in pool.fetches], [8, 8])
        self.assertEqual(stats["evaluated"], 12)
        self.assertEqual([len(b) for b in pool.batches], [2] * 6)
        self.assertEqual(stats["stop_reason"], "low_detection_yield")

    def test_sparse_positives_extend_default_probe_until_one_leaves_recent_window(self):
        candidates = [match(row) for row in range(100)]
        accepted = [candidates[40], candidates[50]]
        pool = CandidatePool(candidates, set(map(match_identity, accepted)))
        matches, _, _, stats = pool.run(2)
        # Two positives in the recent 32 extend the search; once row 40
        # leaves the window, one out of 32 is insufficient to continue.
        self.assertEqual(matches, accepted)
        self.assertEqual([n for n, _ in pool.fetches], [32, 32, 32])
        self.assertEqual(stats["evaluated"], 74)
        self.assertEqual(stats["recent_detection_rate"], 1 / 32)
        self.assertEqual(stats["stop_reason"], "low_detection_yield")

    def test_multiple_boxes_count_as_one_positive_tile_for_feedback(self):
        candidates = [match(row) for row in range(50)]
        pool = CandidatePool(candidates, set())

        def evaluate(batch):
            detections = []
            for candidate in batch:
                detection = prediction(candidate == candidates[7])
                if detection["predictions"]:
                    detection["predictions"] *= 8
                detections.append(detection)
            return detections

        matches, detections, _, stats = adaptive_search(
            pool.fetch,
            evaluate,
            top_k=5,
            options=DetectionOptions(min_candidates=8, window_size=8, min_detection_rate=0.2),
        )
        self.assertEqual(matches, [candidates[7]])
        self.assertEqual(len(detections[match_identity(candidates[7])]["predictions"]), 8)
        self.assertEqual(stats["box_count"], 8)
        self.assertEqual(stats["evaluated"], 8)
        self.assertEqual(stats["recent_detection_rate"], 1 / 8)
        self.assertEqual(stats["stop_reason"], "low_detection_yield")

    def test_waits_for_full_window_if_minimum_is_smaller(self):
        pool = CandidatePool([match(row) for row in range(100)], set())
        _, _, _, stats = pool.run(
            1, DetectionOptions(fetch_size=20, window_size=6, min_candidates=1)
        )
        self.assertEqual(stats["evaluated"], 6)
        self.assertEqual(stats["recent_evaluated"], 6)

    def test_positive_tail_ignores_top_k_and_runs_to_candidate_cap(self):
        candidates = [match(row) for row in range(100)]
        pool = CandidatePool(candidates, set(map(match_identity, candidates)))
        results, _, _, stats = pool.run(2, DetectionOptions(max_candidates=71))
        self.assertEqual([n for n, _ in pool.fetches], [32, 32, 7])
        self.assertEqual(results, candidates[:71])
        self.assertEqual(stats["evaluated"], 71)
        self.assertEqual(stats["recent_detection_rate"], 1.0)
        self.assertEqual(stats["stop_reason"], "candidate_limit")

    def test_preserves_cosine_sort_when_ann_expansion_finds_a_better_match(self):
        first = [match(0, 0.7), match(1, 0.6)]
        second = [match(2, 0.8)]
        details = {"search_seconds": 0.5}
        response = iter([(first, {**details, "approximate": True}), (second, details), ([], {})])
        results, _, search, _ = adaptive_search(
            lambda count, seen: next(response),
            lambda batch: [prediction(row != 1) for _, _, row, _ in batch],
            top_k=2,
            options=DetectionOptions(),
        )
        self.assertEqual(results, [second[0], first[0]])
        self.assertTrue(search["approximate"])

    def test_orders_by_box_confidence_then_similarity_then_retrieval_order(self):
        candidates = [
            match(5, 0.99),
            match(2, 0.85),
            match(9, 0.81),
            match(1, 0.81),
            match(0, 0.60),
        ]
        scores = {5: 0.33, 2: 0.5, 9: 0.5, 1: 0.5, 0: 0.95}
        pool = CandidatePool(candidates, set())
        evaluation_order = []

        def evaluate(batch):
            evaluation_order.extend(batch)
            return [prediction(True, scores[candidate[2]]) for candidate in batch]

        results, detections, _, stats = adaptive_search(
            pool.fetch,
            evaluate,
            top_k=2,
            options=DetectionOptions(fetch_size=3, max_candidates=5),
        )
        self.assertEqual(evaluation_order, candidates)
        self.assertEqual(results, [candidates[index] for index in (4, 1, 2, 3, 0)])
        self.assertEqual([m[3] for m in results], [0.60, 0.85, 0.81, 0.81, 0.99])
        self.assertEqual(detections[match_identity(results[0])]["maximum_box_score"], 0.95)
        self.assertEqual(stats["rounds"], 2)
        self.assertEqual(stats["recent_detection_rate"], 1.0)

    def test_duplicate_only_round_stops_and_does_not_evaluate_existing_ids(self):
        excluded, fresh = match(0), match(1)
        observed = []

        def fetch(count, seen):
            observed.append(seen)
            return [excluded, fresh, fresh], {"approximate": True}

        evaluated = []

        def evaluate(batch):
            evaluated.extend(batch)
            return [prediction(False) for _ in batch]

        _, _, _, stats = adaptive_search(
            fetch,
            evaluate,
            top_k=3,
            options=DetectionOptions(),
            exclude_ids=[match_identity(excluded)],
        )
        self.assertEqual(evaluated, [fresh])
        self.assertEqual(observed[1], (match_identity(excluded), match_identity(fresh)))
        self.assertEqual(stats["rounds"], 2)
        self.assertEqual(stats["stop_reason"], "no_new_candidates")

    def test_hard_candidate_cap_below_minimum_never_expands(self):
        candidates = [match(row) for row in range(20)]
        pool = CandidatePool(candidates, set(map(match_identity, candidates)))
        _, _, _, stats = pool.run(15, DetectionOptions(max_candidates=3))
        self.assertEqual(stats["evaluated"], 3)
        self.assertEqual(stats["returned"], 3)
        self.assertEqual(stats["stop_reason"], "candidate_limit")

    def test_zero_minimum_rate_disables_yield_stop_and_keeps_pages_bounded(self):
        candidates = [match(row) for row in range(1040)]
        pool = CandidatePool(candidates, set())
        _, _, _, stats = pool.run(
            5000,
            DetectionOptions(max_candidates=1040, fetch_size=512, min_detection_rate=0),
        )
        self.assertEqual([n for n, _ in pool.fetches], [512, 512, 16])
        self.assertEqual(stats["evaluated"], 1040)

    def test_empty_pool_never_loads_detector_and_has_no_rate(self):
        pool = CandidatePool([], set())
        results, _, _, stats = pool.run(4)
        self.assertEqual(results, [])
        self.assertEqual(pool.batches, [])
        self.assertEqual(stats["rounds"], 1)
        self.assertIsNone(stats["recent_detection_rate"])
        self.assertEqual(stats["stop_reason"], "no_new_candidates")

    def test_disabled_returns_unfiltered_matches_without_a_detector(self):
        candidates = [match(row) for row in range(20)]
        pool = CandidatePool(candidates, set())
        matches, by_identity, details, stats = pool.run(
            10, DetectionOptions(enabled=False, max_candidates=1)
        )
        self.assertEqual(matches, candidates[:10])
        self.assertEqual(pool.batches, [])
        self.assertEqual(by_identity, {})
        self.assertEqual(details, pool.details)
        self.assertFalse(stats["enabled"])

    def test_invalid_detector_outputs_fail_instead_of_dropping_tiles(self):
        positive = prediction(True)
        box = positive["predictions"][0]
        bad_outputs = [
            [],
            [{"predictions": []}],
            [{**positive, "maximum_box_score": float("nan")}],
            [{**positive, "maximum_box_score": 0.9}],
            [{**prediction(False), "maximum_box_score": 0.2}],
            [{**positive, "predictions": [{**box, "box": [30, 0, 10, 20]}]}],
            [{**positive, "predictions": [{**box, "box": [0, 0, 300, 20]}]}],
            [{**positive, "predictions": [{**box, "score": -1.0}]}],
        ]
        for output in bad_outputs:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "Detector"):
                adaptive_search(
                    lambda count, seen: ([match(0)], {}),
                    lambda batch: output,
                    top_k=1,
                    options=DetectionOptions(),
                )

    def test_retriever_cannot_overrun_requested_candidate_budget(self):
        with self.assertRaisesRegex(ValueError, "more than the requested"):
            adaptive_search(
                lambda count, seen: ([match(0), match(1)], {}),
                lambda batch: self.fail("Detector should not see excess candidates"),
                top_k=25,
                options=DetectionOptions(max_candidates=1),
            )

    def test_validates_option_boundaries_and_round_trip_to_worker_request(self):
        cases = [
            {"enabled": 1},
            {"box_threshold": True},
            {"box_threshold": float("nan")},
            {"text_threshold": -0.01},
            {"batch_size": 3},
            {"fetch_size": 0},
            {"window_size": 0},
            {"min_candidates": 0},
            {"min_detection_rate": float("inf")},
            {"min_detection_rate": 1.01},
            {"max_candidates": 5001},
            {"nms_iou": 0.25, "unsupported": True},
            {"nms_iou": 1.01},
            {"nms_iou": False},
        ]
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_options(fields)
        options = validate_options(
            {"box_threshold": 0, "nms_iou": 1, "max_candidates": 3}, top_k=10
        )
        self.assertEqual(options.max_candidates, 3)
        self.assertEqual(options.nms_iou, 1)
        self.assertEqual(validate_options(asdict(options)), options)
        self.assertEqual(validate_options(), DetectionOptions())


if __name__ == "__main__":
    unittest.main()
