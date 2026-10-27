"""Standalone metrics; no pipeline, model loading, or dataset writes.

Decision records: {sample_id, tile_id (optional), query, probability, label: 0|1|None,
                   split: calibration|test, group_id: spatial/source group}.
Only labeled records contribute to accuracy metrics. A model's matching score
must not be described as calibrated merely because it lies between zero and one.

Detection records: {sample_id, query, exhaustive: bool, boxes: [[x1,y1,x2,y2]],
                    predictions: [{box: [x1,y1,x2,y2], score: float}],
                    width: 256, height: 256}.
Coordinates are native pixel edges, with exclusive right/bottom edges. Empty
ground truth means a verified negative only when exhaustive is true. AP uses
greedy one-to-one IoU matching and 101-point interpolation, macro-averaged over
queries with ground truth, at IoU .50:.05:.95, with 100 predictions per tile/query.
This is a custom COCO-style calculation, NOT official pycocotools COCO AP: crowd,
ignore regions, area-range AP, and cross-query synonym merging are unsupported.
Small-object recall counts ground-truth area <32**2 native pixels, after matching
against all objects at IoU .5 and the specified confidence threshold.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict

import numpy as np


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability/score must be finite and in [0, 1]")
    return value


def _label(value):
    if value not in (0, 1):
        raise ValueError("Labels must be 0, 1, or null")
    return int(value)


def wilson_lower(successes, total, z=1.959963984540054):
    """Two-sided 95% Wilson interval's lower bound by default (independent rows).

    Overlapping spatial tiles are correlated; this bound is descriptive, not a
    substitute for a spatial cluster bootstrap or an independent holdout.
    """
    if total < 0 or not 0 <= successes <= total or z <= 0:
        raise ValueError("Invalid binomial counts or confidence multiplier")
    if not total:
        return None
    fraction = successes / total
    denominator = 1 + z * z / total
    center = fraction + z * z / (2 * total)
    radius = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total**2))
    return max(0.0, (center - radius) / denominator)


def _decision_summary(records, threshold, bins):
    rows = [row for row in records if row.get("label") is not None]
    if not rows:
        return {
            "labeled": 0,
            "unlabeled": len(records),
            "precision": None,
            "recall": None,
            "false_negative_rate": None,
            "brier": None,
            "nll": None,
            "ece": None,
            "recall_wilson_lower_95": None,
        }
    probabilities = np.array([_probability(row["probability"]) for row in rows])
    labels = np.array([_label(row["label"]) for row in rows], dtype=bool)
    accepted = probabilities >= threshold
    tp = int(np.sum(accepted & labels))
    fp = int(np.sum(accepted & ~labels))
    fn = int(np.sum(~accepted & labels))
    tn = int(np.sum(~accepted & ~labels))
    clipped = probabilities.clip(1e-12, 1 - 1e-12)
    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    calibration_bins = []
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(np.sum(selected))
        mean_probability = float(probabilities[selected].mean()) if count else None
        observed_frequency = float(labels[selected].mean()) if count else None
        if count:
            ece += count / len(rows) * abs(mean_probability - observed_frequency)
        calibration_bins.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
            }
        )
    return {
        "labeled": len(rows),
        "unlabeled": len(records) - len(rows),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "false_negative_rate": _ratio(fn, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "accuracy": (tp + tn) / len(rows),
        "accepted_fraction_labeled": (tp + fp) / len(rows),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(clipped) + ~labels * np.log1p(-clipped))),
        "ece": ece,
        "calibration_bins": calibration_bins,
        "recall_wilson_lower_95": wilson_lower(tp, tp + fn),
    }


def evaluate_decisions(records, threshold=0.5, bins=10):
    """Evaluate one model/configuration; threshold acceptance is inclusive."""
    threshold = _probability(threshold)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    records = list(records)
    by_query = defaultdict(list)
    for row in records:
        by_query[str(row.get("query", ""))].append(row)
    return {
        "overall": _decision_summary(records, threshold, bins),
        "by_query": {
            query: _decision_summary(rows, threshold, bins)
            for query, rows in sorted(by_query.items())
        },
    }


def threshold_sweep(records, thresholds=(0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9)):
    """Describe thresholds on the supplied split; choose gates on calibration only."""
    records = list(records)
    return [evaluate_decisions(records, threshold)["overall"] for threshold in thresholds]


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -values))


def _logit(probabilities):
    values = np.asarray([_probability(value) for value in probabilities], dtype=np.float64)
    values = values.clip(1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def apply_platt(probabilities, calibration):
    """Return probabilities under a frozen affine-logit calibration."""
    return _sigmoid(calibration["scale"] * _logit(probabilities) + calibration["offset"])


def fit_platt(records, regularization=1e-3, max_iterations=100):
    """Fit calibration labels only, rejecting calibration/test image/group overlap.

    Both splits need labeled records, nonempty sample_id and group_id. Optional
    tile_id identifies the image independent of query; sample_id is its fallback.
    The calibration split must contain both classes. Assign groups before fitting:
    use source TIFF or a sufficiently large geographic block, not per-chip IDs.
    A disjoint query string alone cannot prevent overlapping-tile leakage.
    """
    if regularization <= 0 or max_iterations < 1:
        raise ValueError("Require positive regularization and iteration count")
    rows = [row for row in records if row.get("label") is not None]
    calibration_rows = [row for row in rows if row.get("split") == "calibration"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not calibration_rows or not test_rows:
        raise ValueError("Labeled calibration and test splits are both required")
    for key in ("sample_id", "group_id"):
        for row in calibration_rows + test_rows:
            if not row.get(key):
                raise ValueError(f"Explicit nonempty {key} is required for leakage checks")
        if {row[key] for row in calibration_rows} & {row[key] for row in test_rows}:
            raise ValueError(f"Calibration/test leakage: overlapping {key}")
    calibration_images = {row.get("tile_id") or row["sample_id"] for row in calibration_rows}
    test_images = {row.get("tile_id") or row["sample_id"] for row in test_rows}
    if calibration_images & test_images:
        raise ValueError("Calibration/test leakage: overlapping tile_id/image identity")
    labels = np.array([_label(row["label"]) for row in calibration_rows], dtype=float)
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration requires positive and negative labels")
    features = np.column_stack(
        (_logit([row["probability"] for row in calibration_rows]), np.ones(len(labels)))
    )
    weights = np.array([1.0, 0.0])
    penalty = np.array([regularization, regularization])

    def objective(candidate):
        logits = features @ candidate
        return float(
            np.sum(np.logaddexp(0, logits) - labels * logits) + 0.5 * np.sum(penalty * candidate**2)
        )

    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        fitted = _sigmoid(features @ weights)
        gradient = features.T @ (fitted - labels) + penalty * weights
        hessian = features.T @ (features * (fitted * (1 - fitted))[:, None]) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        old_loss = objective(weights)
        fraction = 1.0
        for _ in range(30):
            candidate = weights - fraction * step
            if objective(candidate) <= old_loss:
                break
            fraction *= 0.5
        else:
            break
        weights = candidate
        if float(np.max(np.abs(fraction * step))) < 1e-8:
            converged = True
            break
    return {
        "method": "Platt affine logit; calibration split only",
        "scale": float(weights[0]),
        "offset": float(weights[1]),
        "calibration_count": len(calibration_rows),
        "test_count": len(test_rows),
        "regularization": regularization,
        "iterations": iterations,
        "converged": converged,
    }


def break_even(decision_ms, detection_ms, accepted_fraction):
    """Serial per-tile costs at the SAME batch regime; ANN and loading excluded.

    Co-residency and batching changes can invalidate this arithmetic estimate;
    measure the serial chain before interpreting the estimate as end-to-end speed.
    """
    decision_ms, detection_ms = float(decision_ms), float(detection_ms)
    accepted_fraction = _probability(accepted_fraction)
    if not all(math.isfinite(value) for value in (decision_ms, detection_ms)):
        raise ValueError("Costs must be finite")
    if decision_ms < 0 or detection_ms <= 0:
        raise ValueError("Require decision cost >=0 and detection cost >0")
    cascade_ms = decision_ms + accepted_fraction * detection_ms
    return {
        "direct_detection_ms": detection_ms,
        "estimated_cascade_ms": cascade_ms,
        "accepted_fraction": accepted_fraction,
        "maximum_accepted_fraction_for_speedup": max(0.0, 1 - decision_ms / detection_ms),
        "speedup": detection_ms / cascade_ms if cascade_ms else None,
        "faster": cascade_ms < detection_ms,
        "note": "Strict break-even: decision_ms + accepted_fraction*detection_ms < detection_ms",
    }


def _box(value, width, height):
    box = np.asarray(value, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Boxes require four finite native xyxy coordinates")
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("Box must have positive area and remain inside native tile bounds")
    return box


def clip_prediction_boxes(records):
    """Copy detector records and clip predicted boxes to original tile boundaries.

    Invoke explicitly before strict metric computation when a native detector
    emits boxes extending beyond its image. The raw prediction artifact remains
    unchanged. Ground truth and all other metadata are copied unchanged; only
    prediction coordinates are clipped to [0,width] x [0,height] (default 256).
    Empty boxes after clipping are discarded. Malformed/nonfinite coordinates,
    inverted input boxes, invalid scores, or invalid image dimensions raise.
    This helper does not add NMS, change scores, or repair ground truth.
    """
    result = []
    for original in records:
        row = copy.deepcopy(original)
        width, height = float(row.get("width", 256)), float(row.get("height", 256))
        if not all(math.isfinite(value) and value > 0 for value in (width, height)):
            raise ValueError("Native tile dimensions must be finite and positive")
        retained = []
        for prediction in row["predictions"]:
            _probability(prediction["score"])
            box = np.asarray(prediction["box"], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("Boxes require four finite native xyxy coordinates")
            if box[2] < box[0] or box[3] < box[1]:
                raise ValueError("Cannot clip an inverted input prediction box")
            clipped = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                prediction["box"] = clipped.tolist()
                retained.append(prediction)
        row["predictions"] = retained
        result.append(row)
    return result


def box_iou(first, second):
    """Pairwise IoU for valid (N,4) and (M,4) xyxy arrays; empty arrays allowed."""
    first = np.asarray(first, dtype=float).reshape(-1, 4)
    second = np.asarray(second, dtype=float).reshape(-1, 4)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=2)
    first_area = (first[:, 2:] - first[:, :2]).prod(axis=1)
    second_area = (second[:, 2:] - second[:, :2]).prod(axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _prepare_detection(records, max_detections):
    prepared = []
    identities = set()
    for row in records:
        if row.get("exhaustive") is not True:
            continue
        identity = (row["sample_id"], str(row["query"]))
        if identity in identities:
            raise ValueError("Duplicate exhaustive sample/query detection record")
        identities.add(identity)
        width, height = float(row.get("width", 256)), float(row.get("height", 256))
        boxes = [_box(value, width, height) for value in row["boxes"]]
        predictions = [
            (_box(item["box"], width, height), _probability(item["score"]))
            for item in row["predictions"]
        ]
        predictions.sort(key=lambda item: -item[1])
        prepared.append(
            {"identity": identity, "boxes": boxes, "predictions": predictions[:max_detections]}
        )
    return prepared


def _match_detections(records, iou_threshold, score_threshold=0.0):
    predictions = []
    ground_truth = {}
    for row in records:
        identity = row["identity"]
        ground_truth[identity] = row["boxes"]
        for box, score in row["predictions"]:
            if score >= score_threshold:
                predictions.append((score, identity, box))
    predictions.sort(key=lambda value: -value[0])
    matched = {identity: set() for identity in ground_truth}
    true_positives, false_positives = [], []
    small_matched = 0
    for _, identity, box in predictions:
        boxes = ground_truth[identity]
        overlaps = box_iou([box], boxes)[0]
        candidates = [
            index
            for index, overlap in enumerate(overlaps)
            if overlap >= iou_threshold and index not in matched[identity]
        ]
        if candidates:
            best = max(candidates, key=lambda index: overlaps[index])
            matched[identity].add(best)
            area = np.prod(boxes[best][2:] - boxes[best][:2])
            small_matched += int(area < 32**2)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)
    total = sum(len(boxes) for boxes in ground_truth.values())
    small_total = sum(
        int(np.prod(box[2:] - box[:2]) < 32**2) for boxes in ground_truth.values() for box in boxes
    )
    return np.array(true_positives), np.array(false_positives), total, small_matched, small_total


def _interpolated_ap(records, iou_threshold):
    tp, fp, total, _, _ = _match_detections(records, iou_threshold)
    if not total:
        return None
    if not len(tp):
        return 0.0
    recall = np.cumsum(tp) / total
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1)
    return float(
        np.mean([np.max(precision[recall >= level], initial=0) for level in np.linspace(0, 1, 101)])
    )


def evaluate_detections(records, score_threshold=0.5, max_detections=100):
    """Return custom AP and object recall only on explicitly exhaustive annotations."""
    records = list(records)
    score_threshold = _probability(score_threshold)
    if not isinstance(max_detections, int) or max_detections < 1:
        raise ValueError("max_detections must be a positive integer")
    prepared = _prepare_detection(records, max_detections)
    by_query = defaultdict(list)
    for row in prepared:
        by_query[row["identity"][1]].append(row)
    query_results = {}
    for query, rows in sorted(by_query.items()):
        aps = [_interpolated_ap(rows, threshold) for threshold in np.linspace(0.5, 0.95, 10)]
        tp, fp, total, small_matched, small_total = _match_detections(rows, 0.5, score_threshold)
        query_results[query] = {
            "tiles": len(rows),
            "objects": total,
            "ap50": aps[0],
            "ap50_95": float(np.mean(aps)) if total else None,
            "tp_at_score_threshold": int(tp.sum()),
            "fp_at_score_threshold": int(fp.sum()),
            "recall50_at_score_threshold": _ratio(int(tp.sum()), total),
            "small_objects": small_total,
            "small_matched50": small_matched,
            "small_recall50_at_score_threshold": _ratio(small_matched, small_total),
        }
    scored_queries = [result for result in query_results.values() if result["objects"]]
    total = sum(result["objects"] for result in query_results.values())
    tp = sum(result["tp_at_score_threshold"] for result in query_results.values())
    fp = sum(result["fp_at_score_threshold"] for result in query_results.values())
    small_total = sum(result["small_objects"] for result in query_results.values())
    small_matched = sum(result["small_matched50"] for result in query_results.values())
    return {
        "method": "Custom COCO-style 101-point AP; no crowd/ignore/area AP support",
        "exhaustive_records": len(prepared),
        "excluded_nonexhaustive_records": len(records) - len(prepared),
        "max_detections_per_tile_query": max_detections,
        "score_threshold_for_recall": score_threshold,
        "ap50": float(np.mean([result["ap50"] for result in scored_queries]))
        if scored_queries
        else None,
        "ap50_95": float(np.mean([result["ap50_95"] for result in scored_queries]))
        if scored_queries
        else None,
        "objects": total,
        "tp_at_score_threshold": tp,
        "fp_at_score_threshold": fp,
        "precision50_at_score_threshold": _ratio(tp, tp + fp),
        "recall50_at_score_threshold": _ratio(tp, total),
        "small_objects": small_total,
        "small_recall50_at_score_threshold": _ratio(small_matched, small_total),
        "by_query": query_results,
    }
