"""Exploratory audit metrics, explicitly separate from verified ground truth."""

import json
from pathlib import Path

from metrics import (
    apply_platt,
    clip_prediction_boxes,
    evaluate_decisions,
    evaluate_detections,
    fit_platt,
    threshold_sweep,
)

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "local/tiles.json").read_text())
    labels = {r["sample_id"]: r for r in manifest["rows"]}
    output = {
        "label_provenance": manifest["annotation"],
        "warning": "Assistant-created convenience annotations; NOT human-verified accuracy, aerial calibration or high-recall evidence.",
        "spatial_split": "entire source TIFF per split; provider buffers may cross sources; not proven spatial independence",
        "models": {},
    }
    for key in [
        "peekaboolean",
        "intern",
        "d3",
        "blip",
        "intern-neumann",
        "d3-neumann",
        "gdino",
        "owlv2",
        "llmdet",
        "omdet",
        "sam3lite",
    ]:
        prediction_path = ROOT / "local" / (key + "-predictions.json")
        if not prediction_path.exists():
            continue
        predictions = json.loads(prediction_path.read_text())
        rows = [
            {
                **r,
                **{
                    k: v
                    for k, v in labels[r["sample_id"]].items()
                    if k
                    in ["label", "boxes", "exhaustive", "split", "group_id", "annotation_source"]
                },
            }
            for r in predictions
        ]
        result = {"pairs": len(rows), "labeled_pairs": sum(r["label"] is not None for r in rows)}
        if "presence_probability" in rows[0]:
            result["direct_presence_raw_test"] = evaluate_decisions(
                [
                    {**r, "probability": r["presence_probability"]}
                    for r in rows
                    if r["split"] == "test"
                ]
            )
        if "probability" in rows[0]:
            test = [r for r in rows if r["split"] == "test"]
            calibration_rows = [r for r in rows if r["split"] == "calibration"]
            result["raw_all_exploratory"] = evaluate_decisions(rows)
            result["raw_test"] = evaluate_decisions(test)
            result["threshold_test"] = threshold_sweep(
                test, [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
            )
            # This example estimates calibration only on its calibration split.
            # Scores cannot be declared reliable for different queries/rank priors.
            try:
                calibration = fit_platt(rows)
                calibrated_test = [
                    {**r, "probability": float(p)}
                    for r, p in zip(
                        test,
                        apply_platt([r["probability"] for r in test], calibration),
                        strict=True,
                    )
                ]
                result["platt"] = calibration
                result["platt_test"] = evaluate_decisions(calibrated_test)
            except ValueError as exc:
                result["platt_unavailable"] = str(exc)
            positives = [r["probability"] for r in calibration_rows if r["label"] == 1]
            threshold = min(positives) if positives else 0.0
            result["zero_calibration_miss_threshold"] = threshold
            result["test_at_calibration_threshold"] = evaluate_decisions(test, threshold=threshold)
            result["accepted_fraction_at_calibration_threshold"] = sum(
                r["probability"] >= threshold for r in rows
            ) / len(rows)
            result["accepted_fraction_at_half"] = sum(r["probability"] >= 0.5 for r in rows) / len(
                rows
            )
        else:
            # One small round feature on pool tile66 is ambiguous in the review
            # sheet; exclude that tile from exhaustive box scoring, consistently
            # for every detector. Keep its clear object-presence label.
            for r in rows:
                if r["sample_id"] == manifest["rows"][66]["sample_id"]:
                    r["exhaustive"] = False
            rows = clip_prediction_boxes(rows)
            result["box_boundary_policy"] = (
                "clip native predictions to original tile bounds, discard empty; all detectors"
            )
            result["boxes_all_exploratory"] = evaluate_detections(rows, score_threshold=0.1)
            result["box_threshold_sweep"] = {
                str(t): evaluate_detections(rows, score_threshold=t)
                for t in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
            }
        output["models"][key] = result
    (ROOT / "results/evaluation.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v.get("raw_test", v.get("boxes_all_exploratory"))
                for k, v in output["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
