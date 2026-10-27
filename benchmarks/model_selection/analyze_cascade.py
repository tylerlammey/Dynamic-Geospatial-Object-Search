"""Offline paired-output economics; never a live retrieval/decision/detection loop."""

import argparse
import json
from pathlib import Path

from metrics import clip_prediction_boxes, evaluate_detections

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-gates",
        action="store_true",
        help="Also evaluate slower decision baselines in paired scenarios",
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "local/tiles.json").read_text())
    labels = {r["sample_id"]: r for r in manifest["rows"]}
    uncertain_box_id = manifest["rows"][66]["sample_id"]
    timings = {}
    predictions = {}
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
        path = ROOT / "local" / (key + "-predictions.json")
        result_path = ROOT / "results" / (key + ".json")
        if not path.exists() or not result_path.exists():
            continue
        rows = json.loads(path.read_text())
        predictions[key] = {
            r["sample_id"]: {
                **r,
                **{
                    k: v
                    for k, v in labels[r["sample_id"]].items()
                    if k in ["label", "boxes", "exhaustive"]
                },
            }
            for r in rows
        }
        result = json.loads(result_path.read_text())
        timings[key] = {
            b["batch_size"]: b["summary"]["total_ms"]["median"] / b["batch_size"]
            for b in result["batches"]
            if b["status"] == "ok"
        }
        for suffix in ["batch8", "batch-grid"]:
            extra_path = ROOT / "results" / (key + "-" + suffix + ".json")
            if extra_path.exists():
                extra = json.loads(extra_path.read_text())
                timings[key].update(
                    {
                        b["batch_size"]: b["summary"]["total_ms"]["median"] / b["batch_size"]
                        for b in extra["batches"]
                        if b["status"] == "ok"
                    }
                )
    output = {
        "warning": "Convenience sample and assistant annotations; estimates exclude cold loads, extra ANN calls, scheduling and selection-dependent batch fill.",
        "paired_recall": "Actual detector predictions removed on rejected tiles; no independence assumption.",
        "pairs": [],
    }
    gates = ["peekaboolean", "intern", "d3", "blip"] if args.all_gates else ["peekaboolean"]
    for gate in gates:
        if gate not in predictions:
            continue
        for detector in ["gdino", "owlv2", "llmdet", "omdet", "sam3lite"]:
            if detector not in predictions:
                continue
            ids = sorted(predictions[gate].keys() & predictions[detector].keys())
            positive = [i for i in ids if labels[i]["label"] == 1]
            rows = [
                dict(predictions[detector][i])
                for i in ids
                if predictions[detector][i]["exhaustive"] and i != uncertain_box_id
            ]
            rows = clip_prediction_boxes(rows)
            direct = evaluate_detections(rows, score_threshold=0.1)
            for threshold in [0.05, 0.1, 0.5]:
                accepted = {i: predictions[gate][i]["probability"] >= threshold for i in ids}
                a = sum(accepted.values()) / len(ids)
                cascade = evaluate_detections(
                    [
                        {**r, "predictions": r["predictions"] if accepted[r["sample_id"]] else []}
                        for r in rows
                    ],
                    score_threshold=0.1,
                )
                costs = {}
                for batch in sorted(timings[gate].keys() & timings[detector].keys()):
                    d, t = timings[gate][batch], timings[detector][batch]
                    costs[batch] = {
                        "decision_ms_per_tile": d,
                        "direct_detector_ms_per_tile": t,
                        "maximum_accepted_fraction_to_save_compute": 1 - d / t,
                        "estimated_cascade_ms_per_candidate": d + a * t,
                        "estimated_speedup": t / (d + a * t),
                    }
                output["pairs"].append(
                    {
                        "gate": gate,
                        "detector": detector,
                        "threshold": threshold,
                        "accepted_fraction_all_audit_pairs": a,
                        "tile_false_negatives": sum(not accepted[i] for i in positive),
                        "tile_positive_count": len(positive),
                        "direct_boxes": direct,
                        "cascade_boxes": cascade,
                        "costs": costs,
                    }
                )
    (ROOT / "results/cascade.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Wrote", len(output["pairs"]), "paired offline cascade scenarios")


if __name__ == "__main__":
    main()
