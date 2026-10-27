"""Render measured tables without conflating warm batches or unverified labels."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYS = [
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
]


def main():
    lines = [
        "# Measured results",
        "",
        "All latency numbers below are milliseconds; batching quantiles are per batch.",
        "",
        "| Model | Status | B1 median/p95 | B4 median / tiles/s | B8 median / tiles/s | Load s | Peak allocated GiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    summaries = {}
    for key in KEYS:
        path = ROOT / "results" / (key + ".json")
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        bs = {b["batch_size"]: b for b in r["batches"] if b["status"] == "ok"}
        for suffix in ["batch8", "batch-grid"]:
            extra_path = ROOT / "results" / f"{key}-{suffix}.json"
            if extra_path.exists():
                extra = json.loads(extra_path.read_text())
                bs.update({b["batch_size"]: b for b in extra["batches"] if b["status"] == "ok"})

        def cell(n):
            if n not in bs:
                return "unavailable"
            s = bs[n]["summary"]
            tail = s["total_ms"]["p95"] if n == 1 else s["throughput_tiles_second"]
            return f"{s['total_ms']['median']:.1f} / {tail:.1f}"

        peak = (
            max(
                [b.get("peak_allocated_bytes", 0) for b in bs.values()]
                + [r.get("load_peak_allocated_bytes", 0)]
            )
            / 2**30
        )
        lines.append(
            f"| {key} | {r['status']} | {cell(1)} | {cell(4)} | {cell(8)} | {r.get('load_seconds', 0):.2f} | {peak:.2f} |"
        )
        summaries[key] = {
            "status": r["status"],
            "load_seconds": r.get("load_seconds"),
            "peak_allocated_gib": peak,
            "batches": {n: b["summary"] for n, b in bs.items()},
        }
    lines += [
        "",
        "| Model | B1 CPU preparation | H2D | Forward wall / device | Postprocess |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, r in summaries.items():
        if 1 not in r["batches"]:
            continue
        s = r["batches"][1]

        def ms(k):
            return s[k]["median"]

        lines.append(
            f"| {key} | {ms('preprocess_ms'):.2f} | {ms('transfer_ms'):.2f} | {ms('forward_wall_ms'):.2f} / {ms('forward_device_ms'):.2f} | {ms('postprocess_ms'):.2f} |"
        )
    evaluation = ROOT / "results/evaluation.json"
    if evaluation.exists():
        ev = json.loads(evaluation.read_text())
        lines += [
            "",
            "## Exploratory assistant-label audit",
            "",
            "These are NOT human-verified accuracy or calibration results.",
            "",
            "| Model | Test n | Precision | Recall | FNR | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key, r in ev["models"].items():
            if "raw_test" in r:
                s = r["raw_test"]["overall"]

                def f(k):
                    return f"{s[k]:.3f}" if s.get(k) is not None else "n/a"

                lines.append(
                    f"| {key} | {s['labeled']} | {f('precision')} | {f('recall')} | {f('false_negative_rate')} | {f('brier')} | {f('ece')} |"
                )
        lines += [
            "",
            "| Detector | Custom AP50 | Custom AP50:95 | Small recall50 @ .1 |",
            "|---|---:|---:|---:|",
        ]
        for key, r in ev["models"].items():
            if "boxes_all_exploratory" in r:
                s = r["boxes_all_exploratory"]

                def f(k):
                    return f"{s[k]:.3f}" if s.get(k) is not None else "n/a"

                lines.append(
                    f"| {key} | {f('ap50')} | {f('ap50_95')} | {f('small_recall50_at_score_threshold')} |"
                )
    (ROOT / "results/TABLES.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results/summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
