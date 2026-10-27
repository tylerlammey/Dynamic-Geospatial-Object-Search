"""Require full tile/query probability agreement for experimental fast solvers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    output = {}
    for base in ["intern", "d3", "peekaboolean"]:
        variant = base + ("-naive" if base == "peekaboolean" else "-neumann")
        paths = [ROOT / "local" / (k + "-predictions.json") for k in [base, variant]]
        if not all(p.exists() for p in paths):
            continue
        a, b = [
            {r["sample_id"]: r["probability"] for r in json.loads(p.read_text())} for p in paths
        ]
        if a.keys() != b.keys():
            raise ValueError("Variant comparisons need exactly the same tile/query IDs")
        differences = {i: abs(a[i] - b[i]) for i in a}
        worst = max(differences, key=differences.get)
        output[variant] = {
            "reference": base,
            "pairs": len(a),
            "max_abs_probability_difference": differences[worst],
            "mean_abs_probability_difference": sum(differences.values()) / len(a),
            "worst_sample_id": worst,
            "threshold_disagreements": {
                str(t): sum((a[i] >= t) != (b[i] >= t) for i in a)
                for t in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
            },
            "within_0_002": differences[worst] <= 0.002,
            "scope": "This audit only; numerical equivalence is not aerial accuracy or universal error guarantee.",
        }
    (ROOT / "results/variant-agreement.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
