"""Record local weight hashes and borrowed runtime versions; never change an env."""

import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dependencies"))
from dataset import digest  # noqa: E402


def main():
    paths = json.loads((ROOT / "local/model_paths.json").read_text())
    pins = json.loads((ROOT / "model_pins.json").read_text())
    weights = {}
    for key, directory in paths.items():
        path = Path(directory)
        files = [p for p in path.iterdir() if p.suffix in {".safetensors", ".pt", ".bin"}]
        weights[key] = {
            "repo_id": pins[key]["repo_id"],
            "revision": pins[key]["revision"],
            "files": {
                p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(files)
            },
        }
    (ROOT / "results/checkpoint-inventory.json").write_text(json.dumps(weights, indent=2) + "\n")
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "safetensors",
        "tokenizers",
        "timm",
        "numpy",
        "pillow",
        "rasterio",
        "pyarrow",
        "pyproj",
        "shapely",
        "faiss-cpu",
        "threadpoolctl",
        "huggingface-hub",
        "open-clip-torch",
        "peft",
        "accelerate",
    ]
    versions = {p: importlib.metadata.version(p) for p in packages}
    (ROOT / "results/runtime-observed.json").write_text(json.dumps(versions, indent=2) + "\n")
    manifest = json.loads((ROOT / "local/tiles.json").read_text())
    expected_ids = {r["sample_id"] for r in manifest["rows"]}
    final = {}
    for key in [
        "peekaboolean",
        "intern",
        "d3",
        "blip",
        "gdino",
        "owlv2",
        "llmdet",
        "omdet",
        "sam3lite",
    ]:
        result = json.loads((ROOT / "results" / (key + ".json")).read_text())
        predictions = json.loads((ROOT / "local" / (key + "-predictions.json")).read_text())
        if (
            result["status"] != "ok"
            or len(predictions) != len(expected_ids)
            or {r["sample_id"] for r in predictions} != expected_ids
        ):
            raise ValueError(f"Incomplete final candidate audit: {key}")
        final[key] = {
            "status": result["status"],
            "matched_image_query_pairs": len(predictions),
            "artifact": f"results/{key}.json",
            "revision": pins[key]["revision"],
        }
    (ROOT / "results/final-run-status.json").write_text(json.dumps(final, indent=2) + "\n")
    for path in (ROOT / "results").glob("peekaboolean-smoke*.json"):
        result = json.loads(path.read_text())
        if result.get("status") == "started":
            result.update(
                status="native_crash_in_restricted_sandbox",
                diagnosis="Minimal ordinary allocation also failed in sandbox and succeeded on host; no inference measurements from this attempt.",
            )
            path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Recorded",
        sum(len(x["files"]) for x in weights.values()),
        "weight files and runtime versions",
    )


if __name__ == "__main__":
    main()
