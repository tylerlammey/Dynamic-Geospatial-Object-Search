"""Fetch pinned checkpoints only into this benchmark's ignored cache."""

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_DISABLE_XET"] = "1"


def main():
    from huggingface_hub import file_download, snapshot_download

    # Restricted Windows accounts may pass a capability check then fail symlink
    # creation on another worker. Explicitly use HF's normal copy fallback.
    file_download.are_symlinks_supported = lambda *args, **kwargs: False
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "keys",
        nargs="+",
        choices=[
            "peekaboolean",
            "intern",
            "d3",
            "blip",
            "gdino",
            "owlv2",
            "llmdet",
            "omdet",
            "sam3lite",
        ],
    )
    args = parser.parse_args()
    pins = json.loads((ROOT / "model_pins.json").read_text())
    if "peekaboolean" in args.keys and "lfm_base" not in pins:
        repo = "LiquidAI/LFM2.5-VL-450M"
        with urllib.request.urlopen(
            "https://huggingface.co/api/models/" + repo, timeout=60
        ) as response:
            info = json.load(response)
        pins["lfm_base"] = {
            "repo_id": repo,
            "revision": info["sha"],
            "files": [s["rfilename"] for s in info["siblings"]],
        }
        (ROOT / "model_pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    keys = args.keys + (["lfm_base"] if "peekaboolean" in args.keys else [])

    def fetch(key):
        pin = pins[key]
        files = [
            f
            for f in pin["files"]
            if f.endswith((".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken"))
            or f == "head.pt"
        ]
        # No alternative precision formats, MLX/GGUF copies, remote executable code,
        # or .bin pickle checkpoints. Native safetensors weights only.
        files = [f for f in files if "/" not in f]
        if key == "blip":
            files.append(
                "pytorch_model.bin"
            )  # Official model has no safetensors; HF weights_only=True.
        path = snapshot_download(
            pin["repo_id"],
            revision=pin["revision"],
            cache_dir=ROOT / "cache",
            allow_patterns=files,
            max_workers=3,
        )
        print(key, path, flush=True)
        return key, path

    paths = dict(ThreadPoolExecutor(2).map(fetch, keys))
    target = ROOT / "local" / "model_paths.json"
    old = json.loads(target.read_text()) if target.exists() else {}
    old.update(paths)
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(old, indent=2) + "\n")


if __name__ == "__main__":
    main()
