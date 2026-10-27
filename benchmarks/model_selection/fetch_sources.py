"""Download reviewable source/metadata, never execute remote code or install torch."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    "peekaboolean": "bykof/peekaboolean-450m",
    "intern": "internlm/Intern-Decision-0.8B",
    "d3": "vllm-sr/d3-lite",
    "imajev": "mohit67890/imajev-2b",
    "blip": "Salesforce/blip-itm-base-coco",
    "gdino": "IDEA-Research/grounding-dino-base",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "llmdet": "iSEE-Laboratory/llmdet_large",
    "omdet": "omlab/omdet-turbo-swin-tiny-hf",
    "sam3lite": "vil-uob/sam3-litetext-s0",
}


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "standalone-model-benchmark"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


PINS = {}


def metadata(item):
    key, repo = item
    previous = PINS.get(key, {}).get("revision")
    suffix = "/revision/" + previous if previous else ""
    info = json.loads(get("https://huggingface.co/api/models/" + repo + suffix))
    sha = info["sha"]
    save(ROOT / "local" / "metadata" / (key + ".json"), json.dumps(info, indent=2).encode())
    files = [s["rfilename"] for s in info["siblings"]]
    selected = [
        f
        for f in files
        if f.endswith(".py")
        or f
        in {
            "README.md",
            "config.json",
            "adapter_config.json",
            "scorer_config.json",
            "calibration-256.json",
            "preprocessor_config.json",
            "processor_config.json",
            "decision_config.json",
            "calibration.json",
            "chat_template.jinja",
        }
    ]
    hashes = {}
    for name in selected:
        content = get(f"https://huggingface.co/{repo}/resolve/{sha}/{name}")
        save(ROOT / "vendor" / key / name, content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    print(key, repo, sha, selected, flush=True)
    return key, {"repo_id": repo, "revision": sha, "files": files, "source_sha256": hashes}


def main():
    global PINS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Intentionally replace model/source revisions with latest",
    )
    args = parser.parse_args()
    pin_path = ROOT / "model_pins.json"
    PINS = json.loads(pin_path.read_text()) if pin_path.exists() and not args.refresh else {}
    pins = dict(ThreadPoolExecutor(5).map(metadata, MODELS.items()))
    repo = "bykof/peekaboolean"
    commit = (
        PINS.get("peekaboolean_source", {}).get("revision")
        or json.loads(get(f"https://api.github.com/repos/{repo}/commits/main"))["sha"]
    )
    tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
    hashes = {}
    for entry in tree["tree"]:
        name = entry["path"]
        if entry["type"] == "blob" and (
            name.endswith(".py") or name in {"pyproject.toml", "README.md", "LICENSE"}
        ):
            content = get(f"https://raw.githubusercontent.com/{repo}/{commit}/{name}")
            save(ROOT / "vendor" / "peekaboolean_source" / name, content)
            hashes[name] = hashlib.sha256(content).hexdigest()
    pins["peekaboolean_source"] = {"repo": repo, "revision": commit, "sha256": hashes}
    if "lfm_base" in PINS:
        pins["lfm_base"] = PINS["lfm_base"]
    save(ROOT / "model_pins.json", (json.dumps(pins, indent=2) + "\n").encode())
    print("Sources saved for review; nothing executed.", flush=True)


if __name__ == "__main__":
    main()
