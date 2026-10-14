"""Read-only environment observations; no model loading, downloads, or repairs."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_JOB = REPOSITORY / "data/nys-naip-2022-rgb-060m"
PACKAGES = (
    "numpy",
    "rasterio",
    "pyarrow",
    "pyproj",
    "shapely",
    "Pillow",
    "torch",
    "open-clip-torch",
    "huggingface-hub",
    "faiss-cpu",
    "threadpoolctl",
)


def read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def lock_state(path):
    """Probe existing lock bytes without creating files or trusting a saved PID."""
    try:
        handle = Path(path).open("rb")
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unknown"
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                return (
                    "held"
                    if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK)
                    else "unknown"
                )
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                return "held" if exc.errno in (errno.EACCES, errno.EAGAIN) else "unknown"
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return "idle"
    except OSError:
        return "unknown"
    finally:
        handle.close()


def index_metadata(collection, manifest):
    """Mirror the cheap manifest contract; tests compare the backend signature.

    This stdlib-only observation never authorizes loading an index. The search
    reader additionally verifies original bundles and the native index layout.
    """
    payload = {
        key: collection.get(key)
        for key in (
            "run_fingerprint",
            "embedding_dim",
            "chip_count",
            "shards",
        )
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("format_version") != 1 or manifest.get("algorithm") != "ivf-pq":
        raise ValueError("Unsupported index manifest")
    if manifest.get("snapshot_signature") != signature:
        raise ValueError("Index snapshot differs from collection")
    if not re.fullmatch(r"index-[a-f0-9]{20}\.faiss", manifest.get("index_file", "")):
        raise ValueError("Invalid index filename")
    if (
        manifest.get("ntotal") != collection["chip_count"]
        or manifest.get("dimension") != collection["embedding_dim"]
    ):
        raise ValueError("Index count/dimension differs from collection")
    sources, shards = manifest.get("sources", []), collection["shards"]
    if len(sources) != len(shards):
        raise ValueError("Index source mapping differs")
    offset = 0
    for source, shard in zip(sources, shards):
        if (
            source["start"] != offset
            or source["shard_id"] != shard["shard_id"]
            or source["source_uri"] != shard["source_uri"]
            or source["count"] != shard["chip_count"]
        ):
            raise ValueError("Index source mapping differs")
        offset += source["count"]
    return signature


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observe(
    job, collection, config, model_cache=None, *, gpu_probe=False, verify_index_checksum=False
):
    job, collection, config = (Path(path).resolve() for path in (job, collection, config))
    checks = []

    def add(name, status, detail, remedy=None):
        checks.append(
            {
                "name": name,
                "status": status,
                "detail": detail,
                **({"remedy": remedy} if remedy else {}),
            }
        )

    add("python", "ok", {"executable": sys.executable, "version": sys.version.split()[0]})
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    add(
        "dependencies",
        "warning" if any(value is None for value in versions.values()) else "ok",
        versions,
        "Use scripts/setup-dev.ps1 for isolated tests; see docs/rx-9070xt-embedding-setup.md for production GPU setup.",
    )
    add("paths", "ok", {"job": str(job), "collection": str(collection), "config": str(config)})
    selected_config = None
    try:
        selected_config = read_object(config)
        if not re.fullmatch(r"[a-f0-9]{40}", selected_config.get("model_revision", "")):
            raise ValueError("Model revision is not a full commit SHA")
        add(
            "config",
            "ok",
            {
                key: selected_config[key]
                for key in ("dataset_id", "chip_size", "stride", "model_id", "model_revision")
            },
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        selected_config = None
        add(
            "config",
            "error",
            str(exc),
            "Check configs and docs/nys-pipeline.md; do not replace the pinned model.",
        )

    manifest = None
    try:
        manifest = read_object(job / "manifest.json")
        add(
            "imagery_manifest",
            "ok",
            {
                "source_count": len(manifest["tiles"]),
                "complete": (job / "completion.json").is_file(),
            },
        )
    except FileNotFoundError:
        add(
            "imagery_manifest",
            "warning",
            "Local imagery manifest is absent",
            "Fresh clones use docs/DEVELOPMENT.md synthetic fixtures; existing data must not be moved.",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        add("imagery_manifest", "error", str(exc))

    catalog = None
    try:
        root = read_object(collection / "run.json")
        catalog = read_object(collection / "collection.json")
        shards = catalog["shards"]
        if (
            root["run_fingerprint"] != catalog["run_fingerprint"]
            or catalog["chip_count"] != sum(shard["chip_count"] for shard in shards)
            or len({shard["shard_id"] for shard in shards}) != len(shards)
        ):
            raise ValueError("Collection catalog identity/count differs")
        add(
            "collection",
            "ok",
            {
                "published_sources": len(shards),
                "published_chips": catalog["chip_count"],
                "state": catalog.get("state", "unknown"),
            },
        )
    except FileNotFoundError:
        add(
            "collection",
            "warning",
            "Published collection metadata is absent",
            "See docs/DEVELOPMENT.md; no embedding job is started by diagnostics.",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        catalog = None
        add(
            "collection",
            "error",
            str(exc),
            "Investigate metadata before search; preserve published bundles.",
        )

    locks = {
        "download": job / "worker.lock",
        "collection": collection / "worker.lock",
        "gpu": job / "embeddings/gpu-runtime/worker.lock",
        "index_builder": collection / "search-index/worker.lock",
        "supplement": job
        / "supplements/naip-2021-release-gapfill/embeddings/georsclip-vith14-s256-st192/worker.lock",
        "frontend": REPOSITORY / "frontend/logs/frontend-service.lock",
    }
    observations = {name: lock_state(path) for name, path in locks.items()}
    add(
        "ownership",
        "warning" if any(value in ("held", "unknown") for value in observations.values()) else "ok",
        observations,
        "Observations can race; actual operation locks remain authoritative. Do not pause or terminate another owner automatically.",
    )

    supplement = (
        job / "supplements/naip-2021-release-gapfill/embeddings/georsclip-vith14-s256-st192"
    )
    try:
        supplement_root = read_object(supplement / "run.json")
        supplement_catalog = read_object(supplement / "collection.json")
        if supplement_root["run_fingerprint"] != supplement_catalog[
            "run_fingerprint"
        ] or supplement_catalog["chip_count"] != sum(
            shard["chip_count"] for shard in supplement_catalog["shards"]
        ):
            raise ValueError("Supplement catalog identity/count differs")
        add(
            "supplement",
            "ok",
            {
                "collection": str(supplement),
                "published_chips": supplement_catalog["chip_count"],
                "state": supplement_catalog.get("state", "unknown"),
            },
        )
    except FileNotFoundError:
        add("supplement", "skipped", "Optional supplement metadata is absent")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        add(
            "supplement",
            "warning",
            str(exc),
            "Inspect docs/nys-gapfill.md; preserve base-only availability.",
        )

    try:
        settings = read_object(REPOSITORY / "frontend/settings.json")
        add(
            "frontend_cache",
            "ok",
            {
                "cache_limit_bytes": settings["cache_limit_bytes"],
                "imagery_workers": settings["imagery_workers"],
                "usage": "not scanned",
            },
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        add(
            "frontend_cache",
            "warning",
            str(exc),
            "See frontend/README.md for bounded cache settings.",
        )

    if catalog is not None:
        try:
            index = read_object(collection / "search-index/manifest.json")
            signature = index_metadata(catalog, index)
            artifact = (collection / "search-index" / index["index_file"]).resolve()
            if not artifact.is_relative_to((collection / "search-index").resolve()):
                raise ValueError("Index artifact escapes its directory")
            if artifact.stat().st_size != index["index_bytes"]:
                raise ValueError("Index file size differs")
            if verify_index_checksum and file_sha256(artifact) != index["index_sha256"]:
                raise ValueError("Index checksum differs")
            add(
                "index",
                "ok",
                {
                    "snapshot_signature": signature,
                    "bytes": index["index_bytes"],
                    "checksum": "verified" if verify_index_checksum else "not checked",
                    "native_layout": "not loaded",
                },
            )
        except FileNotFoundError:
            add(
                "index",
                "warning",
                "Index manifest or artifact is absent; exact search remains available",
                "See docs/nys-search-index.md.",
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            add(
                "index",
                "error",
                str(exc),
                "Preserve vectors and investigate the derived index; see docs/nys-search-index.md.",
            )
    else:
        add("index", "skipped", "No valid catalog to compare")

    hub_cache = (
        Path(model_cache)
        if model_cache
        else Path(
            os.environ.get("HF_HUB_CACHE")
            or (
                Path(
                    os.environ.get("HF_HOME")
                    or (
                        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
                        / "huggingface"
                    )
                )
                / "hub"
            )
        )
    )
    if selected_config:
        checkpoint = (
            hub_cache
            / ("models--" + selected_config["model_id"].replace("/", "--"))
            / "snapshots"
            / selected_config["model_revision"]
            / selected_config["model_filename"]
        )
        add(
            "model_cache",
            "ok" if checkpoint.is_file() else "warning",
            {
                "hub_cache": str(hub_cache),
                "pinned_checkpoint_present": checkpoint.is_file(),
                "checksum": "not checked",
            },
            "Pass --model-cache for a custom hub cache; see GPU setup guide. No weights are downloaded.",
        )

    ancestor = job
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    try:
        usage = shutil.disk_usage(ancestor)
        add(
            "storage",
            "ok",
            {
                "free_bytes": usage.free,
                "job_cap_bytes": manifest.get("folder_cap_bytes", 300_000_000_000)
                if manifest
                else 300_000_000_000,
                "reserve_bytes": manifest.get("reserve_bytes", 5_000_000_000)
                if manifest
                else 5_000_000_000,
                "job_usage": "not scanned",
            },
        )
    except OSError as exc:
        add("storage", "error", str(exc))
    if gpu_probe:
        try:
            import torch

            available = torch.cuda.is_available()
            add(
                "gpu",
                "ok" if available else "error",
                {
                    "available": available,
                    "hip": torch.version.hip,
                    "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0) if available else None,
                },
                "See docs/rx-9070xt-embedding-setup.md; production inference never falls back to CPU.",
            )
        except (ImportError, OSError, RuntimeError) as exc:
            add(
                "gpu",
                "error",
                str(exc),
                "See frontend/SEARCH_TROUBLESHOOTING.md and the GPU setup guide.",
            )
    else:
        add(
            "gpu",
            "skipped",
            "Pass --gpu-probe to import PyTorch and query device availability; no model is loaded",
        )
    return {
        "schema_version": 1,
        "checks": checks,
        "notice": "Read-only observations, not full integrity verification or an ownership reservation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--collection-dir", type=Path)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/nys-georsclip.json")
    parser.add_argument("--model-cache", type=Path, help="Hugging Face hub cache directory")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--gpu-probe", action="store_true")
    parser.add_argument(
        "--verify-index-checksum",
        action="store_true",
        help="Read the entire selected index file, never statewide imagery",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return 1 for warnings or errors; missing production assets are expected in fresh clones",
    )
    args = parser.parse_args(argv)
    job = args.job_root if args.job_root.is_absolute() else REPOSITORY / args.job_root
    collection = args.collection_dir or job / "embeddings/georsclip-vith14-s256-st192"
    if not collection.is_absolute():
        collection = REPOSITORY / collection
    config = args.config if args.config.is_absolute() else REPOSITORY / args.config
    report = observe(
        job,
        collection,
        config,
        args.model_cache,
        gpu_probe=args.gpu_probe,
        verify_index_checksum=args.verify_index_checksum,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(report["notice"])
        for check in report["checks"]:
            print(f"[{check['status']}] {check['name']}: {json.dumps(check['detail'])}")
            if check.get("remedy") and check["status"] in ("warning", "error"):
                print(f"  {check['remedy']}")
    errors = any(check["status"] == "error" for check in report["checks"])
    warnings = any(check["status"] == "warning" for check in report["checks"])
    return int(errors or (args.strict and warnings))


if __name__ == "__main__":
    raise SystemExit(main())
