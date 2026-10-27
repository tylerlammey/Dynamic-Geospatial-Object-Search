"""Standalone serialized GPU benchmark. No production stage integration."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

from dataset import COLLECTION, JOB, digest, load_images

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dependencies"))
sys.path.insert(0, str(ROOT.parents[1] / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def environment():
    import torch

    props = torch.cuda.get_device_properties(0)
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "pillow",
        "peft",
        "accelerate",
        "huggingface-hub",
    ]
    return {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": props.name,
        "gpu_total_bytes": props.total_memory,
        "hip": torch.version.hip,
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "cpu_threads": torch.get_num_threads(),
        "pinned_memory": False,
        "core_sha256": {
            p: digest(ROOT.parents[1] / "src/geospatial_search" / p)
            for p in ["embed_georsclip.py", "tiling.py", "model.py", "bundles.py"]
        },
        "benchmark_code_sha256": {p.name: digest(p) for p in ROOT.glob("*.py")},
    }


def adapter(key, paths, dtype=None):
    if key == "sam3lite":
        from adapter_sam3lite import Adapter

        result = Adapter(paths[key], dtype or "float16")
    elif key.startswith("d3"):
        from adapter_d3 import Adapter

        result = Adapter(paths["d3"], dtype or "bfloat16")
    elif key.startswith("peekaboolean"):
        from adapters_decision import PeekAdapter

        result = PeekAdapter(
            paths["peekaboolean"],
            paths["lfm_base"],
            dtype or "bfloat16",
            deduplicate=not key.endswith("naive"),
        )
    elif key.startswith("intern"):
        from adapters_decision import InternAdapter

        result = InternAdapter(
            paths["intern"], dtype or "bfloat16", linear_patches=not key.endswith("conv3d")
        )
    elif key == "blip":
        from adapters_decision import BlipAdapter

        result = BlipAdapter(paths[key], dtype or "float16")
    else:
        from adapters_detection import Adapter

        native_key = "llmdet" if key == "llmdet-stock" else key
        result = Adapter(
            native_key,
            paths[native_key],
            dtype or "float16",
            restore_decoder_heads=key != "llmdet-stock",
        )
    if key.endswith("neumann"):
        from solvers import neumann_triangular_solve

        original_forward = result.forward

        def forward_with_experimental_solver(batch):
            with neumann_triangular_solve() as solver:
                output = original_forward(batch)
            result.info["experimental_solver_last_forward"] = dict(solver.stats)
            return output

        result.forward = forward_with_experimental_solver
        result.info["experimental_solver"] = (
            "FP32 Neumann unit-triangular inverse with residual guard and native fallback"
        )
    result.model.to("cuda").eval()
    return result


def transfer(batch, dtype):
    import torch

    dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    return {
        k: v.to("cuda", dtype=dt if v.is_floating_point() else v.dtype, non_blocking=False)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in batch.items()
    }


def timed(a, images, queries):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    cpu = a.prepare(images, queries)
    pre = time.perf_counter()
    batch = transfer(cpu, a.dtype)
    torch.cuda.synchronize()
    sent = time.perf_counter()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    with torch.inference_mode():
        raw = a.forward(batch)
    end.record()
    torch.cuda.synchronize()
    inferred = time.perf_counter()
    values = a.postprocess(raw, batch)
    if len(values) != len(images):
        raise ValueError("Adapter output count does not match the image/query batch")
    torch.cuda.synchronize()
    finish = time.perf_counter()
    # Finite outputs are a compatibility requirement, not an accuracy result.
    for value in values:
        if "probability" in value and not 0 <= value["probability"] <= 1:
            raise ArithmeticError("Nonfinite/out-of-range decision output")
        for pred in value.get("predictions", []):
            if not all(
                __import__("math").isfinite(float(x)) for x in [pred["score"], *pred["box"]]
            ):
                raise ArithmeticError("Nonfinite detector output")
    return (
        values,
        {
            "preprocess_ms": (pre - start) * 1000,
            "transfer_ms": (sent - pre) * 1000,
            "forward_wall_ms": (inferred - sent) * 1000,
            "forward_device_ms": begin.elapsed_time(end),
            "postprocess_ms": (finish - inferred) * 1000,
            "total_ms": (finish - start) * 1000,
        },
    )


def summarize(trials, batch_size):
    import numpy as np

    stats = {}
    for key in trials[0]:
        values = [v[key] for v in trials]
        stats[key] = {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "mean": float(np.mean(values)),
            "min": min(values),
            "max": max(values),
        }
    stats["throughput_tiles_second"] = (
        batch_size * len(trials) * 1000 / sum(t["total_ms"] for t in trials)
    )
    stats["median_amortized_ms_per_tile"] = stats["total_ms"]["median"] / batch_size
    return stats


def main():
    import torch

    from geospatial_search.embedding_control import gpu_access, idle_collection

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Timing-only follow-up; keep the independent prediction audit",
    )
    parser.add_argument(
        "--co-resident", help="Load a second adapter and measure sequential co-resident inference"
    )
    parser.add_argument(
        "--with-retriever",
        action="store_true",
        help="Also retain the existing pinned GeoRSCLIP query model for a VRAM experiment",
    )
    args = parser.parse_args()
    if (
        not torch.cuda.is_available()
        or not torch.version.hip
        or "9070 XT" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("Expected the requested RX 9070 XT ROCm GPU; no CPU fallback")
    if args.repeats < 1 or args.warmup < 1 or any(b < 1 for b in args.batches):
        raise ValueError("Positive batch sizes, warmup, and repetitions required")
    torch.set_num_threads(8)
    torch.manual_seed(20261026)
    paths = json.loads((ROOT / "local/model_paths.json").read_text())
    manifest_path = ROOT / "local/tiles.json"
    manifest = json.loads(manifest_path.read_text())
    if args.limit:
        manifest["rows"] = manifest["rows"][: args.limit]
    target = (
        ROOT
        / "results"
        / f"{args.model}{'-' + args.dtype if args.dtype else ''}{'-' + args.tag if args.tag else ''}.json"
    )
    target.parent.mkdir(exist_ok=True)
    result = {
        "model": args.model,
        "status": "started",
        "manifest_sha256": digest(manifest_path),
        "environment": environment(),
        "protocol": vars(args),
        "timing_scope": "synchronized serial CPU preparation + H2D + forward + postprocess; source TIFF decoding separate",
        "batches": [],
    }
    save(target, result)
    # An OS-backed ownership reservation is held for the entire benchmark, not
    # just observed before model loading. No PAUSE/STOP/control signals changed.
    with ExitStack() as locks:
        locks.enter_context(idle_collection(COLLECTION))
        locks.enter_context(gpu_access(JOB))
        try:
            images, read_ms = load_images(manifest)
            result["source_read_ms"] = summarize(
                [{"read_ms": x, "total_ms": x} for x in read_ms], 1
            )
            queries = [r["query"] for r in manifest["rows"]]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            a = adapter(args.model, paths, args.dtype)
            torch.cuda.synchronize()
            result["load_seconds"] = time.perf_counter() - start
            result["model_info"] = a.info
            result["resident_allocated_bytes"] = torch.cuda.memory_allocated()
            result["load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["parameter_count"] = sum(p.numel() for p in a.model.parameters())
            if args.co_resident:
                second = adapter(args.co_resident, paths)
                torch.cuda.synchronize()
                result["co_resident"] = {
                    "model": args.co_resident,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                }
            if args.with_retriever:
                from geospatial_search.model import load_config, load_model

                retriever, _, _, contract = load_model(load_config(), offline=True)
                torch.cuda.synchronize()
                result["resident_retriever"] = {
                    "model_id": contract["model_id"],
                    "revision": contract["model_revision"],
                    "parameter_dtype": str(next(retriever.parameters()).dtype),
                    "allocated_bytes_after_loading": torch.cuda.memory_allocated(),
                }
            save(target, result)
            for batch_size in args.batches:
                record = {"batch_size": batch_size, "status": "started", "warmup": [], "trials": []}
                result["batches"].append(record)
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    for i in range(args.warmup + args.repeats):
                        # One query repeated over independent tiles reflects an ANN
                        # request; rotate queries between trials. No cross-tile cache.
                        offset = (i % 12) * 8
                        indices = [(offset + j) % len(images) for j in range(batch_size)]
                        ims = [images[j] for j in indices]
                        qs = [queries[indices[0]]] * batch_size
                        values, timing = timed(a, ims, qs)
                        if args.co_resident:
                            _, other = timed(second, ims, qs)
                            timing = {k: timing[k] + other[k] for k in timing}
                        record["warmup" if i < args.warmup else "trials"].append(timing)
                        if i == 0:
                            record["first_batch_ms"] = timing["total_ms"]
                        if (i + 1) % 10 == 0:
                            print(args.model, "batch", batch_size, "iteration", i + 1, flush=True)
                    record["summary"] = summarize(record["trials"], batch_size)
                    record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    record["status"] = "ok"
                    print(args.model, batch_size, record["summary"]["total_ms"], flush=True)
                except Exception as exc:
                    record.update(
                        status="failed", error=repr(exc), traceback=traceback.format_exc()
                    )
                    print(args.model, batch_size, "FAILED", repr(exc), flush=True)
                    torch.cuda.empty_cache()
                save(target, result)
            if not args.co_resident and not args.skip_predictions:
                predictions = []
                batch_size = min(4, len(images))
                for start in range(0, len(images), batch_size):
                    values, timing = timed(
                        a, images[start : start + batch_size], queries[start : start + batch_size]
                    )
                    for row, value in zip(manifest["rows"][start : start + batch_size], values):
                        predictions.append({**row, **value})
                out = ROOT / "local" / (target.stem + "-predictions.json")
                save(out, predictions)
                result["prediction_file"] = out.relative_to(ROOT).as_posix()
                result["prediction_count"] = len(predictions)
            result["status"] = (
                "ok" if all(b["status"] == "ok" for b in result["batches"]) else "partial"
            )
        except Exception as exc:
            result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        save(target, result)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
