r"""Attribute Intern's warm latency without changing its numerical operations.

Run after the serialized suite releases its GPU lock::

    .\.venv\Scripts\python.exe -B benchmarks/model_selection/profile_intern.py

The parent run.py supplies the same adapter, dtype, transfer, CPU thread count,
native source window and GPU ownership locks as the benchmark. Instrumentation
synchronizes before and after every selected operator/module, so its wall times
are diagnostic and may exceed the separately measured uninstrumented baseline.
Nested timings are inclusive: do not add chunk, triangular-solve and LM times.
No kernel substitution, generated tokens, Kineto, or dataset copy is used.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import statistics
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

# run installs the isolated dependency path and offline settings before torch /
# Transformers are imported. Importing it does not reserve or run the GPU.
import run as benchmark


def _source_record(function):
    record = {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", repr(function)),
    }
    try:
        path = Path(inspect.getsourcefile(inspect.unwrap(function)))
        record.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except (TypeError, OSError):
        record["path"] = None
    return record


def _shape_records(values):
    """Shapes and dtypes only; no imagery, activation values or query text."""
    import torch

    return [
        {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        for value in values
        if isinstance(value, torch.Tensor)
    ]


class SynchronizedProfile:
    """Temporary wrappers preserve all arguments, results and original bindings."""

    def __init__(self, model, events=False):
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5

        self.torch = torch
        self.qwen = modeling_qwen3_5
        self.model = model
        self.events = events
        self.records = []
        self.sources = {}
        self.iteration = None
        self._restore = []
        self._handles = []
        self._starts = defaultdict(list)

    def _start(self):
        self.torch.cuda.synchronize()
        begin = None
        if self.events:
            begin = self.torch.cuda.Event(enable_timing=True)
            begin.record()
        return time.perf_counter(), begin

    def _finish(self, label, start, inputs=()):
        end = None
        if self.events:
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
        self.torch.cuda.synchronize()
        wall_start, begin = start
        record = {
            "iteration": self.iteration,
            "label": label,
            "wall_ms": (time.perf_counter() - wall_start) * 1000,
            "inputs": _shape_records(inputs),
        }
        if self.events:
            record["device_ms"] = begin.elapsed_time(end)
        self.records.append(record)

    def _wrap(self, owner, attribute, label):
        original = getattr(owner, attribute, None)
        if not callable(original):
            return
        self.sources[label] = _source_record(original)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            started = self._start()
            try:
                return original(*args, **kwargs)
            finally:
                # Preserve the operator's return object and mutation behavior;
                # diagnostics retain no tensor outputs or model references.
                self._finish(label, started, args[:5])

        setattr(owner, attribute, measured)
        self._restore.append((owner, attribute, original))

    def _hook(self, module, label):
        def before(_module, inputs):
            self._starts[label].append((self._start(), inputs[:1]))

        def after(_module, _inputs, _output):
            if self._starts[label]:
                started, inputs = self._starts[label].pop()
                self._finish(label, started, inputs)

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after, always_call=True))

    def __enter__(self):
        # Current native prefill binds these module-level functions. The update
        # and recurrent calls should be zero because this adapter disables KV.
        for name in (
            "torch_chunk_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_recurrent_gated_delta_rule",
        ):
            self._wrap(self.qwen, name, f"operator.{name}")
        self._wrap(self.torch.linalg, "solve_triangular", "operator.solve_triangular")
        backbone = self.model.model
        self._hook(backbone.visual, "module.visual")
        self._hook(backbone.visual.patch_embed, "module.visual.patch_embed")
        self._hook(backbone.language_model, "module.language_model")
        for name, module in backbone.language_model.named_modules():
            kind = type(module).__name__
            if kind in ("Qwen3_5GatedDeltaNet", "Qwen3_5Attention", "Qwen3_5MLP"):
                self._hook(module, f"module.{kind}.{name}")
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in reversed(self._handles):
            handle.remove()
        for owner, attribute, original in reversed(self._restore):
            setattr(owner, attribute, original)
        return False


def summaries(records, repeats):
    """Aggregate both invocation costs and sums within each full forward."""
    by_label = defaultdict(list)
    grouped = defaultdict(lambda: defaultdict(float))
    for record in records:
        by_label[record["label"]].append(record["wall_ms"])
        grouped[record["label"]][record["iteration"]] += record["wall_ms"]
    result = {}
    for label, values in by_label.items():
        per_forward = [grouped[label][i] for i in range(repeats)]
        result[label] = {
            "calls": len(values),
            "calls_per_forward": len(values) / repeats,
            "call_median_ms": statistics.median(values),
            "call_sum_ms": sum(values),
            "per_forward_sum_ms": per_forward,
            "per_forward_sum_median_ms": statistics.median(per_forward),
        }
    return result


def main():
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    from geospatial_search.embedding_control import gpu_access, idle_collection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument(
        "--events", action="store_true", help="Also record per-scope CUDA event time"
    )
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.baseline_repeats) < 1:
        raise ValueError("Positive warmup/repeat counts are required")
    if (
        not torch.cuda.is_available()
        or not torch.version.hip
        or "9070 XT" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("Expected RX 9070 XT ROCm; no CPU fallback")
    torch.set_num_threads(8)
    torch.manual_seed(20261026)
    paths = json.loads((benchmark.ROOT / "local/model_paths.json").read_text())
    manifest_path = benchmark.ROOT / "local/tiles.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"] = manifest["rows"][:1]
    output = benchmark.ROOT / "results/intern-profile.json"
    output.parent.mkdir(exist_ok=True)
    result = {
        "status": "started",
        "model": "intern",
        "protocol": vars(args),
        "manifest_sha256": benchmark.digest(manifest_path),
        "sample_id": manifest["rows"][0]["sample_id"],
        "timing_note": "Per-scope synchronized diagnostic; nested times are inclusive and not additive. "
        "Compare uninstrumented forwards to estimate instrumentation overhead.",
        "environment": benchmark.environment(),
        "baseline": [],
        "instrumented": [],
    }
    benchmark.save(output, result)
    with ExitStack() as locks:
        locks.enter_context(idle_collection(benchmark.COLLECTION))
        locks.enter_context(gpu_access(benchmark.JOB))
        try:
            images, read_ms = benchmark.load_images(manifest)
            queries = [manifest["rows"][0]["query"]]
            result["source_read_ms"] = read_ms
            started = time.perf_counter()
            adapter = benchmark.adapter("intern", paths)
            torch.cuda.synchronize()
            result["load_seconds"] = time.perf_counter() - started
            result["model_info"] = adapter.info
            result["source"] = _source_record(qwen.torch_chunk_gated_delta_rule)
            result["torchdynamo_exporting"] = bool(qwen.is_torchdynamo_exporting())
            # Preparation and transfer happen once, outside every forward below.
            prepared = adapter.prepare(images, queries)
            batch = benchmark.transfer(prepared, adapter.dtype)
            torch.cuda.synchronize()
            result["input_shapes"] = {
                k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)
            }
            with torch.inference_mode():
                for iteration in range(args.warmup):
                    adapter.forward(batch)
                    torch.cuda.synchronize()
                    print("intern profile warmup", iteration + 1, flush=True)
                for iteration in range(args.baseline_repeats):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    raw = adapter.forward(batch)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    result["baseline"].append(
                        {"forward_wall_ms": elapsed, "answers": adapter.postprocess(raw, batch)}
                    )
                    print("intern uninstrumented", iteration + 1, elapsed, flush=True)
                with SynchronizedProfile(adapter.model, events=args.events) as profile:
                    for iteration in range(args.repeats):
                        profile.iteration = iteration
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        raw = adapter.forward(batch)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - started) * 1000
                        result["instrumented"].append(
                            {"forward_wall_ms": elapsed, "answers": adapter.postprocess(raw, batch)}
                        )
                        print("intern instrumented", iteration + 1, elapsed, flush=True)
                result["operator_sources"] = profile.sources
                result["calls"] = profile.records
                result["summary"] = summaries(profile.records, args.repeats)
                families = []
                for record in profile.records:
                    if record["label"].startswith(
                        (
                            "module.Qwen3_5GatedDeltaNet.",
                            "module.Qwen3_5Attention.",
                            "module.Qwen3_5MLP.",
                        )
                    ):
                        families.append(
                            {**record, "label": ".".join(record["label"].split(".")[:2])}
                        )
                result["module_family_summary"] = summaries(families, args.repeats)
            baseline = statistics.median(row["forward_wall_ms"] for row in result["baseline"])
            instrumented = statistics.median(
                row["forward_wall_ms"] for row in result["instrumented"]
            )
            result["baseline_forward_median_ms"] = baseline
            result["instrumented_forward_median_ms"] = instrumented
            result["instrumentation_wall_ratio"] = instrumented / baseline
            # Matching decisions are expected because wrappers do not change ops.
            probabilities = [
                r["answers"][0]["probability"] for r in result["baseline"] + result["instrumented"]
            ]
            result["maximum_probability_spread"] = max(probabilities) - min(probabilities)
            result["status"] = "ok"
        except Exception as exc:
            import traceback

            result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            print(result["traceback"], flush=True)
        benchmark.save(output, result)
    print(output, flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
