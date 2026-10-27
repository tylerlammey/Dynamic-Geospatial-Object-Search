"""Run models in separate processes, one GPU owner at a time; preserve failures."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        default=[
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
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--batches", type=int, nargs="+", default=None)
    parser.add_argument("--suite-name", default="baseline")
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-predictions", action="store_true")
    args = parser.parse_args()
    records = []
    for key in args.models:
        log = ROOT / "local" / (key + ".log")
        command = [
            sys.executable,
            "-u",
            "-B",
            str(ROOT / "run.py"),
            "--model",
            key,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        batches = args.batches or ([1, 2, 4] if key == "llmdet" else [1, 4, 8])
        command += ["--batches", *map(str, batches)]
        if args.tag:
            command += ["--tag", args.tag]
        if args.skip_predictions:
            command += ["--skip-predictions"]
        print("Starting", key, flush=True)
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as stream:
            # Bounded timeout protects against unsupported driver/operator hangs.
            try:
                process = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800
                )
                status = {"exit_code": process.returncode}
            except subprocess.TimeoutExpired:
                status = {"timeout_seconds": 1800}
        path = ROOT / "results" / (key + ("-" + args.tag if args.tag else "") + ".json")
        result = json.loads(path.read_text()) if path.exists() else {}
        if result.get("status") == "started":
            result.update(status="native_crash_or_interruption", process=status)
            path.write_text(json.dumps(result, indent=2) + "\n")
        records.append(
            {
                "model": key,
                **status,
                "wall_seconds": time.perf_counter() - start,
                "benchmark_status": result.get("status"),
                "log": log.relative_to(ROOT).as_posix(),
            }
        )
        suite_name = (
            "suite.json" if args.suite_name == "baseline" else f"suite-{args.suite_name}.json"
        )
        (ROOT / "results" / suite_name).write_text(json.dumps(records, indent=2) + "\n")
        print("Finished", records[-1], flush=True)
    return int(any(r.get("exit_code") != 0 for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
