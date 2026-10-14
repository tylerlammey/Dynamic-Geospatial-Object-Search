"""Run repository synthetic suites and fail when expected tests are skipped."""

from __future__ import annotations

import argparse
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def skipped_are_errors(result: unittest.TestResult) -> bool:
    """The offline contract requires all discovered tests, including Faiss."""
    return bool(result.skipped) or result.testsRun == 0 or not result.wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("backend", "frontend"))
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "frontend/service")]
    try:
        for dependency in ("faiss", "threadpoolctl", "torch"):
            importlib.import_module(dependency)
    except ImportError as error:
        print(f"Required synthetic-test dependency unavailable: {error}", file=sys.stderr)
        print("Run scripts/setup-dev.ps1 for the isolated test environment.", file=sys.stderr)
        return 2
    tests = ROOT / ("tests" if args.suite == "backend" else "frontend/tests")
    suite = unittest.defaultTestLoader.discover(str(tests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print("Skipped tests are a validation failure:", file=sys.stderr)
        for test, reason in result.skipped:
            print(f"  {test}: {reason}", file=sys.stderr)
    return int(skipped_are_errors(result))


if __name__ == "__main__":
    raise SystemExit(main())
