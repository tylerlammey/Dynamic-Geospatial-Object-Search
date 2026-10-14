"""Diagnostics must work without models and must not write to observed jobs."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/doctor.py"
SPEC = importlib.util.spec_from_file_location("doctor", SCRIPT)
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


class Diagnostics(unittest.TestCase):
    def test_missing_job_does_not_create_assets_or_import_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "missing"
            with patch.dict("sys.modules", {"torch": None}):
                report = doctor.observe(
                    job, job / "embeddings", doctor.REPOSITORY / "configs/nys-georsclip.json"
                )
            self.assertFalse(job.exists())
            checks = {item["name"]: item for item in report["checks"]}
            self.assertEqual(checks["collection"]["status"], "warning")
            self.assertEqual(checks["gpu"]["status"], "skipped")

    def test_lock_probe_preserves_bytes_and_does_not_create_absent_file(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "worker.lock"
            self.assertEqual(doctor.lock_state(lock), "absent")
            self.assertFalse(lock.exists())
            lock.write_bytes(b"test-owner")
            self.assertEqual(doctor.lock_state(lock), "idle")
            self.assertEqual(lock.read_bytes(), b"test-owner")

    def test_index_contract_matches_backend_and_rejects_stale_mapping(self):
        from geospatial_search.search_index import snapshot_signature

        catalog = {
            "run_fingerprint": "synthetic",
            "embedding_dim": 16,
            "chip_count": 2,
            "shards": [{"shard_id": "a", "chip_count": 2, "source_uri": "utm18/a.tif"}],
        }
        manifest = {
            "format_version": 1,
            "algorithm": "ivf-pq",
            "snapshot_signature": snapshot_signature(catalog),
            "ntotal": 2,
            "dimension": 16,
            "index_file": "index-" + "a" * 20 + ".faiss",
            "sources": [{"start": 0, "shard_id": "a", "count": 2, "source_uri": "utm18/a.tif"}],
        }
        self.assertEqual(doctor.index_metadata(catalog, manifest), snapshot_signature(catalog))
        manifest["sources"][0]["start"] = 1
        with self.assertRaisesRegex(ValueError, "mapping"):
            doctor.index_metadata(catalog, manifest)
        manifest["sources"][0]["start"] = 0
        manifest["snapshot_signature"] = "stale"
        with self.assertRaisesRegex(ValueError, "snapshot"):
            doctor.index_metadata(catalog, manifest)

    def test_catalog_corruption_is_reported_without_changing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(json.dumps({"run_fingerprint": "a"}))
            (root / "collection.json").write_text(
                json.dumps({"run_fingerprint": "b", "shards": [], "chip_count": 0})
            )
            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()
            }
            report = doctor.observe(root, root, doctor.REPOSITORY / "configs/nys-georsclip.json")
            self.assertEqual(
                next(item for item in report["checks"] if item["name"] == "collection")["status"],
                "error",
            )
            after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()
            }
            self.assertEqual(before, after)

    def test_invalid_config_reports_error_without_cache_lookup_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "invalid.json"
            config.write_text(json.dumps({"model_revision": "a" * 40}))
            report = doctor.observe(root, root, config)
            checks = {item["name"]: item for item in report["checks"]}
            self.assertEqual(checks["config"]["status"], "error")
            self.assertNotIn("model_cache", checks)


if __name__ == "__main__":
    unittest.main()
