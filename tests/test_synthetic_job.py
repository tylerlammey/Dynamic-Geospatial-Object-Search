"""Synthetic development assets preserve contracts without claiming model inference."""

# Repository script imports follow explicit path initialization.
# ruff: noqa: E402

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))
from create_synthetic_job import (
    COLLECTION,
    DATASET_ID,
    MODEL_ID,
    create_job,
    isolated_path,
    verify_job,
)

from geospatial_search.bundles import validate_bundle
from geospatial_search.download_nys_naip import sha256
from geospatial_search.tiling import load_inventory


class SyntheticJobTests(unittest.TestCase):
    def setUp(self):
        (REPOSITORY / "temp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="synthetic-test-", dir=REPOSITORY / "temp"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_deterministic_masked_bundle_and_provenance(self):
        first, second = self.root / "first", self.root / "second"
        marker = create_job(first)
        self.assertEqual(marker, create_job(second))
        self.assertEqual(verify_job(first)["model_id"], MODEL_ID)
        self.assertEqual(
            {
                str(path.relative_to(first)): sha256(path)
                for path in first.rglob("*")
                if path.is_file()
            },
            {
                str(path.relative_to(second)): sha256(path)
                for path in second.rglob("*")
                if path.is_file()
            },
        )
        catalog = json.loads((first / COLLECTION / "collection.json").read_text())
        self.assertEqual(catalog["chip_count"], 7)  # One masked candidate fails the 95% contract.
        for shard in catalog["shards"]:
            directory = first / COLLECTION / "shards" / shard["shard_id"]
            run = validate_bundle(directory, marker["run_fingerprint"], verify_hashes=True)
            self.assertTrue(run["synthetic"])
            self.assertEqual(run["model_contract"]["model_id"], MODEL_ID)
            rows = pq.read_table(directory / "chips.geoparquet").to_pylist()
            self.assertTrue(
                all(
                    row["dataset_id"] == DATASET_ID and row["valid_fraction"] >= 0.95
                    for row in rows
                )
            )
            self.assertEqual([row["embedding_id"] for row in rows], list(range(len(rows))))
            vectors = np.load(directory / "embeddings.f16.npy", allow_pickle=False)
            self.assertEqual(vectors.dtype, np.float16)
            self.assertTrue(
                np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1, atol=0.002)
            )
        with rasterio.open(first / "imagery/utm18/synthetic-0.tif") as dataset:
            self.assertEqual(dataset.count, 3)
            self.assertEqual(dataset.dtypes, ("uint8",) * 3)
            self.assertEqual(dataset.crs.to_epsg(), 26918)
            self.assertEqual(dataset.res, (0.6, 0.6))
            self.assertEqual(int(np.count_nonzero(dataset.dataset_mask() == 0)), 80 * 80)
        self.assertFalse(list(first.rglob("*.msk")))
        with self.assertRaises(ValueError):
            load_inventory(first)  # A production inventory must reject development data.

    def test_refuses_existing_directory_and_production_paths(self):
        existing = self.root / "existing"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            create_job(existing)
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaises(ValueError):
            isolated_path(REPOSITORY / "data/nys-naip-2022-rgb-060m")
        with self.assertRaises(ValueError):
            isolated_path(REPOSITORY / "temp")

    def test_refuses_redirected_path(self):
        from unittest.mock import patch

        with patch.object(Path, "is_junction", return_value=True):
            with self.assertRaisesRegex(ValueError, "junctions"):
                isolated_path(self.root / "redirected")

    def test_rejects_tampered_bundle_before_service_start(self):
        output = self.root / "job"
        create_job(output)
        run_path = output / COLLECTION / "shards/synthetic-0/run.json"
        run = json.loads(run_path.read_text())
        run["synthetic"] = False
        run_path.write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, "bundle identity"):
            verify_job(output)

    def test_rejects_modified_source_before_service_start(self):
        output = self.root / "job"
        create_job(output)
        source = output / "imagery/utm18/synthetic-0.tif"
        with source.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "source checksum"):
            verify_job(output)


if __name__ == "__main__":
    unittest.main()
