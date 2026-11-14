"""Exercise the production search worker with native synthetic windows and a fake detector."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import rasterio
from pyproj import Transformer
from retrieval import ExactRetrieval, validate_detection_request
from search_chips import SearchChips

from geospatial_search import search_text
from geospatial_search.embedding_control import idle_collection
from geospatial_search.tiling import read_rgb

ROOT = Path(__file__).resolve().parents[2]


class DetectionRetrieval(unittest.TestCase):
    def setUp(self):
        with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
            from create_synthetic_job import COLLECTION, create_job

        scratch = ROOT / "frontend/temp"
        scratch.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(directory.cleanup)
        frontend = Path(directory.name) / "frontend"
        frontend.mkdir()
        self.adapter = ExactRetrieval(ROOT, frontend)
        self.adapter.job = Path(directory.name) / "job"
        self.marker = create_job(self.adapter.job)
        self.adapter.collection = self.adapter.job / COLLECTION
        self.root_run, self.catalog = search_text.read_collection(self.adapter.collection)
        self.shard = self.catalog["shards"][1]["shard_id"]
        self.query = "synthetic aircraft"
        key = hashlib.sha256(
            (self.catalog["run_fingerprint"] + "\0" + self.query).encode()
        ).hexdigest()
        vectors = frontend / "state/text-vectors"
        vectors.mkdir(parents=True)
        vector = [1.0] + [0.0] * 15
        (vectors / f"{key}.json").write_text(
            json.dumps({"run_fingerprint": self.catalog["run_fingerprint"], "vector": vector}),
            encoding="utf-8",
        )

    def test_ann_refills_and_preserves_native_pixels_and_projected_boxes(self):
        from rasterio.windows import Window

        calls = []
        box = [32.0, 40.0, 80.0, 98.0]

        def fetch(entries, vector, count, *, backend, effort, exclude_ids, progress):
            dataset = entries[0][0]
            self.assertEqual(backend, "ann")
            self.assertEqual(effort, "balanced")
            self.assertEqual(vector[0], 1)
            excluded = set(map(tuple, exclude_ids))
            calls.append(excluded)
            rows = [
                (dataset, self.shard, row_id, 0.9 - 0.1 * row_id)
                for row_id in range(4)
                if (dataset.dataset_id, self.shard, row_id) not in excluded
            ]
            return rows[:count], {
                "backend": "ivf-pq",
                "approximate": True,
                "effort": "balanced",
                "search_seconds": 0.01,
                "collections": [],
            }

        detector = Mock()
        detector.info = {"model_id": "synthetic-detector"}
        seen_images = []
        mock_yield = iter((False, True, True))

        def predict(images, query, *, box_threshold, text_threshold, nms_iou):
            self.assertEqual(query, self.query)
            self.assertEqual((box_threshold, text_threshold, nms_iou), (0.30, 0.25, 0.50))
            self.assertLessEqual(len(images), 2)
            outputs = []
            for image in images:
                seen_images.append(np.asarray(image))
                accepted = next(mock_yield)
                outputs.append(
                    {
                        "predictions": [{"box": box, "score": 0.6, "text_label": query}]
                        if accepted
                        else [],
                        "maximum_box_score": 0.6 if accepted else 0.0,
                        "score_kind": "synthetic-box-score",
                    }
                )
            return outputs

        detector.predict.side_effect = predict
        updates = []
        request = {
            "query": self.query,
            "top_k": 1,
            "include_supplements": False,
            "search_backend": "ann",
            "exclude_ids": [[self.shard, 3]],
            "detection": {"enabled": True, "fetch_size": 2, "max_candidates": 3},
        }
        with (
            patch("geospatial_search.federated.search", side_effect=fetch),
            patch.object(self.adapter, "load_detector", return_value=detector),
        ):
            output = self.adapter.text(request, updates.append)

        dataset_id = self.root_run["recipe"]["dataset_id"]
        self.assertEqual(
            calls,
            [
                {(dataset_id, self.shard, 3)},
                {(dataset_id, self.shard, i) for i in (0, 1, 3)},
            ],
        )
        self.assertEqual([row["embedding_id"] for row in output["results"]], [1, 2])
        self.assertEqual(len(seen_images), 3)
        self.assertEqual(output["detection"]["evaluated"], 3)
        self.assertEqual(output["detection"]["rounds"], 2)
        self.assertTrue(any(update["state"] == "detecting" for update in updates))
        detector.close.assert_called_once()
        accepted = output["results"][0]
        source = self.adapter.job / "imagery" / accepted["source_uri"]
        with rasterio.open(source) as dataset:
            original = read_rgb(dataset, Window(accepted["pixel_x"], accepted["pixel_y"], 256, 256))
            np.testing.assert_array_equal(seen_images[1], np.asarray(original))
            transform = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
            corner = dataset.transform * (accepted["pixel_x"] + 32, accepted["pixel_y"] + 40)
            expected = transform.transform(*corner)
        prediction = accepted["detection"]["predictions"][0]
        self.assertEqual(prediction["box"], box)
        np.testing.assert_allclose(prediction["geometry"]["coordinates"][0][0], expected)
        with idle_collection(self.adapter.collection):
            pass  # Completion releases the original collection lock.

    def test_exact_fallback_reuses_one_scan_when_first_page_has_no_boxes(self):
        from geospatial_search.federated import search

        detector = Mock()
        detector.info = {"model_id": "synthetic-detector"}
        detector.predict.side_effect = [
            [{"predictions": [], "maximum_box_score": 0, "score_kind": "synthetic"}],
            [
                {
                    "predictions": [{"box": [1, 2, 3, 4], "score": 0.6, "text_label": "plane"}],
                    "maximum_box_score": 0.6,
                    "score_kind": "synthetic",
                },
            ],
        ]
        with (
            patch("geospatial_search.federated.search", wraps=search) as fetch,
            patch.object(self.adapter, "load_detector", return_value=detector),
        ):
            output = self.adapter.text(
                {
                    "query": self.query,
                    "top_k": 1,
                    "detection": {"fetch_size": 1, "max_candidates": 2},
                },
                lambda _: None,
            )
        fetch.assert_called_once()
        self.assertEqual(output["search"]["backend"], "exact")
        self.assertEqual(output["detection"]["rounds"], 2)
        self.assertEqual(output["detection"]["evaluated"], 2)
        self.assertEqual(len(output["results"]), 1)

    def test_legacy_request_is_similarity_only_and_does_not_load_detector(self):
        with patch.object(self.adapter, "load_detector") as model:
            output = self.adapter.text(
                {"query": self.query, "search_backend": "exact", "top_k": 1}, lambda _: None
            )
        model.assert_not_called()
        self.assertEqual(len(output["results"]), 1)
        self.assertNotIn("detection", output)
        self.assertNotIn("detection", output["results"][0])

    def test_source_verification_rejects_a_conflicting_hash_for_a_cached_path(self):
        from geospatial_search.gapfill import Collection

        dataset = Collection("test", self.adapter.collection, self.adapter.job / "imagery", 2022)
        matches = [(dataset, self.shard, 0, 1.0)]
        chips = SearchChips()
        chips.images(matches)
        key = ("test", self.shard, 0)
        chips.rows[key]["source_sha256"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "checksum"):
            chips.images(matches)

    def test_reject_invalid_detection_payloads(self):
        for value in (None, [], True, {"enabled": 1}, {"box_threshold": float("nan")}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_detection_request({"detection": value}, 25)
