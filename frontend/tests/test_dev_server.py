"""Offline HTTP-handler checks for isolated synthetic development, without sockets."""

# Repository script imports follow explicit path initialization.
# ruff: noqa: E402

import io
import json
import sys
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "scripts"))
from create_synthetic_job import DATASET_ID, create_job
from dev_server import SyntheticApplication
from imagery import lonlat_tile
from server import Handler


class SyntheticServiceTests(unittest.TestCase):
    def setUp(self):
        (REPOSITORY / "frontend/temp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="synthetic-test-", dir=REPOSITORY / "frontend/temp"
        )
        root = Path(self.temporary.name)
        self.job = root / "job"
        create_job(self.job)
        self.app = SyntheticApplication(self.job, root / "state", REPOSITORY / "frontend/build")

    def tearDown(self):
        self.app.shutdown()
        self.temporary.cleanup()

    def request(self, route, data=None, authenticated=True, origin=None):
        handler = Handler.__new__(Handler)
        handler.server = SimpleNamespace(application=self.app, server_port=12345)
        handler.connection = None
        handler.path = route
        headers = {"Host": "127.0.0.1:12345"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.app.token}"
        if origin:
            headers["Origin"] = origin
        if data is not None:
            headers["Content-Type"] = "application/json"
        body = b"" if data is None else json.dumps(data).encode()
        headers["Content-Length"] = str(len(body))
        handler.headers = Message()
        for key, value in headers.items():
            handler.headers[key] = value
        handler.rfile = io.BytesIO(body)
        responses = []
        handler.send = lambda status, blob, mime="application/json": responses.append(
            (status, blob)
        )
        (handler.do_GET if data is None else handler.do_POST)()
        return responses[0]

    def test_real_imagery_routes_and_authentication(self):
        self.assertEqual(self.request("/api/status", authenticated=False)[0], 403)
        self.assertEqual(self.request("/api/status", origin="https://example.com")[0], 403)
        _, body = self.request("/api/config")
        config = json.loads(body)
        self.assertTrue(config["synthetic"])
        self.assertIn("SYNTHETIC", config["attribution"])
        self.assertEqual(config["sources"][0]["dataset_id"], DATASET_ID)
        west, south, east, north = self.app.inventory.bounds
        lon, lat = (west + east) / 2, (south + north) / 2
        z, x, y = lonlat_tile(lon, lat, 18)
        status, blob = self.request(f"/api/tiles/{z}/{x}/{y}.png")
        self.assertEqual(status, 200)
        with Image.open(io.BytesIO(blob)) as image:
            self.assertEqual(image.size, (256, 256))
            self.assertGreater(image.convert("RGBA").getextrema()[3][1], 0)
        status, blob = self.request(f"/api/inspect?lon={lon}&lat={lat}")
        self.assertEqual(status, 200)
        chip = json.loads(blob)
        self.assertEqual(chip["pixel_width"], 256)
        status, blob = self.request(
            f"/api/preview?source={chip['source_uri']}&x={chip['pixel_x']}&y={chip['pixel_y']}"
        )
        self.assertEqual(status, 200)
        self.assertTrue(blob.startswith(b"\x89PNG"))

    def test_mock_search_exclusion_history_and_model_without_subprocess(self):
        with patch("server.subprocess.Popen", side_effect=AssertionError("No real model process")):
            self.assertEqual(self.request("/api/model/load", {})[0], 202)
            self.assertEqual(self.app.model_status()["state"], "ready")
            status, body = self.request("/api/search", {"query": "example patterns", "top_k": 3})
            self.assertEqual(status, 202)
            task = json.loads(body)["task_id"]
            status, body = self.request(f"/api/search-progress?id={task}")
            self.assertEqual(json.loads(body)["state"], "complete")
            status, body = self.request(f"/api/results?id={task}")
            results = json.loads(body)
            self.assertTrue(results["synthetic"])
            self.assertEqual(len(results["results"]), 3)
            self.assertEqual(results["results"][0]["geometry"]["type"], "Polygon")
            seen = [
                [row["dataset_id"], row["shard_id"], row["embedding_id"]]
                for row in results["results"]
            ]
            next_task = self.app.start_search(
                {"query": "example patterns", "top_k": 7, "exclude_ids": seen}
            )["task_id"]
            more = self.app.saved(next_task)["results"]
            self.assertEqual(len(more), 4)
            self.assertFalse(
                any(
                    [row["dataset_id"], row["shard_id"], row["embedding_id"]] in seen
                    for row in more
                )
            )
            self.assertEqual(len(self.app.history()), 2)
            status, _ = self.request("/api/history/clear", {"confirm": "all-search-history"})
            self.assertEqual(status, 200)
            self.assertEqual(self.app.history(), [])
            self.assertFalse((self.job / "search-results").exists())
            self.assertEqual(self.request("/api/model/unload", {})[0], 200)
            self.assertEqual(self.app.model_status()["state"], "unloaded")
            self.assertIsNone(self.app.model_worker)

    def test_rejects_production_job_and_reused_state(self):
        with self.assertRaises(ValueError):
            SyntheticApplication(
                REPOSITORY / "data/nys-naip-2022-rgb-060m",
                self.app.frontend.parent / "new",
                self.app.build,
            )
        with self.assertRaises(FileExistsError):
            SyntheticApplication(self.job, self.app.frontend, self.app.build)

    def test_mock_detection_refills_and_returns_native_pixel_boxes(self):
        query = {"query": "synthetic fixture", "top_k": 3}
        original = self.app.saved(self.app.start_search(query)["task_id"])
        status, body = self.request("/api/search", {**query, "detection": {"enabled": True}})
        self.assertEqual(status, 202)
        detected = self.app.saved(json.loads(body)["task_id"])
        self.assertEqual(detected["detection"]["model_id"], "synthetic-mocked-detector")
        self.assertGreater(detected["detection"]["rounds"], 1)
        self.assertEqual(detected["detection"]["evaluated"], 7)
        self.assertEqual(detected["detection"]["stop_reason"], "no_new_candidates")
        self.assertEqual(len(detected["results"]), 6)
        self.assertNotIn(
            original["results"][0]["chip_id"], [r["chip_id"] for r in detected["results"]]
        )
        self.assertEqual(detected["results"][0]["detection"]["maximum_box_score"], 0.92)
        self.assertLess(detected["results"][0]["score"], detected["results"][1]["score"])
        for result in detected["results"]:
            self.assertEqual(result["detection"]["score_kind"], "synthetic_fixture")
            box = result["detection"]["predictions"][0]
            self.assertEqual(
                box["box"], [0, 52, 140, 142] if result["rank"] == 1 else [40, 52, 180, 142]
            )
            self.assertEqual(box["touches_tile_edge"], result["rank"] == 1)
            self.assertEqual(box["geometry"]["type"], "Polygon")
            for lon, lat in box["geometry"]["coordinates"][0]:
                self.assertTrue(result["west"] <= lon <= result["east"])
                self.assertTrue(result["south"] < lat < result["north"])

        status, body = self.request(
            "/api/search",
            {
                **query,
                "detection": {"enabled": True, "max_candidates": 2},
            },
        )
        self.assertEqual(status, 202)
        capped = self.app.saved(json.loads(body)["task_id"])
        self.assertEqual(len(capped["results"]), 1)
        self.assertEqual(capped["detection"]["evaluated"], 2)
        self.assertEqual(capped["detection"]["stop_reason"], "candidate_limit")

        status, body = self.request(
            "/api/search",
            {
                **query,
                "detection": {
                    "enabled": True,
                    "box_threshold": 1,
                    "window_size": 2,
                    "min_candidates": 4,
                },
            },
        )
        self.assertEqual(status, 202)
        filtered = self.app.saved(json.loads(body)["task_id"])
        self.assertEqual(filtered["results"], [])
        self.assertEqual(filtered["detection"]["evaluated"], 5)
        self.assertEqual(filtered["detection"]["stop_reason"], "low_detection_yield")
        self.assertEqual(
            self.request("/api/search", {**query, "detection": {"enabled": 1}})[0], 400
        )


if __name__ == "__main__":
    unittest.main()
