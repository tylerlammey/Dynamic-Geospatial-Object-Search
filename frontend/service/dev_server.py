"""Isolated synthetic UI service: real imagery display, mocked model and text encoding.

This entry point is never selected by the desktop or production service. All
state is beneath ignored temp folders and no model subprocess is started.
"""

# Repository-anchored imports follow path initialization for direct script execution.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "scripts"))
sys.path.insert(0, str(REPOSITORY / "src"))
from create_synthetic_job import COLLECTION, DATASET_ID, isolated_path, verify_job
from imagery import Inventory, TileRenderer
from retrieval import validate_query, validate_search_options
from server import Application, Server
from storage import DiskCache, atomic_json, contained, frontend_ownership


class SyntheticApplication(Application):
    def __init__(self, job, state, build):
        self.job = isolated_path(Path(job))
        self.marker = verify_job(self.job)
        self.frontend = isolated_path(Path(state))
        if (
            self.frontend == self.job
            or self.frontend.is_relative_to(self.job)
            or self.job.is_relative_to(self.frontend)
        ):
            raise ValueError("Synthetic job and service state must use separate directories")
        # State must be new: never attach this entry point to a production owner.
        self.frontend.mkdir(parents=True, exist_ok=False)
        atomic_json(
            self.frontend / "synthetic-service.json",
            {"synthetic": True, "dataset_id": DATASET_ID},
            self.frontend,
        )
        self.repository = REPOSITORY
        self.collection = self.job / COLLECTION
        self.build = Path(build).resolve()
        self.settings = {
            "cache_limit_bytes": 20_000_000,
            "imagery_workers": 2,
            "offline_model": True,
        }
        self.cache = DiskCache(self.frontend / "cache", self.settings["cache_limit_bytes"])
        self.inventory = Inventory(self.job, strict=False, include_supplements=False)
        # Inventory's production defaults are replaced only on this instance.
        self.inventory.sources = [
            replace(source, dataset_id=DATASET_ID, release_year=None, acquisition_date=None)
            for source in self.inventory.sources
        ]
        self.inventory.base_sources = self.inventory.sources
        self.inventory.by_uri = {source.uri: source for source in self.inventory.sources}
        self.inventory.datasets = [
            {"dataset_id": DATASET_ID, "release_year": None, "sources": len(self.inventory.sources)}
        ]
        self.renderer = TileRenderer(self.inventory, self.cache, 2)
        self.base_inventory = self.inventory
        self.base_renderer = self.renderer
        self.imagery_error = None
        self.token = secrets.token_urlsafe(32)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="synthetic-imagery")
        self.map_slots = threading.BoundedSemaphore(14)
        self.search_lock = threading.Lock()
        self.search = self.search_id = self.search_log = None
        self.model_worker = self.model_worker_id = self.model_log = None
        self.state_lock = threading.RLock()
        self.poll_lock = threading.Lock()
        self.last_status = None
        self.status_at = 0
        self.overview_lock = threading.Lock()
        self.overview_thread = None
        self.overview_cancel = threading.Event()
        self.overview_status = {"state": "idle", "ready": False}
        self.mock_model_state = "unloaded"
        self.tasks = {}
        for name in ("logs", "state/searches", "temp"):
            (self.frontend / name).mkdir(parents=True, exist_ok=True)
        self.catalog = json.loads((self.collection / "collection.json").read_text(encoding="utf-8"))

    def config(self, base_only=False):
        config = super().config(base_only)
        config["imagery"]["year"] = None
        return {
            **config,
            "synthetic": True,
            "attribution": "SYNTHETIC DEVELOPMENT ONLY · generated patterns; mocked model/search",
        }

    def status(self, base_only=False):
        return {
            "state": "complete",
            "collection_state": "complete",
            "busy": False,
            "partial": False,
            "synthetic": True,
            "collection_busy": False,
            "gpu_busy": False,
            "published_sources": len(self.catalog["shards"]),
            "total_sources": self.catalog["total_sources"],
            "published_chips": self.catalog["chip_count"],
            "run_fingerprint": self.catalog["run_fingerprint"],
            "coverage": [shard["source_uri"] for shard in self.catalog["shards"]],
            "model": self.model_status(),
            "cache": self.cache.stats(),
            "imagery_error": None,
            "metrics": self.renderer.metrics,
            "search_index": {"ready": False, "algorithm": "synthetic-exact"},
            "overview": {
                **self.overview_status,
                "ready": self.renderer.overview_ready(),
                "epoch": self.renderer.display_epoch,
            },
        }

    def model_status(self):
        return {
            "state": self.mock_model_state,
            "synthetic": True,
            "gpu": "MOCK · no GPU or model loaded",
        }

    def start_model(self, value):
        if type(value.get("allow_partial", False)) is not bool:
            raise ValueError("allow_partial must be boolean")
        self.mock_model_state = "ready"
        return self.model_status()

    def stop_model(self, force=False):
        self.mock_model_state = "unloaded"
        return self.model_status()

    def backend_history(self):
        return []

    def clear_history(self, include_backend=False):
        return super().clear_history(include_backend=False)

    def saved(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch("[a-f0-9]{32}", identifier):
            raise ValueError("Invalid synthetic result ID")
        return json.loads(
            contained(
                self.frontend, self.frontend / "state/searches" / f"{identifier}.json"
            ).read_text(encoding="utf-8")
        )

    def start_search(self, value):
        import numpy as np
        from shapely import from_wkb

        from geospatial_search.bundles import selected_rows, top_matches, validate_bundle

        query, top_k, _ = validate_query(value)
        _, _, seen = validate_search_options(value)
        if type(value.get("include_supplements", True)) is not bool:
            raise ValueError("include_supplements must be boolean")
        identifier = uuid.uuid4().hex
        seed = int.from_bytes(hashlib.sha256(query.encode()).digest()[:8], "big")
        vector = np.random.default_rng(seed).standard_normal(16, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        results = []
        with self.search_lock:
            for shard in self.catalog["shards"]:
                folder = contained(self.collection, self.collection / "shards" / shard["shard_id"])
                run = validate_bundle(folder, self.marker["run_fingerprint"], verify_hashes=True)
                if (
                    not run.get("synthetic")
                    or run["model_contract"]["model_id"] != self.marker["model_id"]
                ):
                    raise ValueError("Synthetic service cannot search production bundles")
                vectors = np.load(folder / "embeddings.f16.npy", allow_pickle=False)
                matches = top_matches(vectors, vector, len(vectors))
                rows = selected_rows(folder / "chips.geoparquet", [index for index, _ in matches])
                for index, score in matches:
                    if [DATASET_ID, shard["shard_id"], index] in seen or [
                        shard["shard_id"],
                        index,
                    ] in seen:
                        continue
                    row = rows[index]
                    geometry = from_wkb(row.pop("geometry")).__geo_interface__
                    results.append(
                        {
                            **row,
                            "geometry": geometry,
                            "shard_id": shard["shard_id"],
                            "score": score,
                            "synthetic": True,
                            "release_year": None,
                            "acquisition_date": None,
                        }
                    )
            results.sort(key=lambda row: (-row["score"], row["shard_id"], row["embedding_id"]))
            results = [{**row, "rank": rank} for rank, row in enumerate(results[:top_k], 1)]
            data = {
                "query": query,
                "result_id": identifier,
                "synthetic": True,
                "warning": self.marker["warning"],
                "run_fingerprint": self.marker["run_fingerprint"],
                "results": results,
                "searched_source_tiles": len(self.catalog["shards"]),
                "total_source_tiles": self.catalog["total_sources"],
                "searched_chip_count": self.catalog["chip_count"],
                "searched_source_uris": [shard["source_uri"] for shard in self.catalog["shards"]],
                "partial_collection": False,
                "created_utc": time.time(),
                "search": {
                    "backend": "synthetic-mocked-text",
                    "approximate": False,
                    "search_seconds": 0,
                },
            }
            atomic_json(
                self.frontend / "state/searches" / f"{identifier}.json", data, self.frontend
            )
            self.tasks[identifier] = {
                "state": "complete",
                "result_id": identifier,
                "synthetic": True,
            }
        return {"task_id": identifier, "synthetic": True}

    def search_progress(self, identifier):
        if identifier not in self.tasks:
            raise ValueError("Unknown synthetic search task")
        return self.tasks[identifier]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument(
        "--state",
        type=Path,
        default=REPOSITORY / "frontend/temp" / f"synthetic-service-{uuid.uuid4().hex}",
    )
    parser.add_argument("--build", type=Path, default=REPOSITORY / "frontend/build")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if not (args.build / "index.html").is_file():
        parser.error("UI build is missing; run frontend/scripts/build.ps1 first")
    app = SyntheticApplication(args.job, args.state, args.build)
    ready = contained(app.frontend, args.ready_file or app.frontend / "ready.json")
    with frontend_ownership(app.frontend):
        server = Server(("127.0.0.1", args.port), app)
        atomic_json(
            ready, {"port": server.server_port, "token": app.token, "synthetic": True}, app.frontend
        )
        west, south, east, north = app.inventory.bounds
        print("SYNTHETIC DEVELOPMENT ONLY · no real imagery, semantic model, or GPU", flush=True)
        print(
            f"http://127.0.0.1:{server.server_port}/?view={(west + east) / 2},{(south + north) / 2},18#{app.token}",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            app.shutdown()
            ready.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
