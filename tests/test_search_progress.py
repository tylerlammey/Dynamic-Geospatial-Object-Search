"""Exact-search progress is per invocation and cannot replace CLI rendering."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_tiling_embedding import row

from geospatial_search import search_text
from geospatial_search.bundles import publish_bundle
from geospatial_search.download_nys_naip import Budget


class SearchProgress(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "temp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="nys-search-progress-", dir=temporary_root
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        budget = Budget(self.root, cap=10_000_000, reserve=0)
        entries = []
        for name, vectors in (("a", [[1, 0], [0, 1]]), ("b", [[1, 0], [-1, 0]])):
            source = f"utm18/{name}.tif"
            publish_bundle(
                budget,
                self.root / "shards" / name,
                np.array(vectors, dtype=np.float32),
                [row(0, source), row(1, source)],
                {"run_fingerprint": "run", "embedding_dim": 2, "source_uri": source},
            )
            entries.append({"shard_id": name, "chip_count": 2, "source_uri": source})
        self.catalog = {"run_fingerprint": "run", "chip_count": 4, "shards": entries}
        self.query = np.array([1, 0], dtype=np.float32)

    def test_callback_reports_scan_then_metadata_and_preserves_exact_ties(self):
        updates = []
        with patch.object(
            search_text, "tqdm", side_effect=AssertionError("Callback search used CLI renderer")
        ):
            matches = search_text.exact_search_collection(
                self.root, self.catalog, self.query, 3, progress=updates.append
            )
        self.assertEqual(matches, [("a", 0, 1.0), ("b", 0, 1.0), ("a", 1, 0.0)])
        self.assertEqual(
            updates,
            [
                {"state": "searching", "scanned_sources": 0},
                {"state": "searching", "scanned_sources": 1},
                {"state": "metadata", "scanned_sources": 2},
            ],
        )

    def test_search_wrapper_passes_callback_for_exact_and_auto_fallback(self):
        for backend in ("exact", "auto"):
            with self.subTest(backend=backend):
                updates, details = [], {}
                matches = search_text.search_collection(
                    self.root,
                    self.catalog,
                    self.query,
                    2,
                    backend=backend,
                    exclude_ids=[["a", 0]],
                    diagnostics=details,
                    progress=updates.append,
                )
                self.assertEqual(matches, [("b", 0, 1.0), ("a", 1, 0.0)])
                self.assertEqual(details["backend"], "exact")
                self.assertEqual(updates[-1], {"state": "metadata", "scanned_sources": 2})
                self.assertEqual(len(updates), 3)

    def test_default_cli_renderer_remains_available_after_callback_search(self):
        original = search_text.tqdm
        updates = []
        callback_matches = search_text.exact_search_collection(
            self.root, self.catalog, self.query, 3, progress=updates.append
        )
        self.assertIs(search_text.tqdm, original)
        with patch.object(
            search_text, "tqdm", side_effect=lambda entries, **kwargs: entries
        ) as renderer:
            cli_matches = search_text.exact_search_collection(
                self.root, self.catalog, self.query, 3
            )
            renderer.assert_called_once_with(
                self.catalog["shards"], desc="Searching bundles", unit="tile"
            )
        self.assertEqual(callback_matches, cli_matches)


if __name__ == "__main__":
    unittest.main()
