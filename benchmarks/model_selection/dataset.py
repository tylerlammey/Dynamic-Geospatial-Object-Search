"""Reference published embedded rows; decode the original native windows in RAM."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))
JOB = REPO / "data/nys-naip-2022-rgb-060m"
COLLECTION = JOB / "embeddings/georsclip-vith14-s256-st192"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build():
    import numpy as np
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_limits

    from geospatial_search.embedding_control import idle_collection
    from geospatial_search.search_index import snapshot_signature
    from geospatial_search.search_text import read_collection, search_collection

    with idle_collection(COLLECTION), threadpool_limits(limits=1, user_api="blas"):
        recipe, catalog = read_collection(COLLECTION)
        query_path = JOB / "verification/search-selection/text-queries.json"
        cached = json.loads(query_path.read_text())
        if cached["run_fingerprint"] != catalog["run_fingerprint"]:
            raise ValueError("Cached query vectors use a different collection recipe")
        rng = random.Random(20261026)
        rows, diagnostics = [], []
        for q in cached["queries"]:
            stats = {}
            matches = search_collection(
                COLLECTION,
                catalog,
                np.asarray(q["vector"], dtype=np.float32),
                500,
                backend="ann",
                effort="balanced",
                diagnostics=stats,
            )
            selections = [(rank + 1, matches[rank], "ann") for rank in [0, 1, 4, 19, 99, 249, 499]]
            source = rng.choice(catalog["shards"])
            selections.append(
                (None, (source["shard_id"], rng.randrange(source["chip_count"]), None), "uniform")
            )
            for rank, (shard, row_id, score), stratum in selections:
                table = pq.read_table(COLLECTION / "shards" / shard / "chips.geoparquet")
                row = table.slice(int(row_id), 1).to_pylist()[0]
                if (
                    row["embedding_id"] != row_id
                    or row["pixel_width"] != 256
                    or row["pixel_height"] != 256
                ):
                    raise ValueError("Published row identity or native resolution mismatch")
                identity = f"nys_naip_2022_rgb060/{shard}/{row_id}"
                sample_id = hashlib.sha256((identity + "|" + q["query"]).encode()).hexdigest()[:20]
                group_id = shard  # Entire source stays in one split; neighboring rows overlap.
                rows.append(
                    {
                        "sample_id": sample_id,
                        "tile_id": identity,
                        "query": q["query"],
                        "dataset_id": "nys_naip_2022_rgb060",
                        "shard_id": shard,
                        "embedding_id": int(row_id),
                        "source_uri": row["source_uri"],
                        "source_sha256": row["source_sha256"],
                        "pixel_x": row["pixel_x"],
                        "pixel_y": row["pixel_y"],
                        "pixel_width": 256,
                        "pixel_height": 256,
                        "ann_rank": rank,
                        "ann_score": score,
                        "stratum": stratum,
                        "group_id": group_id,
                        "split": "test"
                        if int(hashlib.sha256(shard.encode()).hexdigest(), 16) % 2
                        else "calibration",
                        "label": None,
                        "boxes": [],
                        "exhaustive": False,
                    }
                )
            diagnostics.append({"query": q["query"], **stats})
            print("Selected", q["query"], flush=True)
        result = {
            "schema_version": 1,
            "seed": 20261026,
            "run_fingerprint": catalog["run_fingerprint"],
            "snapshot_signature": snapshot_signature(catalog),
            "query_vectors_sha256": digest(query_path),
            "catalog_sha256": digest(COLLECTION / "collection.json"),
            "selection": diagnostics,
            "rows": rows,
        }
        target = ROOT / "local/tiles.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
        return result


def load_images(manifest, verify_sources=False):
    """Use the identical production read_rgb helper, with no intermediate PNGs."""
    import numpy as np
    import pyarrow.parquet as pq

    if (
        manifest.get("catalog_sha256")
        and digest(COLLECTION / "collection.json") != manifest["catalog_sha256"]
    ):
        raise ValueError("Published collection catalog changed since benchmark tile selection")
    import rasterio
    from rasterio.windows import Window

    from geospatial_search.tiling import read_rgb

    images, timings, verified, tables = [], [], set(), {}
    for row in manifest["rows"]:
        shard = row["shard_id"]
        if shard not in tables:
            tables[shard] = pq.read_table(COLLECTION / "shards" / shard / "chips.geoparquet")
        published = tables[shard].slice(row["embedding_id"], 1).to_pylist()[0]
        for key in [
            "embedding_id",
            "source_uri",
            "source_sha256",
            "pixel_x",
            "pixel_y",
            "pixel_width",
            "pixel_height",
        ]:
            if row[key] != published[key]:
                raise ValueError(f"Manifest does not describe the embedded row: {key}")
        source = (JOB / "imagery" / row["source_uri"]).resolve()
        if not source.is_relative_to((JOB / "imagery").resolve()):
            raise ValueError("Source escapes immutable imagery")
        if verify_sources and source not in verified:
            if digest(source) != row["source_sha256"]:
                raise ValueError("Source checksum mismatch")
            verified.add(source)
        start = time.perf_counter()
        with rasterio.open(source) as dataset:
            window = Window(row["pixel_x"], row["pixel_y"], 256, 256)
            if np.count_nonzero(dataset.dataset_mask(window=window)) / (256 * 256) < 0.95:
                raise ValueError("Published tile no longer satisfies mask coverage")
            image = read_rgb(dataset, window)
        timings.append((time.perf_counter() - start) * 1000)
        images.append(image)
    return images, timings


def sheets(manifest):
    from PIL import Image, ImageDraw

    images, _ = load_images(manifest, verify_sources=True)
    queries = list(dict.fromkeys(row["query"] for row in manifest["rows"]))
    for qi, query in enumerate(queries):
        indices = [i for i, r in enumerate(manifest["rows"]) if r["query"] == query]
        canvas = Image.new("RGB", (4 * 256, 2 * 284 + 28), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f"{qi}: {query} | source pixels 256 x 256, no labels yet", fill="black")
        for j, i in enumerate(indices):
            x, y = (j % 4) * 256, (j // 4) * 284 + 28
            canvas.paste(images[i], (x, y))
            row = manifest["rows"][i]
            draw.text((x + 3, y + 258), f"{i}: rank {row['ann_rank'] or 'uniform'}", fill="black")
        canvas.save(ROOT / "local" / f"sheet-{qi:02d}.jpg", quality=95)
    print("Wrote small review sheets only; no image dataset copied.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", action="store_true")
    args = parser.parse_args()
    path = ROOT / "local/tiles.json"
    manifest = json.loads(path.read_text()) if path.exists() else build()
    if args.sheets:
        sheets(manifest)
