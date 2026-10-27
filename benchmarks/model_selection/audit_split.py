"""Read-only spatial split audit of the benchmark's published embedded rows.

Run only when the parent's GPU benchmark has finished: this script acquires the
same collection ownership lock, without loading Torch, decoding imagery, or
changing any collection. The only output is results/split-audit.json by default.

Example from the repository root:
    .\\.venv-dev\\Scripts\\python.exe -B benchmarks/model_selection/audit_split.py

Overlap means intersection area divided by the smaller ACTUAL stored polygon's
area exceeds 0.001 (0.1%). Areas are evaluated in the published EPSG:4326 planar
coordinates, so ratios are a local overlap diagnostic, not geodesic area values.
Different queries of one qualified tile share one geometry; a shared tile across
splits is reported separately even though it is not a distinct geometry pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from dataset import COLLECTION, JOB, ROOT, digest

COLLECTIONS = {
    "nys_naip_2022_rgb060": COLLECTION,
    "nys_naip_2021_release_gapfill_rgb060": (
        JOB / "supplements/naip-2021-release-gapfill/embeddings/georsclip-vith14-s256-st192"
    ),
}


def _identity(row):
    dataset_id, shard_id, embedding_id = row["dataset_id"], row["shard_id"], row["embedding_id"]
    if dataset_id not in COLLECTIONS:
        raise ValueError(f"Unsupported dataset_id: {dataset_id!r}")
    if not isinstance(shard_id, str) or not shard_id:
        raise ValueError("Each qualified row requires a nonempty shard_id")
    if type(embedding_id) is not int or embedding_id < 0:
        raise ValueError("Each qualified row requires a nonnegative integer embedding_id")
    return dataset_id, shard_id, embedding_id


def _describe(identity, record):
    return {
        "dataset_id": identity[0],
        "shard_id": identity[1],
        "embedding_id": identity[2],
        "splits": sorted(record["splits"]),
        "queries": sorted(record["queries"]),
    }


def _shared(mapping, names):
    result = []
    for identity, splits in sorted(mapping.items()):
        if len(splits) > 1:
            values = identity if isinstance(identity, tuple) else (identity,)
            result.append({**dict(zip(names, values, strict=True)), "splits": sorted(splits)})
    return result


def audit(manifest, overlap_fraction=0.001):
    """Acquire collection locks, verify published identities, and audit polygons."""
    import pyarrow.parquet as pq
    from shapely import from_wkb
    from shapely.strtree import STRtree

    from geospatial_search.embedding_control import idle_collection

    if not math.isfinite(overlap_fraction) or not 0 <= overlap_fraction <= 1:
        raise ValueError("overlap_fraction must be finite and in [0,1]")
    rows = manifest["rows"]
    if not rows:
        raise ValueError("The benchmark manifest has no rows")
    identities = [_identity(row) for row in rows]
    datasets = sorted({identity[0] for identity in identities})
    unique = {}
    sources = defaultdict(set)
    tiles = defaultdict(set)
    groups = defaultdict(set)
    counts = defaultdict(lambda: defaultdict(lambda: {"positive": 0, "negative": 0, "unknown": 0}))
    tables = {}
    catalog_hashes = {}

    with ExitStack() as locks:
        for dataset_id in datasets:
            directory = COLLECTIONS[dataset_id]
            locks.enter_context(idle_collection(directory))
            catalog_hashes[dataset_id] = digest(directory / "collection.json")

        for row, identity in zip(rows, identities, strict=True):
            split = row["split"]
            query = row["query"]
            if not isinstance(split, str) or not split:
                raise ValueError("Each row needs a nonempty split")
            if not isinstance(query, str) or not query:
                raise ValueError("Each row needs a nonempty query")
            label = row.get("label")
            if label is not None and label not in (0, 1):
                raise ValueError("Labels must be 0, 1, or null")
            label_name = "unknown" if label is None else "positive" if label == 1 else "negative"
            counts[split][query][label_name] += 1
            sources[identity[:2]].add(split)
            tiles[identity].add(split)
            if row.get("group_id"):
                groups[str(row["group_id"])].add(split)
            if identity in unique:
                unique[identity]["splits"].add(split)
                unique[identity]["queries"].add(query)
                continue

            dataset_id, shard_id, embedding_id = identity
            table_key = identity[:2]
            if table_key not in tables:
                shards = (COLLECTIONS[dataset_id] / "shards").resolve()
                path = (shards / shard_id / "chips.geoparquet").resolve()
                if not path.is_relative_to(shards):
                    raise ValueError("Manifest shard escapes the published collection")
                tables[table_key] = pq.read_table(
                    path,
                    columns=[
                        "embedding_id",
                        "geometry",
                        "source_uri",
                        "pixel_x",
                        "pixel_y",
                        "pixel_width",
                        "pixel_height",
                    ],
                )
            table = tables[table_key]
            if embedding_id >= table.num_rows:
                raise ValueError("Manifest row exceeds the published shard")
            published = table.slice(embedding_id, 1).to_pylist()[0]
            for field in (
                "embedding_id",
                "source_uri",
                "pixel_x",
                "pixel_y",
                "pixel_width",
                "pixel_height",
            ):
                if row[field] != published[field]:
                    raise ValueError(
                        f"Manifest differs from its published row: {identity}, {field}"
                    )
            polygon = from_wkb(published["geometry"])
            if (
                polygon.geom_type not in {"Polygon", "MultiPolygon"}
                or polygon.is_empty
                or not polygon.is_valid
                or polygon.area <= 0
            ):
                raise ValueError(f"Invalid published WGS84 footprint: {identity}")
            unique[identity] = {"polygon": polygon, "splits": {split}, "queries": {query}}

        ordered = sorted(unique.items())
        geometries = [record["polygon"] for _, record in ordered]
        tree = STRtree(geometries)
        overlaps = []
        cross_split_intersections = 0
        maximum_overlap_fraction = 0.0
        for index, (identity, record) in enumerate(ordered):
            polygon = record["polygon"]
            candidates = tree.query(polygon, predicate="intersects")
            for other_index in sorted(int(value) for value in candidates):
                if other_index <= index:
                    continue
                other_identity, other = ordered[other_index]
                if not any(
                    first != second for first in record["splits"] for second in other["splits"]
                ):
                    continue
                intersection_area = polygon.intersection(other["polygon"]).area
                if intersection_area <= 0:
                    continue
                cross_split_intersections += 1
                fraction = intersection_area / min(polygon.area, other["polygon"].area)
                maximum_overlap_fraction = max(maximum_overlap_fraction, fraction)
                if fraction > overlap_fraction:
                    overlaps.append(
                        {
                            "first": _describe(identity, record),
                            "second": _describe(other_identity, other),
                            "intersection_over_smaller_polygon_area": fraction,
                        }
                    )

    shared_sources = _shared(sources, ("dataset_id", "shard_id"))
    shared_tiles = _shared(tiles, ("dataset_id", "shard_id", "embedding_id"))
    shared_groups = _shared(groups, ("group_id",))
    return {
        "schema_version": 1,
        "status": "complete",
        "input_pairs": len(rows),
        "unique_qualified_tiles": len(unique),
        "unique_sources": len(sources),
        "catalog_sha256": catalog_hashes,
        "geometry": "Published GeoParquet WKB polygons in EPSG:4326; no bounds-only approximation",
        "overlap_threshold_fraction": overlap_fraction,
        "cross_split_positive_area_intersections": cross_split_intersections,
        "maximum_cross_split_overlap_fraction": maximum_overlap_fraction,
        "cross_split_overlap_pairs_above_threshold": overlaps,
        "cross_split_shared_sources": shared_sources,
        "cross_split_shared_tiles": shared_tiles,
        "cross_split_shared_group_ids": shared_groups,
        "query_class_counts_by_split": {
            split: dict(sorted(queries.items())) for split, queries in sorted(counts.items())
        },
        "leakage_detected": bool(overlaps or shared_sources or shared_tiles or shared_groups),
        "limitations": [
            "Area ratios use WGS84 planar coordinates, not geodesic square meters.",
            "No observed overlap does not establish geographic independence or validate labels.",
            "Distinct nearby tiles may remain correlated despite having disjoint footprints.",
            "Class counts include paired queries on reused tiles; they are not independent image counts.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "local/tiles.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/split-audit.json")
    parser.add_argument("--overlap-fraction", type=float, default=0.001)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Audit output must remain in the dedicated benchmark results folder")
    manifest_path = args.manifest.resolve()
    if output == manifest_path:
        raise ValueError("The read-only audit cannot replace its input manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    result = audit(manifest, args.overlap_fraction)
    result["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    result["audit_code_sha256"] = digest(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "unique_tiles": result["unique_qualified_tiles"],
                "overlap_pairs": len(result["cross_split_overlap_pairs_above_threshold"]),
                "shared_sources": len(result["cross_split_shared_sources"]),
                "shared_tiles": len(result["cross_split_shared_tiles"]),
                "leakage_detected": result["leakage_detected"],
                "output": str(output),
            }
        )
    )
    return int(result["leakage_detected"])


if __name__ == "__main__":
    raise SystemExit(main())
