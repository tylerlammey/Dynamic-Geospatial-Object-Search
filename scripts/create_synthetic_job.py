"""Generate a tiny offline development job; its vectors are not GeoRSCLIP output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

DATASET_ID = "synthetic-development-only"
MODEL_ID = "synthetic-seeded-unit-vectors-v1"
MARKER = "synthetic-development-job.json"
COLLECTION = "embeddings/synthetic-unit-vectors"


def isolated_path(path: Path) -> Path:
    """Only ignored development folders are writable, with no redirected ancestors."""
    path = Path(path).absolute()
    if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
        raise ValueError("Synthetic paths cannot contain symlinks or junctions")
    path = path.resolve()
    roots = (REPOSITORY / "temp", REPOSITORY / "frontend/temp")
    if not any(path != root and path.is_relative_to(root) for root in roots):
        raise ValueError("Synthetic output must be a child of repository temp/ or frontend/temp/")
    return path


def verify_job(path: Path) -> dict:
    path = isolated_path(path)
    marker = json.loads((path / MARKER).read_text(encoding="utf-8"))
    if (
        marker.get("dataset_id") != DATASET_ID
        or marker.get("model_id") != MODEL_ID
        or marker.get("synthetic") is not True
        or marker.get("collection") != COLLECTION
    ):
        raise ValueError("Expected an explicitly synthetic development job")
    for child in path.rglob("*"):
        if child.is_symlink() or child.is_junction():
            raise ValueError("Synthetic jobs cannot contain redirected paths")
    from geospatial_search.bundles import validate_bundle
    from geospatial_search.download_nys_naip import sha256
    from geospatial_search.paths import contained

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((path / "completion.json").read_text(encoding="utf-8"))
    run = json.loads((path / COLLECTION / "run.json").read_text(encoding="utf-8"))
    catalog = json.loads((path / COLLECTION / "collection.json").read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("synthetic") is not True
        or manifest.get("tile_count") != 2
        or len(manifest.get("tiles", [])) != 2
        or completion.get("manifest_sha256") != sha256(path / "manifest.json")
        or run.get("synthetic") is not True
        or run.get("model_contract", {}).get("model_id") != MODEL_ID
        or run.get("run_fingerprint") != marker.get("run_fingerprint")
        or catalog.get("run_fingerprint") != marker.get("run_fingerprint")
    ):
        raise ValueError("Synthetic fixture provenance or model contract differs")
    if (
        catalog.get("synthetic") is not True
        or catalog.get("embedding_dim") != 16
        or catalog.get("chip_count") != sum(shard["chip_count"] for shard in catalog["shards"])
        or len(catalog["shards"]) != len(manifest["tiles"])
    ):
        raise ValueError("Synthetic collection catalog differs")
    for entry, shard in zip(manifest["tiles"], catalog["shards"]):
        name = entry["name"]
        if (
            name not in ("synthetic-0.tif", "synthetic-1.tif")
            or shard["shard_id"] != Path(name).stem
        ):
            raise ValueError("Unknown synthetic source identity")
        record = json.loads((path / "records" / f"{name}.json").read_text(encoding="utf-8"))
        expected_uri = f"utm18/{name}"
        if record.get("synthetic") is not True or record.get("output") != f"imagery/{expected_uri}":
            raise ValueError("Synthetic source provenance differs")
        source = contained(path, path / record["output"])
        if (
            sha256(source) != record["output_sha256"]
            or source.stat().st_size != record["output_bytes"]
        ):
            raise ValueError("Synthetic source checksum differs")
        folder = contained(path, path / COLLECTION / "shards" / shard["shard_id"])
        bundle = validate_bundle(folder, marker["run_fingerprint"], verify_hashes=True)
        if (
            bundle.get("synthetic") is not True
            or bundle.get("model_contract") != run["model_contract"]
            or bundle.get("source_uri") != expected_uri
            or bundle.get("source_sha256") != record["output_sha256"]
            or bundle.get("chip_count") != shard["chip_count"]
        ):
            raise ValueError("Synthetic bundle identity or model contract differs")
    return marker


def create_job(output: Path, seed: int = 2022) -> dict:
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    from geospatial_search.bundles import publish_bundle
    from geospatial_search.download_nys_naip import Budget, sha256
    from geospatial_search.tiling import Footprints, Grid

    output = isolated_path(output)
    if output.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing directory; choose a new synthetic output"
        )
    output.mkdir(parents=True)
    budget = Budget(output, cap=20_000_000, reserve=1_000_000)
    grid = Grid()
    contract = {
        "model_id": MODEL_ID,
        "model_revision": "1",
        "arch": "synthetic",
        "model_fingerprint": MODEL_ID,
        "preprocess_fingerprint": "none-synthetic",
        "embedding_dim": 16,
        "embedding_dtype": "float16",
        "amp": False,
        "synthetic": True,
        "metric": "cosine",
        "normalization": "float32-l2",
        "rgb_band_order": [1, 2, 3],
    }
    recipe = {
        "dataset_id": DATASET_ID,
        "seed": seed,
        "model": contract,
        "chip_size": grid.size,
        "stride": grid.stride,
        "min_valid_fraction": grid.min_valid_fraction,
        "generator_version": 1,
    }
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    root_run = {
        "run_fingerprint": fingerprint,
        "recipe": recipe,
        "model_contract": contract,
        "embedding_dim": 16,
        "embedding_dtype": "float16",
        "format_version": 1,
        "synthetic": True,
    }
    rng = np.random.default_rng(seed)
    tiles, shards = [], []
    y, x = np.indices((448, 448))
    for source_index in range(2):
        name = f"synthetic-{source_index}.tif"
        relative = f"imagery/utm18/{name}"
        transform = from_origin(600_000 + source_index * 448 * 0.6, 4_700_000, 0.6, 0.6)
        pixels = np.stack(
            (
                (x // 8 * 17 + source_index * 40) % 256,
                (y // 8 * 13 + source_index * 30) % 256,
                ((x + y) // 16 * 23) % 256,
            )
        ).astype("uint8")
        mask = np.full((448, 448), 255, dtype="uint8")
        if source_index == 0:
            mask[:80, :80] = 0
        with rasterio.Env(GDAL_PAM_ENABLED=False, GDAL_TIFF_INTERNAL_MASK=True):
            with MemoryFile() as memory:
                with memory.open(
                    driver="GTiff",
                    width=448,
                    height=448,
                    count=3,
                    dtype="uint8",
                    crs="EPSG:26918",
                    transform=transform,
                    compress="deflate",
                    tiled=True,
                ) as dataset:
                    dataset.write(pixels)
                    dataset.write_mask(mask)
                budget.write_bytes(output / relative, memory.read())
        digest = sha256(output / relative)
        bounds = transform_bounds(
            "EPSG:26918", "EPSG:4326", *rasterio.transform.array_bounds(448, 448, transform)
        )
        entry = {
            "name": name,
            "zone": 18,
            "url": f"synthetic://{name}",
            "crs": "EPSG:26918",
            "width": 448,
            "height": 448,
            "transform": list(transform)[:6],
        }
        tiles.append(entry)
        record = {
            **entry,
            "output": relative,
            "source_url": entry["url"],
            "output_bytes": (output / relative).stat().st_size,
            "output_sha256": digest,
            "bounds_wgs84": list(bounds),
            "jpeg_quality": None,
            "synthetic": True,
            "acquisition_date": None,
            "release_year": None,
        }
        budget.atomic_json(output / "records" / f"{name}.json", record)
        rows = []
        with rasterio.open(output / relative) as dataset:
            footprints = Footprints(dataset, grid.size)
            for window in grid.windows(dataset.width, dataset.height):
                valid = float(
                    np.count_nonzero(dataset.dataset_mask(window=window)) / (grid.size**2)
                )
                if valid < grid.min_valid_fraction:
                    continue
                rows.append(
                    {
                        **footprints.metadata(window),
                        "embedding_id": len(rows),
                        "dataset_id": DATASET_ID,
                        "chip_id": f"{DATASET_ID}/{name}/{int(window.col_off)}-{int(window.row_off)}",
                        "source_uri": f"utm18/{name}",
                        "source_tile_id": Path(name).stem,
                        "source_sha256": digest,
                        "pixel_x": int(window.col_off),
                        "pixel_y": int(window.row_off),
                        "pixel_width": grid.size,
                        "pixel_height": grid.size,
                        "valid_fraction": valid,
                        "scale_id": grid.scale_id,
                        "rgb_band_order": "1,2,3",
                        "model_id": MODEL_ID,
                        "model_fingerprint": MODEL_ID,
                        "preprocess_fingerprint": "none-synthetic",
                        "run_fingerprint": fingerprint,
                    }
                )
        vectors = rng.standard_normal((len(rows), 16), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        run = {**root_run, "source_uri": f"utm18/{name}", "source_sha256": digest}
        publish_bundle(budget, output / COLLECTION / "shards" / Path(name).stem, vectors, rows, run)
        shards.append(
            {"shard_id": Path(name).stem, "source_uri": run["source_uri"], "chip_count": len(rows)}
        )
    manifest = {
        "dataset_id": DATASET_ID,
        "synthetic": True,
        "tile_count": len(tiles),
        "bands": [1, 2, 3],
        "resolution_m_per_pixel": 0.6,
        "tiles": tiles,
    }
    budget.atomic_json(output / "manifest.json", manifest)
    budget.atomic_json(
        output / "completion.json",
        {
            "synthetic": True,
            "tile_count": len(tiles),
            "manifest_sha256": sha256(output / "manifest.json"),
        },
    )
    budget.atomic_json(output / COLLECTION / "run.json", root_run)
    budget.atomic_json(
        output / COLLECTION / "collection.json",
        {
            "state": "complete",
            "synthetic": True,
            "format_version": 1,
            "embedding_dim": 16,
            "run_fingerprint": fingerprint,
            "total_sources": len(tiles),
            "shards": shards,
            "chip_count": sum(shard["chip_count"] for shard in shards),
        },
    )
    marker = {
        "dataset_id": DATASET_ID,
        "model_id": MODEL_ID,
        "synthetic": True,
        "seed": seed,
        "collection": COLLECTION,
        "run_fingerprint": fingerprint,
        "warning": "Generated patterns and seeded vectors. No semantic model or real aerial imagery.",
    }
    # Marker is published last so incomplete fixture writes cannot become a dev job.
    budget.atomic_json(output / MARKER, marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "temp/synthetic-job")
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    marker = create_job(args.output, args.seed)
    print(json.dumps({"job": str(args.output.resolve()), **marker}, indent=2))


if __name__ == "__main__":
    main()
