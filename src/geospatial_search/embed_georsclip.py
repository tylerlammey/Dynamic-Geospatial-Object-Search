"""Create GeoRSCLIP embeddings for georeferenced RGB chips from JP2 imagery.

The source rasters are never modified. This program emits a GeoParquet manifest
and a normalized float16 NumPy matrix whose row number is `embedding_id`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import open_clip
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from pyproj import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds
from shapely import to_wkb
from shapely.geometry import box
from tqdm import tqdm
from huggingface_hub import hf_hub_download

LOG = logging.getLogger("geospatial_search.embed")


@dataclass(frozen=True)
class Chip:
    source_path: Path
    source_uri: str
    source_tile_id: str
    x: int
    y: int
    width: int
    height: int
    stride: int
    native_bounds: tuple[float, float, float, float]
    wgs84_bounds: tuple[float, float, float, float]
    crs_wkt: str

    @property
    def chip_id(self) -> str:
        return (
            f"{self.source_tile_id}/s{self.width}_st{self.stride}"
            f"/x{self.x:07d}_y{self.y:07d}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def axis_starts(length: int, chip_size: int, stride: int) -> list[int]:
    """Return starts which cover the final source pixel without padded chips."""
    if length < chip_size:
        return []
    starts = list(range(0, length - chip_size + 1, stride))
    final_start = length - chip_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def plan_chips(
    imagery_dir: Path, chip_size: int, stride: int
) -> tuple[list[Chip], str]:
    """Plan full-size chips and footprints before allocating vector storage."""
    if chip_size <= 0 or stride <= 0:
        raise ValueError("chip_size and stride must be positive")
    paths = sorted(path for path in imagery_dir.rglob("*") if path.suffix.lower() == ".jp2")
    if not paths:
        raise FileNotFoundError(f"No .jp2 files found under {imagery_dir}")

    chips: list[Chip] = []
    collection_crs: str | None = None
    for source_path in tqdm(paths, desc="Planning chips", unit="tile"):
        with rasterio.open(source_path) as dataset:
            if dataset.count < 3:
                raise ValueError(f"{source_path} has {dataset.count} bands; RGB requires 3")
            if dataset.crs is None:
                raise ValueError(f"{source_path} has no coordinate reference system")
            crs_wkt = dataset.crs.to_wkt()
            if collection_crs is None:
                collection_crs = crs_wkt
            elif collection_crs != crs_wkt:
                raise ValueError("All inputs in one run must use the same CRS")

            xs = axis_starts(dataset.width, chip_size, stride)
            ys = axis_starts(dataset.height, chip_size, stride)
            if not xs or not ys:
                LOG.warning("Skipping %s because it is smaller than one chip", source_path)
                continue
            source_uri = source_path.relative_to(imagery_dir).as_posix()
            for y in ys:
                for x in xs:
                    window = Window(x, y, chip_size, chip_size)
                    native = tuple(float(v) for v in window_bounds(window, dataset.transform))
                    wgs84 = tuple(
                        float(v)
                        for v in transform_bounds(
                            dataset.crs, "EPSG:4326", *native, densify_pts=21
                        )
                    )
                    chips.append(
                        Chip(
                            source_path=source_path,
                            source_uri=source_uri,
                            source_tile_id=source_path.stem,
                            x=x,
                            y=y,
                            width=chip_size,
                            height=chip_size,
                            stride=stride,
                            native_bounds=native,
                            wgs84_bounds=wgs84,
                            crs_wkt=crs_wkt,
                        )
                    )
    if not chips or collection_crs is None:
        raise RuntimeError("No full-size chips were planned")
    return chips, collection_crs


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert band-first pixels to HWC uint8 without per-chip contrast changes."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected a 3xHxW RGB array, received {rgb.shape}")
    if rgb.dtype == np.uint8:
        converted = rgb
    elif np.issubdtype(rgb.dtype, np.integer):
        info = np.iinfo(rgb.dtype)
        converted = np.clip(rgb.astype(np.float32), 0, info.max) * (255.0 / info.max)
        converted = converted.astype(np.uint8)
    else:
        max_value = 1.0 if float(np.nanmax(rgb)) <= 1.0 else 255.0
        converted = np.clip(rgb, 0, max_value) * (255.0 / max_value)
        converted = converted.astype(np.uint8)
    return np.moveaxis(converted, 0, -1)


def chip_batches(chips: list[Chip], batch_size: int) -> Iterator[list[Chip]]:
    for start in range(0, len(chips), batch_size):
        yield chips[start : start + batch_size]


def read_chip(dataset: rasterio.io.DatasetReader, chip: Chip) -> Image.Image:
    window = Window(chip.x, chip.y, chip.width, chip.height)
    # Provider documentation defines bands 1/2/3 as natural-color RGB.
    rgb = dataset.read(indexes=(1, 2, 3), window=window, boundless=False)
    return Image.fromarray(to_uint8(rgb), mode="RGB")


def load_model(args: argparse.Namespace, device: torch.device):
    """Download the repository's OpenCLIP checkpoint once and load it on ROCm."""
    cache_dir = str(args.model_cache_dir) if args.model_cache_dir else None
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=args.model_id,
            filename=args.model_filename,
            cache_dir=cache_dir,
        )
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.arch, pretrained=args.base_pretrained
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        LOG.warning("Checkpoint has %d unused keys", len(unexpected))
    if missing:
        LOG.warning("Model has %d keys not supplied by checkpoint", len(missing))
    return model.to(device).eval(), preprocess, checkpoint_path, sha256_file(checkpoint_path)


def encode_batch(
    model: torch.nn.Module, preprocess, images: list[Image.Image], device: torch.device, amp: bool
) -> np.ndarray:
    inputs = torch.stack([preprocess(image) for image in images]).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp
    ):
        features = model.encode_image(inputs)
    return F.normalize(features.float(), dim=-1).cpu().numpy()


def manifest_rows(
    chips: list[Chip], dataset_id: str, model_id: str, model_fingerprint: str,
    preprocess_fingerprint: str, scale_id: str
) -> list[dict]:
    rows: list[dict] = []
    for embedding_id, chip in enumerate(chips):
        minx, miny, maxx, maxy = chip.native_bounds
        west, south, east, north = chip.wgs84_bounds
        rows.append(
            {
                "embedding_id": np.uint64(embedding_id),
                "dataset_id": dataset_id,
                "chip_id": chip.chip_id,
                "source_uri": chip.source_uri,
                "source_tile_id": chip.source_tile_id,
                "pixel_x": chip.x,
                "pixel_y": chip.y,
                "pixel_width": chip.width,
                "pixel_height": chip.height,
                "native_minx": minx,
                "native_miny": miny,
                "native_maxx": maxx,
                "native_maxy": maxy,
                "west": west,
                "south": south,
                "east": east,
                "north": north,
                "centroid_lon": (west + east) / 2,
                "centroid_lat": (south + north) / 2,
                "crs_wkt": chip.crs_wkt,
                "scale_id": scale_id,
                "rgb_band_order": "1,2,3",
                "model_id": model_id,
                "model_fingerprint": model_fingerprint,
                "preprocess_fingerprint": preprocess_fingerprint,
                "geometry": to_wkb(box(minx, miny, maxx, maxy), output_dimension=2),
            }
        )
    return rows


def write_geoparquet(rows: list[dict], path: Path, crs_wkt: str) -> None:
    """Write a standards-compliant WKB GeoParquet file without Pandas/GeoPandas."""
    minx = min(row["native_minx"] for row in rows)
    miny = min(row["native_miny"] for row in rows)
    maxx = max(row["native_maxx"] for row in rows)
    maxy = max(row["native_maxy"] for row in rows)
    geo_metadata = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon"],
                "crs": CRS.from_wkt(crs_wkt).to_json_dict(),
                "bbox": [minx, miny, maxx, maxy],
            }
        },
    }
    table = pa.Table.from_pylist(rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"geo"] = json.dumps(geo_metadata, separators=(",", ":")).encode("utf-8")
    pq.write_table(
        table.replace_schema_metadata(metadata), path, compression="zstd"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="BiliSakura/GeoRSCLIP-ViT-H-14")
    parser.add_argument("--model-filename", default="GeoRSCLIP-ViT-H-14.pt")
    parser.add_argument(
        "--model-cache-dir", type=Path,
        help="Optional persistent Hugging Face cache directory; omit to use the normal user cache",
    )
    parser.add_argument("--dataset-id", default="albany_sp24")
    parser.add_argument("--chip-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--arch", default="ViT-H-14")
    parser.add_argument("--base-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm CUDA device is visible. Install AMD ROCm Windows PyTorch wheels, "
            "then verify torch.cuda.is_available() is True."
        )
    device = torch.device("cuda:0")
    LOG.info("Using %s", torch.cuda.get_device_name(device))
    chips, collection_crs = plan_chips(args.imagery_dir, args.chip_size, args.stride)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.output_dir / "embeddings.f16.npy"
    metadata_path = args.output_dir / "chips.geoparquet"
    run_path = args.output_dir / "run.json"
    if any(path.exists() for path in (vector_path, metadata_path, run_path)):
        raise FileExistsError(f"Output exists in {args.output_dir}; choose a new output directory")

    model, preprocess, checkpoint_path, model_fingerprint = load_model(args, device)
    preprocess_fingerprint = hashlib.sha256(repr(preprocess).encode("utf-8")).hexdigest()
    vector_store: np.memmap | None = None
    written = 0
    by_source: dict[Path, list[Chip]] = {}
    for chip in chips:
        by_source.setdefault(chip.source_path, []).append(chip)
    progress = tqdm(total=len(chips), desc="Embedding chips", unit="chip")
    for source_path, source_chips in by_source.items():
        with rasterio.open(source_path) as dataset:
            for batch in chip_batches(source_chips, args.batch_size):
                features = encode_batch(
                    model, preprocess, [read_chip(dataset, chip) for chip in batch],
                    device, amp=not args.disable_amp,
                )
                if vector_store is None:
                    vector_store = np.lib.format.open_memmap(
                        vector_path, mode="w+", dtype=np.float16,
                        shape=(len(chips), features.shape[1]),
                    )
                vector_store[written : written + len(batch)] = features.astype(np.float16)
                written += len(batch)
                progress.update(len(batch))
    progress.close()
    if vector_store is None or written != len(chips):
        raise RuntimeError(f"Wrote {written} embeddings for {len(chips)} planned chips")
    vector_store.flush()

    model_id = args.model_id
    scale_id = f"s{args.chip_size}_st{args.stride}"
    write_geoparquet(
        manifest_rows(
            chips, args.dataset_id, model_id, model_fingerprint,
            preprocess_fingerprint, scale_id,
        ),
        metadata_path,
        collection_crs,
    )
    run_path.write_text(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "created_at": datetime.now(UTC).isoformat(),
                "chip_count": len(chips),
                "chip_size": args.chip_size,
                "stride": args.stride,
                "model_id": model_id,
                "model_filename": args.model_filename,
                "model_cache_path": str(checkpoint_path),
                "model_fingerprint": model_fingerprint,
                "preprocess_fingerprint": preprocess_fingerprint,
                "model_cache_dir": str(args.model_cache_dir) if args.model_cache_dir else None,
                "embedding_dim": int(vector_store.shape[1]),
                "embedding_dtype": "float16",
                "metric": "cosine (L2-normalized inner product)",
                "rgb_band_order": [1, 2, 3],
                "source_crs_wkt": collection_crs,
            }, indent=2,
        ) + "\n", encoding="utf-8",
    )
    LOG.info("Wrote %d vectors of dimension %d to %s", len(chips), vector_store.shape[1], vector_path)
    LOG.info("Wrote GeoParquet chip manifest to %s", metadata_path)


if __name__ == "__main__":
    main()
