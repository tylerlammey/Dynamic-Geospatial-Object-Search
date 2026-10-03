"""Search GeoRSCLIP image embeddings with text and render the top PNG results."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from geospatial_search.embed_georsclip import load_model, to_uint8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Natural-language search text")
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("data/embeddings/albany_sp24/georsclip-vith14-s512-st256/embeddings.f16.npy"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/embeddings/albany_sp24/georsclip-vith14-s512-st256/chips.geoparquet"),
    )
    parser.add_argument("--imagery-dir", type=Path, default=Path("imagery/city_Albany_sp24"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/search-results"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-id", default="BiliSakura/GeoRSCLIP-ViT-H-14")
    parser.add_argument("--model-filename", default="GeoRSCLIP-ViT-H-14.pt")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--arch", default="ViT-H-14")
    parser.add_argument("--base-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def encode_text(args: argparse.Namespace, device: torch.device) -> np.ndarray:
    model, _, _, _ = load_model(args, device)
    tokenizer = __import__("open_clip").get_tokenizer(args.arch)
    tokens = tokenizer([args.query]).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=not args.disable_amp
    ):
        feature = model.encode_text(tokens)
    return F.normalize(feature.float(), dim=-1).cpu().numpy()[0]


def top_matches(vectors: np.ndarray, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    if top_k <= 0 or top_k > len(vectors):
        raise ValueError(f"top-k must be between 1 and {len(vectors)}")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError(f"Vector dimension {vectors.shape[1]} does not match query dimension {query.shape[0]}")
    scores = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), 4096):
        end = min(start + 4096, len(vectors))
        scores[start:end] = vectors[start:end].astype(np.float32) @ query
    indices = np.argpartition(scores, -top_k)[-top_k:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return [(int(index), float(scores[index])) for index in indices]


def render_chip(row: dict, imagery_dir: Path, rank: int, score: float, output_path: Path) -> None:
    source = imagery_dir / row["source_uri"]
    with rasterio.open(source) as dataset:
        pixels = dataset.read(
            indexes=(1, 2, 3),
            window=rasterio.windows.Window(
                row["pixel_x"], row["pixel_y"], row["pixel_width"], row["pixel_height"]
            ),
        )
    image = Image.fromarray(to_uint8(pixels), mode="RGB")
    labelled = Image.new("RGB", (image.width, image.height + 38), "black")
    labelled.paste(image, (0, 38))
    ImageDraw.Draw(labelled).text(
        (8, 10), f"#{rank}  score={score:.4f}  {row['chip_id']}", fill="white"
    )
    labelled.save(output_path)


def write_contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumbnails = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((320, 320))
        thumbnails.append(image.copy())
    columns = 5
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, rows * 320), "black")
    for index, image in enumerate(thumbnails):
        x = (index % columns) * 320
        y = (index // columns) * 320
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm CUDA device is available for GeoRSCLIP text encoding")
    if not args.embedding_path.is_file() or not args.metadata_path.is_file():
        raise FileNotFoundError("Embedding output is missing; run embed_georsclip first")
    device = torch.device("cuda:0")
    print(f"Using {torch.cuda.get_device_name(device)}")
    print(f"Encoding query: {args.query!r}")
    query = encode_text(args, device)
    vectors = np.load(args.embedding_path, mmap_mode="r")
    print(f"Searching {len(vectors):,} image embeddings...")
    matches = top_matches(vectors, query, args.top_k)

    columns = ["embedding_id", "chip_id", "source_uri", "pixel_x", "pixel_y", "pixel_width", "pixel_height", "centroid_lon", "centroid_lat"]
    rows_by_id = {int(row["embedding_id"]): row for row in pq.read_table(args.metadata_path, columns=columns).to_pylist()}
    safe_query = re.sub(r"[^a-z0-9]+", "-", args.query.lower()).strip("-")[:48] or "query"
    run_dir = args.output_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{safe_query}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pngs: list[Path] = []
    result_rows: list[dict] = []
    for rank, (embedding_id, score) in enumerate(matches, start=1):
        row = rows_by_id[embedding_id]
        png_path = run_dir / f"rank-{rank:02d}_score-{score:.4f}.png"
        render_chip(row, args.imagery_dir, rank, score, png_path)
        pngs.append(png_path)
        result_rows.append({"rank": rank, "score": score, **row, "png": png_path.name})
        print(f"#{rank:02d} {score:.4f} {row['chip_id']}")
    write_contact_sheet(pngs, run_dir / "contact-sheet.png")
    (run_dir / "results.json").write_text(json.dumps({"query": args.query, "results": result_rows}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(pngs)} PNGs and contact-sheet.png to {run_dir}")


if __name__ == "__main__":
    main()
