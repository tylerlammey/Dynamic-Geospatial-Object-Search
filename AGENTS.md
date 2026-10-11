# Dynamic Geospatial Object Search: Agent Guide

## Purpose

This repository prototypes semantic text-to-aerial-imagery search. It turns
georeferenced Albany, NY, 2024 orthophotography into GeoRSCLIP image embeddings,
then retrieves and renders the image chips most similar to a natural-language
query.

The searchable unit is a georeferenced chip, not an entire source tile. The
repository deliberately keeps heavy source data and derived results out of Git.

## Repository layout

```text
src/geospatial_search/
  embed_georsclip.py  # chip planning, RGB extraction, embedding generation
  search_text.py      # text encoding, exact similarity search, PNG rendering
  download_nys_naip.py # resumable statewide 2022 NAIP RGB acquisition
docs/
  rx-9070xt-embedding-setup.md       # Windows + ROCm setup and runbook
  georsclip-embedding-search-design.md # architecture and longer-term design
  nys-naip-download.md             # statewide downloader operating guide
scripts/              # Windows download launcher and progress monitor
tests/                # synthetic offline downloader contract tests
imagery/              # local source JP2 rasters and sidecars; ignored
data/                 # local embeddings, manifests, and rendered results; ignored
requirements.txt      # Python dependencies other than ROCm PyTorch
CITATION.cff          # citation metadata
```

## Data and model contract

- Albany imagery is four-band JPEG 2000. Use bands **1, 2, 3** as RGB. Do not
  substitute NIR or pass all four bands to the three-channel GeoRSCLIP model.
- The baseline chip grid is 512 x 512 source pixels at a 256-pixel stride.
- Every embedding is L2-normalized in float32, then stored as float16.
  Dot-product ranking therefore represents cosine similarity.
- `embedding_id` is the row number of `embeddings.f16.npy`; it must continue to
  match the corresponding row in `chips.geoparquet`.
- `chips.geoparquet`, `embeddings.f16.npy`, and `run.json` are one atomic
  compatibility set. Never mix components from runs with different imagery,
  grid, model checkpoint, or preprocessing fingerprint.
- The source JP2s are immutable. Read windows with Rasterio; do not emit chip
  images except for debugging or result presentation.

## Runtime requirements

- Target platform: Windows with an AMD RX 9070 XT using ROCm-enabled PyTorch.
- Install the ROCm PyTorch wheel separately; plain PyPI `torch` is CPU-only on
  Windows. Follow `docs/rx-9070xt-embedding-setup.md`.
- Set `PYTHONPATH=src` when running directly from a checkout.
- Model: `BiliSakura/GeoRSCLIP-ViT-H-14`, loaded as OpenCLIP `ViT-H-14` using
  base pretrained tag `laion2b_s32b_b79k`, then overlaying the downloaded
  checkpoint.
- The Hugging Face checkpoint cache is local-only. If a cache directory is put
  in this repository, use an ignored path such as `hf-cache/` or `model-cache/`.

## Commands

PowerShell examples:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

# Generate an embedding bundle. Use a new output directory on every retry.
python -m geospatial_search.embed_georsclip `
  --imagery-dir imagery/city_Albany_sp24 `
  --output-dir data/embeddings/albany_sp24/georsclip-vith14-s512-st256 `
  --dataset-id albany_sp24 `
  --batch-size 32

# Search the default embedding bundle and render the top ten results.
python -m geospatial_search.search_text --query "baseball field"
```

Before a full run, smoke-test embedding on a directory containing one JP2 tile
and a distinct output directory. If GPU memory is exhausted, retry to a *new*
output path with batch size 16, then 8. AMP is enabled by default; only disable
it while diagnosing numeric or ROCm-kernel problems.

## Generated files and Git policy

`.gitignore` intentionally excludes:

- source imagery (`imagery/`), derived data (`data/`), ad-hoc output (`output/`)
- model checkpoints and in-repository model caches
- virtual environments, Python caches, coverage/test/lint caches
- environment files, local editor configuration, and OS metadata

Commit source code, documentation, dependency declarations, tests, small
fixtures that are legally shareable, and configuration needed to reproduce a
run. Do not commit real imagery, downloaded checkpoints, embedding arrays,
GeoParquet outputs, timestamped result PNGs, or API credentials. If a tracked
derived file must be removed later, adding an ignore rule alone will not untrack
it; use `git rm --cached <path>` only after confirming the target.

## Implementation notes

- `embed_georsclip.py` checks for an available `torch.cuda` device (the ROCm
  Windows wheel exposes the AMD GPU through this API) and intentionally refuses
  to run on CPU.
- Embedding generation exits if its target output already contains any expected
  result file. This prevents accidental mixing or partial overwrites.
- `search_text.py` runs exact batched CPU-side dot products over the memory-
  mapped vectors and creates a new timestamped folder under `data/search-results`.
- Result previews reopen the original JP2 and use each metadata row's pixel
  window. Metadata coordinates are meaningful only with their CRS.
- Downloader tests run with `PYTHONPATH=src` and
  `python -m unittest discover -s tests -v`; use synthetic imagery only. The
  embedding/search implementation has no automated test suite, build configuration,
  or package metadata. Add tests before changing chip planning, RGB conversion,
  vector/metadata alignment, or ranking behavior.
- The NYS downloader writes only under its selected job folder (default
  `data/nys-naip-2022-rgb-060m`) and shares the original data-local worker lock.
  Do not move/delete files or restart an active job while preparing source changes.
  Statewide TIFF/multiple-CRS embedding integration is still outstanding.

## Change guardrails

- Preserve reproducibility fields in `run.json` and the GeoParquet schema.
- Maintain normalized-vector cosine semantics unless the query and index paths
  are changed together.
- Keep model revision/preprocessing/grid changes as distinct embedding spaces;
  do not silently compare incompatible vectors.
- Avoid introducing CUDA-only FAISS or cuVS assumptions: this development
  machine uses AMD ROCm. Exact CPU search or CPU FAISS HNSW is appropriate for
  the current Albany scale.
- Check imagery licensing and coordinate-sharing restrictions before publishing
  datasets, screenshots, or result locations.

## References

Read the setup guide for installation and operational details, and the design
document for the intended data model, evaluation criteria, ANN roadmap, and
multi-city scaling plan.
