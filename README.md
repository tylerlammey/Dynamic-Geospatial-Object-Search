# Dynamic Geospatial Object Search

Search georeferenced aerial imagery with natural-language queries. This project
creates a GeoRSCLIP embedding for each image chip in a source orthophoto
collection, then returns and renders the chips most similar to a text query.

The current implementation targets Albany, New York, 2024 four-band JPEG 2000
orthophotography on Windows with an AMD Radeon RX 9070 XT and ROCm-enabled
PyTorch.

## How it works

1. Read each source JP2 without modifying it.
2. Extract full-size, 512 x 512 pixel chips at a 256-pixel stride.
3. Use source bands 1, 2, and 3 as RGB; the fourth NIR band is retained in the
   source imagery but is not supplied to the three-channel model.
4. Encode each chip with GeoRSCLIP and store an L2-normalized float16 vector.
5. Store chip spatial metadata and source-pixel windows in GeoParquet.
6. Encode a text query with the matching text encoder, rank by dot product
   (cosine similarity), and render the highest-ranked source windows.

## Outputs

An embedding run writes one compatible bundle:

| File | Description |
| --- | --- |
| `embeddings.f16.npy` | Row-major, normalized float16 embedding matrix. |
| `chips.geoparquet` | Chip IDs, source windows, geographic footprints, and compatibility metadata. |
| `run.json` | Dataset, grid, checkpoint hash, preprocessing fingerprint, and vector contract. |

`embedding_id` is the row number in `embeddings.f16.npy` and links each vector
to the equivalent GeoParquet row.

## Prerequisites

- Python 3.12
- Windows AMD ROCm runtime and a compatible Radeon GPU for embedding and text
  encoding
- Source JP2 imagery available locally

Install ROCm-enabled PyTorch separately, then install this project's Python
dependencies. The ordinary PyPI PyTorch package is CPU-only on Windows and is
not sufficient for this workflow. See the detailed
[RX 9070 XT setup guide](docs/rx-9070xt-embedding-setup.md).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install the matching ROCm PyTorch wheel first; see the setup guide.
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Run an embedding job

First run a smoke test against a directory containing one JP2 tile and a new
output directory. For the full Albany collection:

```powershell
python -m geospatial_search.embed_georsclip `
  --imagery-dir imagery/city_Albany_sp24 `
  --output-dir data/embeddings/albany_sp24/georsclip-vith14-s512-st256 `
  --dataset-id albany_sp24 `
  --batch-size 32
```

If the GPU runs out of memory, retry to a **new** output directory using batch
size 16, then 8. The program refuses to write over an existing embedding bundle.

## Search with text

```powershell
python -m geospatial_search.search_text --query "baseball field"
```

By default, results are written to a new timestamped directory under
`data/search-results/` containing labelled ranking PNGs, a contact sheet, and
`results.json`.

## Download New York State imagery

A separate resumable downloader acquires all 3,805 New York 2022 NAIP tiles as
RGB GeoTIFFs at native 0.6 m/pixel, with four workers, progress/ETA, provenance,
and a 300 GB folder cap. It keeps every generated file under ignored `data/`:

```powershell
.\scripts\download-nys-naip.ps1
```

See the [New York State download guide](docs/nys-naip-download.md) for dependencies,
the pilot, background operation, pause/resume, storage estimates, and tests.
The current Albany embedder needs TIFF and statewide scaling changes before
using this collection for embeddings.

## Repository policy

Source imagery, downloaded model files, local caches, embeddings, GeoParquet
outputs, and rendered search results are intentionally ignored by Git. Commit
only code, documentation, reproducibility configuration, and legally shareable
small fixtures. Respect the imagery provider's license and coordinate-sharing
restrictions before distributing imagery or results.

## Project status and documentation

This is an initial exact-search prototype. The architecture, data contract,
evaluation plan, and future ANN-index options are described in the
[design document](docs/georsclip-embedding-search-design.md). Operational
installation and run instructions are in the
[setup guide](docs/rx-9070xt-embedding-setup.md).

If you use embeddings produced by this project, cite the GeoRSCLIP work listed
in [CITATION.cff](CITATION.cff).

## License

This repository is released under the [MIT License](LICENSE). Model and imagery
licenses apply separately.
