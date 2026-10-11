# Dynamic Geospatial Object Search

Semantic text search over New York State's 2022 NAIP aerial imagery. Sources
are native 0.6 m/pixel RGB GeoTIFFs. GeoRSCLIP encodes 256 x 256 pixel chips
(153.6 x 153.6 meters), using a 192-pixel stride for **25% overlap**.

## Run

Use Python 3.12. Install the appropriate GPU PyTorch build first; this machine
uses an AMD RX 9070 XT with ROCm. See the
[GPU setup guide](docs/rx-9070xt-embedding-setup.md).

```powershell
.\scripts\setup-nys.ps1 -Embeddings
.\scripts\embed-nys.ps1 -Plan
.\scripts\embed-nys.ps1 -SmokeTest -Offline
# Full statewide run, after checking the smoke test:
.\scripts\embed-nys.ps1 -Offline
# Reattach to background progress:
.\scripts\embed-nys.ps1 -Monitor
# Safely pause and test whichever embeddings are saved:
.\scripts\embed-nys.ps1 -Pause
.\scripts\search-nys.ps1 -Query 'baseball field' -Offline
.\scripts\embed-nys.ps1 -Unpause -Offline
```

The download on this machine is complete. On a new computer, install downloader
dependencies with `scripts\setup-nys.ps1`, then run
`scripts\download-nys-naip.ps1`. See the
[download guide](docs/nys-naip-download.md) and the
[tiling, embedding, and search guide](docs/nys-pipeline.md) for progress,
pause/resume, directory structure, storage, and verification.
The embedding launcher starts a background worker or attaches to the existing
one. Closing the progress monitor leaves embedding running; use `-Pause` to
save the current source tile and release the GPU before searching.

## Layout

```text
configs/nys-georsclip.json       # Shared grid and pinned model settings
scripts/                        # Setup, download, embed, search launchers
src/geospatial_search/          # Native-window readers, GPU model, atomic bundles
tests/                          # Synthetic regression tests; no real imagery
docs/                           # NYS operating guides
data/nys-naip-2022-rgb-060m/      # Ignored imagery, provenance, embeddings, results
temp/                           # Ignored temporary AI scripts/plans
```

Chips are read directly from TIFF windows rather than saved as millions of
individual image files. Each source TIFF gets one compatible bundle containing
`embeddings.f16.npy`, `chips.geoparquet`, and `run.json`. Vectors are normalized
in float32 and stored as float16; search uses exact cosine ranking across
memory-mapped bundles. All geographic footprints use WGS84, while pixel windows
retain their original raster CRS and transform.

Imagery, checkpoints, embeddings, local environments, and generated previews
stay out of Git. Code is [MIT licensed](LICENSE); imagery and model licenses
apply separately. Cite GeoRSCLIP using [CITATION.cff](CITATION.cff).
