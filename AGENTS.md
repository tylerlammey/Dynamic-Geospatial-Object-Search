# Dynamic Geospatial Object Search: Agent Guide

## Scope

The frozen New York State 2022 NAIP base and its separately labeled 2021-release
gap-fill supplement are active. Read README.md, docs/nys-pipeline.md,
and docs/nys-naip-download.md. Keep the completed source imagery immutable at
data/nys-naip-2022-rgb-060m/imagery/utm17, utm18, and utm19. Preserve downloader
manifest, records, and provider metadata; do not relocate these paths.
Read docs/nys-gapfill.md when working with supplements. Keep their imagery,
records, recipes, and vectors separate under the primary job's supplements/;
share the primary 300 GB budget and GPU lock. Preserve acquisition dates and
release years. Valid base pixels take display precedence, and base-only display
and search must remain available. Globally qualify rows by
(dataset_id, shard_id, embedding_id).

## Code and configuration

- configs/nys-georsclip.json: shared default grid and pinned GeoRSCLIP checkpoint.
- src/geospatial_search/download_nys_naip.py: acquisition and shared folder budget.
- tiling.py: lazy pixel windows, source inventory, RGB extraction, WGS84 footprints.
- model.py: pinned model loading and float32 normalization.
- bundles.py: atomic NumPy/GeoParquet/run bundles and bounded exact ranking.
- embed_georsclip.py: bounded threaded readers, GPU inference, durable resume.
- embedding_control.py: graceful pause, collection ownership, shared GPU lock.
- search_text.py: compatible text encoding, collection search, result rendering.
- search_index.py: CPU IVF-PQ candidates, original-vector rescoring, seen-ID exclusion.
- build_search_index.py: budgeted, checksummed, snapshot-bound atomic index publication.
- benchmark_search_index.py: real-query comparisons with the unchanged exact baseline.
- preview_tiles.py: CPU previews of the selected chip size.
- scripts/: repository-anchored PowerShell setup/download/embed/search launchers.
- tests/: synthetic offline regressions.
- temp/: ignored temporary scripts and plans; avoid ad-hoc files in tracked folders.

## Data contract

Use RGB bands 1, 2, 3 from three-band uint8 GeoTIFFs at native 0.6 m/pixel.
The selected chip is 256 x 256 pixels at stride 192: 25% overlap. Final windows
remain full size, anchored to source edges. Preserve masks; exclude chips with
less than 95% valid pixels. Do not emit individual chip images during embedding.

GeoParquet geometry and display coordinates use EPSG:4326 with longitude first.
Keep original NAD83 UTM CRS, source windows, and affine transforms separately.
Do not reproject source imagery just to unify metadata coordinates.

Normalize embeddings in float32, then store float16. Dot products implement
cosine ranking. Each source TIFF has its own shard: embedding_id equals the
vector row number within that shard, and (shard_id, embedding_id) identifies a
collection row. Never mix embeddings.f16.npy, chips.geoparquet, and run.json
across shards or compatibility fingerprints. Publish the three files together.
The optional search index is derived from one published snapshot and lives in
that collection's search-index/. Preserve original vectors. ANN candidate
selection is approximate; returned scores use the original float16 vectors
converted to float32. Keep exact search available and measure recall before
changing indexed defaults. Additional-match requests use qualified seen IDs.

Pin BiliSakura/GeoRSCLIP-ViT-H-14 to the config's full commit SHA. Build OpenCLIP
ViT-H-14 with laion2b_s32b_b79k preprocessing, then load the exact checkpoint
strictly. Preserve checkpoint, preprocessing, source, grid, code, and dependency
fingerprints. Batch size and reader count may change on resume; incompatible
changes require a distinct output directory.

## Runtime and safety

Windows AMD RX 9070 XT uses ROCm-enabled PyTorch, exposed through torch.cuda.
Install GPU PyTorch separately; see docs/rx-9070xt-embedding-setup.md. Never
silently fall back to CPU. Do not assume CUDA-only FAISS/cuVS support.
Disable pinned transfers on Windows ROCm: its pinned allocator can hang at
shutdown. CUDA builds can use pinned buffers.

Every generated pipeline artifact belongs under the existing NYS job folder.
Use the shared 300 GB Budget guard, including temporary writes and PNG results.
The persistent Hugging Face cache may remain outside that folder; if in the
repository, use an ignored model-cache/ or hf-cache/ path. Keep Windows awake
during long embedding runs. The launcher runs in the background and can attach
to existing progress. PAUSE finishes and saves the active source tile before
releasing GPU memory and OS locks; STOP/Ctrl+C may abandon the current tile.
Preserve complete bundles, wait for pause acknowledgement before search, and
hold collection plus job-wide GPU locks for both embedding and query inference.

Before changing window planning, masks, RGB, vector alignment, normalization,
or ranking, add meaningful tests. Run:
    $env:PYTHONPATH = 'src'
    .\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
Before a full GPU run, use scripts/embed-nys.ps1 -SmokeTest and verify text search.
Do not launch the full embedding job merely to prepare or review the pipeline.

## Git policy

Commit code, docs, configuration, dependencies, and legally shareable synthetic
tests. Keep imagery, data, output, checkpoints, caches, virtual environments,
credentials, and temp files out of Git. Preserve useful local provenance.
Check imagery licensing and coordinate-sharing conditions before publishing.
