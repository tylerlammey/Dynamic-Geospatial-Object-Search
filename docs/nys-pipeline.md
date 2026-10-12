# NYS tiling, embeddings, and search

## Configuration and coverage

`configs/nys-georsclip.json` is the shared default configuration. The selected
chip is **256 x 256 native pixels**, covering **153.6 x 153.6 meters** at 0.6
m/pixel. The stride is **192 pixels** (115.2 meters): neighboring chips overlap
by **64 pixels / 38.4 meters / 25%** on both axes. Final windows are anchored
against each source raster edge; they remain full size and may overlap more.
No padding or source-image resampling is performed.

Overlap gives small objects near a chip border another view, but retrieval is
not an object detector and coverage does not guarantee detection. Each TIFF has
its own grid; buffered provider tiles may also overlap. Chips crossing TIFF
boundaries are not synthesized, and provider buffers are not deduplicated.
Windows with less than 95% valid pixels are excluded using the source mask;
valid objects in heavily masked edge windows may therefore be excluded.

OpenCLIP applies the checkpoint's deterministic RGB preprocessing to each chip,
resizing the square to a 224-pixel model input. GeoRSCLIP ViT-H/14 produces
1,024-dimensional vectors. The Hugging Face revision is pinned to
`09af41b95fbb7b4bf884904e0d3c2777e83740bc`; the checkpoint and preprocessing
are also fingerprinted. The existing Hugging Face cache is reused. `-Offline`
requires that checkpoint to be cached already; omit it on the first model run
on another computer. GPU PyTorch is required for both image and text encoding.

## Storage and runtime

The completed inventory contains 3,805 TIFFs totaling **114.91 GB**. This grid
has **11,985,388 candidate windows** before mask filtering, including source
buffers and final-edge overlaps. Their float16 vectors occupy at most
**24.55 GB**, plus NumPy headers, compressed GeoParquet, and run manifests.
Metadata size is measured as bundles are completed; it is not included in the
vector estimate. No full statewide chip-image collection is written.

Download, embedding publication, and search-result writes share the same
**300,000,000,000-byte folder cap** and reserve 5 GB. Writers check byte sizes
before publishing data, including temporary bundle files. This is software
accounting rather than an operating-system quota; arbitrary external writes
are not controlled. The model cache normally sits outside the job directory.

Four reader threads use separate GDAL handles and prefetch two batches each.
GDAL decoding, image transforms, and PyTorch already use native compiled code;
no separate C++ build is needed. Only one source's metadata/vector staging
array is accumulated for publication, with a 50,000-chip safety limit per TIFF.
RAM use stays bounded independently of statewide chip count.

The progress bar explicitly shows **percent complete**, processed/total windows,
**elapsed runtime (HH:MM:SS)**, ETA, vectors, and completed source tiles. Runtime
includes startup and accumulates active time across resumes; time while the job
is stopped is excluded. Skipped windows still count as processed. Percent is
based on candidate windows, while `durable_embedded_chips` counts published
vectors. ETA starts after measurable progress and uses recent completed sources
once available. The same fields are in `status.json` and `-Status`, alongside
reader-wait, GPU-inference/transfer, and publication timings.

Example display:

```text
NYS embeddings: 43.42% |...| 5204157/11985388, runtime 07:42:15 | ETA 10:05:30 | vectors 5,204,157 | tiles 1652/3805
```

The default batch size is now **128**, measured on the RX 9070 XT. See the
[performance verification](nys-embedding-performance.md). On a smaller GPU,
override with `-BatchSize 32` or 16. Use the full run's measured ETA after several
tiles for a current estimate; short benchmarks cannot guarantee statewide time.

## Commands

All launchers resolve paths relative to the checkout, including when called
from another working directory. Run these from the repository root:

```powershell
.\scripts\embed-nys.ps1 -Plan
.\scripts\embed-nys.ps1 -SmokeTest -Offline
.\scripts\embed-nys.ps1 -Offline
.\scripts\embed-nys.ps1 -Monitor
.\scripts\embed-nys.ps1 -Status
```

`-Plan` checks downloader provenance and saves a count estimate; it does not
load the GPU model. `-SmokeTest` embeds up to 128 candidate windows from one TIFF
in each UTM zone into a new timestamped smoke collection. It prints the exact
output directory. Test that collection's text search with:

```powershell
.\scripts\search-nys.ps1 -Query 'baseball field' -Offline `
  -CollectionDir 'data\nys-naip-2022-rgb-060m\embeddings\smoke-TIMESTAMP'
```

The full launcher starts a hidden background worker and opens its progress
monitor. Rerunning the same command attaches to the existing worker, or resumes
an idle unfinished collection. `-Monitor` only attaches; `-Status` prints one
snapshot. `-Background` starts or attaches and returns immediately. Closing the
monitor, including Ctrl+C, leaves the worker running. Keep Windows awake and
plugged in. Logs are in the collection's `runtime/` folder.

An OS lock prevents duplicate workers in the same collection, including when
two launchers race. A separate job-wide GPU lock prevents another NYS embedding
or search process from loading a competing model in a different collection.
Locks are released automatically if a process crashes; stale PID/status files
do not block resume. `-Foreground` remains available for debugging. Plan and
smoke commands normally run in the foreground; `-SmokeTest -Background` creates
a distinct smoke collection in the background.

To safely pause, search the available subset, and continue:

```powershell
.\scripts\embed-nys.ps1 -Pause
.\scripts\search-nys.ps1 -Query 'fighter jet' -Offline -TopK 10
.\scripts\embed-nys.ps1 -Unpause -Offline
```

Pause finishes and publishes the current source tile, cancels CPU prefetch,
and releases GPU memory before the pause command returns. It may take a tile's
remaining processing time. The monitor shows `pausing` and then `paused`.
All chips computed for that tile are saved. `-PauseTimeoutSeconds` defaults to
300; a timeout leaves the request pending and the worker running. Wait for
successful acknowledgement before searching. `-Unpause` is an alias for
`-Resume`. Search holds the collection lock until its model is released, so
resume cannot race a query.

Only complete source bundles are durable. A crash, Ctrl+C in foreground mode,
or a legacy `STOP` signal may interrupt a tile; that tile is reprocessed on
resume. Already-published bundles are checksummed and skipped. Resume removes
`STOP` and `PAUSE`. Changing the grid, source manifest/checksums, model, preprocessing, AMP,
dependency versions, or pipeline code changes compatibility and requires a new
output directory. Batch size, worker count, and prefetch depth may change during
resume. A corrupted published bundle fails verification; investigate it rather
than mixing old and new output.

Search a completed or safely paused statewide run with:

```powershell
.\scripts\search-nys.ps1 -Query 'fighter jet' -Offline -TopK 10
```

Paused collections are searchable automatically; results record the number of
embeddings/source tiles searched and indicate partial coverage. Use
`-AllowPartial` explicitly for published bundles from a failed/interrupted run
whose collection is idle. Active workers are rejected before loading the query
model. A collection with no saved vectors asks you to finish at least one tile.
Custom collections require matching `-OutputDir` for embedding/control and
`-CollectionDir` for search. Build the CPU IVF-PQ index with
`scripts/build-search-index.ps1` for interactive retrieval. The default search
uses that index when its snapshot matches, then rescores candidates using the
original vectors. `-Backend Exact` retains the full 24.55 GB scan as a baseline.
Missing or stale indexes automatically use exact search. See the
[search index guide](nys-search-index.md) for benchmarks, quality controls,
rebuilding after collection changes, and the future decision-model interface.
The matched original source
TIFFs are checksummed before rendering. Result folders contain labelled PNGs,
`contact-sheet.png`, and `results.json` with scores, source windows, shard IDs,
and coordinates. Rendering accepts 1 to 100 results.

## Directory and coordinate contract

```text
data/nys-naip-2022-rgb-060m/
  imagery/utm17/*.tif       # Immutable RGB sources; original paths preserved
  imagery/utm18/*.tif
  imagery/utm19/*.tif
  records/                 # Downloader provenance and source checksums
  metadata/                # Provider indexes
  manifest.json
  completion.json          # Completed imagery acquisition
  tiling/native060cm-s256-st192/plan.json
  embeddings/georsclip-vith14-s256-st192/
    run.json               # Entire collection recipe and compatibility hash
    collection.json        # Atomically updated published-source catalog
    status.json
    completion.json        # Only after the whole collection is done
    worker.lock
    PAUSE                  # Zero-byte signal while a pause is requested
    runtime/*.log          # Background worker console/error logs
    search-index/          # Derived CPU index; never replaces original bundles
      index-HASH.faiss
      manifest.json        # Collection mapping, snapshot, checksum, parameters
      status.json          # CPU build percent, elapsed time, ETA
    shards/SOURCE_TILE/
      embeddings.f16.npy
      chips.geoparquet
      run.json
  embeddings/smoke-TIMESTAMP/
  search-results/TIMESTAMP-query/
```

Each source's three bundle files are first written in a hidden staging
directory and then published together through an atomic directory rename.
Never mix files from different bundles. `embedding_id` is the vector row number
**within its shard**; the unique collection lookup key is `(shard_id,
embedding_id)`. `chip_id` also captures dataset, source, scale, and pixel origin.

GeoParquet polygons, centroids, and display bounds use **EPSG:4326 WGS84** across
all zones. Boundaries are densified before transformation. `crs_wkt`, native
bounds, pixel windows, and window affine transforms preserve each TIFF's
NAD83 UTM grid (EPSG:26917/26918/26919). Unified display coordinates do not change
the source resolution or distort the imagery through reprojection. Ground
accuracy still depends on the provider and map projection.

## Verification

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests use synthetic rasters/vectors to check edge coverage, masks, RGB order,
WGS84 geometry in all zones, normalized cosine ranking, row alignment, atomic
publication, corruption detection, interrupted-run resume, and folder budgets.
They also verify graceful pause, partial search, cross-process exclusion,
pause timeouts, and shared GPU ownership.
No real imagery or model checkpoint is included as a test fixture.

CPU-only selected-size previews are available with:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m geospatial_search.preview_tiles
```

Imagery attribution: Courtesy USDA-FSA; distributed by NOAA Digital Coast.
See the [download guide](nys-naip-download.md) for provider metadata and use
conditions, and [CITATION.cff](../CITATION.cff) for GeoRSCLIP citation.
