# NYS Imagery Search desktop

The desktop has ranked results on the left, a zoomable local imagery
map in the center, and a native-chip/context inspector on the right. It reads
the existing NYS 2022 NAIP TIFFs and compatible published GeoRSCLIP bundles.
The application does not upload imagery or require an online basemap/account.

## Launch on this checkout

The dependencies, UI bundle, and Windows application have already been built
on this machine. From the repository root:

```powershell
.\frontend\scripts\launch.ps1
```

The launcher opens a visible desktop window and confirms that it appeared
before reporting success. Repeating it shows and focuses the existing window.
It builds changed UI source when necessary. Closing the
window stops only its own local service and any query process it started.
It does not pause or terminate the embedding worker.

If startup fails, the launcher reports an error instead of silently returning.
See `frontend/logs/desktop-launch.log` for desktop startup and window visibility,
and `frontend/logs/desktop-service.log` for the local Python service.

A packaged Windows x64 application is also available:

```powershell
& '.\frontend\dist\win-unpacked\NYS Imagery Search.exe'
```

Keep the entire `win-unpacked/` folder together at this location. This is an
unsigned checkout-based build, not a standalone portable imagery installation
or an installer. It still uses this checkout's Python environment, backend
modules, configuration, original imagery, and model cache. Moving the executable
alone will not work. To launch a copied build against an existing checkout, set
`NYS_FRONTEND_ROOT` to that checkout's existing `frontend` directory first.

## Use the map

The initial view fits the downloaded NYS extent. Pan, mouse-wheel zoom, the
zoom buttons, and **Fit state** work locally. **Click actual imagery** to inspect
an original 256 × 256 source-pixel chip and its surrounding imagery without a
text query or GPU. A gold map outline identifies that native window. The context
thumbnail opens a larger view. Panel borders can be dragged, or resized with
arrow keys when focused. Ctrl+K focuses text search.

Statewide views now use a photographic EPSG:3857 display mosaic at 128 m per
pixel with internal overview levels, rather than sparse sampled source colors.
Zoom levels 5-10 read only a window of that prepared display raster. Levels
11-13 use cached, mask-aware averaged 512 x 512 photographic overviews for each
source. Zoom 14 and above reads the needed native windows. All display images
are reprojected and interpolated for viewing; the embedding/source grids remain
unchanged.

The initial photographic preparation reads each original source once. Use
**Settings > Prepare / resume photographic overview** while embedding is idle
or manually paused. Progress is visible on the map. Prepared source overviews
are reusable if preparation is interrupted; a resumed run rebuilds the modest
statewide mosaic from them. The app refuses to start this substantial CPU work
while embedding is active and stops preparation if embedding resumes. It never
pauses embedding itself.

The initial pilot estimated roughly 10-20 minutes and 1-3 GB of generated
imagery; actual progress reports the measured ETA. Prepared data stays under
`frontend/cache/` and shares its 5 GB guard, including atomic temporary writes.
A statewide display request opens one derived raster, not thousands of TIFFs.
Visible tiles are cached, browser fetches are bounded, and obsolete requests
are cancelled both in the browser and in queued/source-loop service work.

If the photographic overview has not been prepared, the app shows preparation
instructions at statewide zoom instead of displaying a misleading pixelated
patchwork. Native close-up imagery remains available. Clearing the imagery
cache also clears these derived overviews; prepare them again afterward.

Pixels, masks, and source CRS are read-only. Source TIFFs receive no overviews,
sidecars, compression changes, or rewritten metadata. Original sources remain
NAD83 UTM17/18/19; the display uses EPSG:3857 and geographic overlays use
EPSG:4326, longitude first. The map, source grid, and search-chip grid are
distinct. Zooming beyond native 0.6 m/pixel does not add imagery detail.

Imagery attribution: **Courtesy USDA-FSA; distributed by NOAA Digital Coast**.
Existing imagery and model licensing/coordinate-sharing conditions continue
to apply. No imagery or location artifacts were published.

## Search and saved results

Use **Load model to GPU** in the header to load and warm up the existing pinned
GeoRSCLIP checkpoint once. The button changes to **Model on GPU** when ready;
later text queries reuse the same process/model and its in-memory search index.
Loading is optional: ordinary search still works without it. The frontend
shows loading errors and never falls back to a CPU embedding model.

The loaded model reserves both the collection and job-wide GPU locks, including
between searches. Choose **Unload model** before resuming embedding or running
CLI searches. It finishes an active query before allowing manual unload;
closing the frontend releases its own worker and locks. Loading does not create
image embeddings or change the checkpoint or collection. Model status and logs
stay under `frontend/logs/model-status.json` and `model-worker.log`. Clearing
search history leaves the resident model loaded.

If search reports **WinError 4551** loading AMD PyTorch's `aotriton_v2.dll`, see
[the Windows Application Control diagnosis](SEARCH_TROUBLESHOOTING.md). This
machine previously blocked that unsigned DLL through Smart App Control. Resolving
it requires a Windows security decision or a compatible trusted signed runtime;
frontend setup and imagery preparation do not change those settings.

Enter a phrase and a result count (1–5,000). **Balanced** uses the compatible
CPU IVF-PQ index and rescores candidates from original vectors. **Thorough**
expands both the indexed search and candidate pool; **Exact** scans all vectors.
A missing or stale index falls back to exact search. The collection/model
compatibility checks and original source coordinates remain in effect. See the
[search index guide](../docs/nys-search-index.md) for measured speed and recall.
Only 20 thumbnail results are rendered per page; candidate count and thumbnail
rendering are separate. Clusters retain all overlapping matches. Select a
cluster rank or card to synchronize the footprint and inspector. Use previous/
next and **Fit all matches** to navigate. Similarity is a ranking score, not
object-detection confidence.

Search history includes **compatible existing CLI searches** from the
NYS `search-results/` folder and new frontend searches. You can open their real
chips while embedding runs. Historical CLI results include their searched source
count but not the full shard list: their coverage overlay is labeled as the
current published coverage. New frontend searches store their immutable source
snapshot, and the overlay shows that exact searched coverage. All saved-result
metadata is checked against compatible published bundles before use.

Frontend result JSON files live in `frontend/state/searches/`. In **History >
Clear all search history**, confirm removal to delete these results plus cached
query vectors (`frontend/state/text-vectors/`) and per-query logs
(`frontend/logs/search/`). The dialog reports storage reclaimed and refreshes
the saved-search list, and resets the displayed results, markers, query, and
chip inspector. It also deletes saved CLI result JSON, rank PNGs, and contact
sheets under `data/nys-naip-2022-rgb-060m/search-results/`, including incomplete
search output folders. Source imagery, embeddings, downloader provenance,
bookmarks, and the loaded GPU model are preserved. Cleanup refuses redirected
paths or unexpected files and waits for an idle search/model. With no resident
model, it acquires the existing collection/GPU locks to prevent racing a backend
search. The legacy API confirmation `frontend-history-only` still scopes cleanup
to frontend files; the UI now explicitly confirms `all-search-history`.
**Settings > Clear frontend imagery cache** separately clears cached imagery
and overviews.

If the collection or job-wide GPU lock is held, a new query is rejected before
loading a model. The app shows the existing manual command:

```powershell
.\scripts\embed-nys.ps1 -Pause
```

The frontend never executes that command or writes pause/stop signals. Wait for
the existing command's acknowledgement if you choose to pause manually. Map
browsing remains available throughout. Idle interrupted collections require an
explicit **Settings** checkbox; paused partial collections are already allowed.

Each query process holds both original collection/GPU locks, reuses the pinned
model loader and float32 text encoding, and releases GPU allocations before
releasing locks. Repeated text queries can reuse a CPU query vector tied to the
collection fingerprint. Explicitly loading the model keeps it resident between
queries. Cached
result chips resolve by `(shard_id, embedding_id)` back to original pixel windows.

Bookmarks live under `frontend/state/`. The latest 30 frontend searches and 200
CPU text vectors are retained. Clear all search history also removes backend
saved searches.
The retrieval boundary is `service/retrieval.py`; future consumers can request
more results without changing the renderer. Requests accept search quality
and qualified already-seen IDs; the future decision model has not been added.

## Cache and process limits

`settings.json` controls the default 5,000,000,000-byte imagery cache, two CPU
imagery workers, offline model loading. Relaunch after
editing it. The cache limit applies to disk tiles and previews, including staged
temporary files during atomic publication. LRU eviction happens before a write.
This frontend cache has separate accounting from the existing NYS job budget.
Browser profile, logs, and small persisted JSON state are outside the imagery
cache accounting; all still live under `frontend/`.

**Settings → Clear frontend imagery cache** requires an explicit confirmation
and can delete only generated `frontend/cache/` files. It preserves imagery,
embeddings, bookmarks, and search history. Active cache writes use a generation
check so requests started before clearing cannot repopulate cleared entries.
Only one frontend service can own this shared cache at a time. Close a browser
development service before opening the desktop app, or vice versa.

There are at most 16 HTTP handlers, 2 active raster work items by default, and
a bounded raster queue. Raster handles are opened and closed per source read,
not accumulated for all TIFFs. GDAL cache/warp memory is bounded and PAM writes
are disabled. The browser fetches at most four map tiles concurrently, with a
one-tile retention margin. Electron's hardware acceleration is disabled so map
rendering does not depend on the embedding GPU.

The service binds only to `127.0.0.1` on an automatically allocated port. A
per-launch session token, loopback Host validation, same-origin checks, input
validation, content security policy, and path containment protect local APIs.
The desktop renderer has no Node integration; it uses context isolation and
sandboxing. Only frontend-owned child processes are stopped on shutdown.

## Setup, development, and rebuilding

Prerequisites: Windows x64; Node.js 22.12+ or the installed Node 24; npm; this
project's existing Python 3.12 environment with Rasterio, NumPy, Pillow, Shapely,
PyProj, and the existing embedding/search dependencies. The live `.venv` is
reused read-only. No Python dependency was installed or changed. Set
`NYS_FRONTEND_PYTHON` to an appropriate isolated frontend interpreter if needed;
it must also contain the existing backend's query dependencies for GPU search.
Install extra Python packages only into `frontend/.venv/`, never the live worker
environment. Python always runs with `-B` and GDAL PAM disabled.

For a fresh checkout with the existing data/environment already configured:

```powershell
# Download frontend dependencies; caches and lockfiles stay in frontend/.
.\frontend\scripts\setup.ps1

# Generate/resume photographic source overviews and the statewide pyramid.
# Works with the app open or closed. Run while embedding is idle/paused.
.\frontend\scripts\prepare-imagery.ps1

# Alternatively, install/build, prepare imagery, and open the app.
.\frontend\scripts\setup.ps1 -PrepareImagery

# Build TypeScript and bundled local assets.
.\frontend\scripts\build.ps1

# Development desktop run (automatically rebuilds changed UI).
.\frontend\scripts\launch.ps1

# Optional browser development view; Ctrl+C stops its owned local service.
.\frontend\scripts\serve.ps1 -Browser

# Build the checkout-based Windows app directory.
.\frontend\scripts\build.ps1 -Package
```

Setup deliberately skips npm lifecycle scripts and runs only the required
Electron runtime download explicitly. It does not add packages to root
requirements or the worker's environment. No Rust/MSVC toolchain is required
for this JavaScript-only desktop build. The lockfile pins the installed package
versions. The packaged build has not been signed or distributed.

`setup.ps1 -PrepareImagery` opens the desktop window when preparation finishes,
including when the imagery is already ready. Add `-NoLaunch` for unattended
setup. Ordinary `setup.ps1` installs/builds and prints the launch command.

`prepare-imagery.ps1` attaches to an open app's preparation, or starts its own
hidden local service when the app is closed. It shows source count, progress,
and ETA; reuses completed source previews after interruption; and skips work
when the photographic overview is already ready. `-Force` rebuilds the statewide
display mosaic using cached source previews. Ctrl+C stops preparation only when
the script owns the service; when attached to an open app, it stops the monitor.
The script never pauses embedding or shuts down an existing app.

Folder layout:

```text
frontend/
  ui/                 TypeScript, HTML, CSS; Leaflet UI
  desktop/            Electron wrapper
  service/            Local imagery, cache, retrieval, query worker
  scripts/            Setup/prepare imagery/build/launch/browser development
  tests/              Synthetic and live browser verification
  settings.json       Frontend cache and worker settings
  package*.json       Frontend dependencies and lockfile
  build/              Ignored bundled UI
  dist/win-unpacked/  Ignored Windows application directory
  cache/              Ignored display tiles/native previews
  state/              Ignored bookmarks/searches/CPU query vectors
  logs/               Ignored owned process logs/status/protection records
  temp/               Ignored temporary files
  .tools/             Ignored npm/Electron/build download caches
  .browser-profile/   Ignored Electron/verification browser profiles
```

## Verification performed

Frontend service tests (synthetic fixtures under `frontend/tests/.tmp/`):

```powershell
$env:PYTHONPATH = 'src;frontend/service'
.\.venv\Scripts\python.exe -B -m unittest discover -s frontend/tests -v
```

They check photographic averaging and partial mask edges, prepared-state tile reads without opening originals, cancelled requests, all three UTM zones, RGB, internal mask transparency, native chip
windows, tile coordinate conversion, RGB previews, bounded atomic/LRU
cache writes including staging, cache-clear/path containment, service ownership,
query inputs, quality controls, seen-ID exclusion, partial published coverage, exact helper integration, model
contract rejection, and fingerprinted CPU text-vector reuse. GPU encoding is
stubbed in those tests.

Live browser checks:

```powershell
node .\frontend\tests\verify-ui.mjs

# Verify prepared statewide/county imagery against an already-open desktop.
node .\frontend\tests\verify-ui.mjs --existing-service --overview
```

These use installed Edge and the actual local TIFFs for a small area. They check
visible XYZ tile loading, zoom requests, map-click native previews, existing
compatible CLI search results, footprint/card/cluster selection, context,
previous/next, partial coverage, active-worker query rejection, API access
checks, local-only resources, runtime errors, and 1440 × 960 / 1093 × 720 /
1920 × 1080 / 900 × 700 viewports. Test-only search fixtures are intercepted in
the browser; they are never published into the real collection. Ignored
screenshots/report are in `frontend/tests/artifacts/`. This verifies CSS viewport
adaptation, not all physical Windows monitor/DPI combinations.

The prepared full-state, regional, county, and neighborhood views were also
verified against the real local collection. All 3,805 photographic source
overviews and the 6,966 x 5,488 statewide raster are prepared on this machine.

The search-index update passed **25 frontend service regressions** and **56
backend regressions**. The live browser check also verified Settings, all three
search-mode request payloads, photographic views, result selection, history,
context, and four viewport sizes. The Windows package was rebuilt successfully.

Real authenticated HTTP searches verified indexed retrieval and an immediate
Thorough request for additional unseen chip IDs. All 13 returned PNG previews
were compared pixel for pixel with their original TIFF windows. Completion
acknowledgement now waits for the owned query process to exit, allowing a later
decision model to request more matches without racing cleanup.

A real GPU text-encoding/CLI query used the existing pinned GeoRSCLIP checkpoint.
Original embedding-code hashes and all fingerprinted dependency versions match
the saved collection recipe. Imagery and embedding bundles remain at their
original paths. Generated indexes, benchmarks, previews, and browser artifacts
remain ignored. See the [search index guide](../docs/nys-search-index.md) for
measured full-state retrieval speed and recall.

Primary implementation references: [Leaflet 1.9.4 API](https://leafletjs.com/reference-1.9.4.html),
[Rasterio WarpedVRT](https://rasterio.readthedocs.io/en/stable/api/rasterio.vrt.html),
and [Electron security](https://www.electronjs.org/docs/latest/tutorial/security).
