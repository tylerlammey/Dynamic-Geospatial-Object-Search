# Frontend agent guide

These instructions apply under `frontend/` and supplement the root AGENTS.md.
For a new substantial frontend task, read `../docs/PROJECT_STATE.md`. Use
`README.md` here for architecture, operation, and detailed testing instructions.

## Scope and invariants

- Keep frontend changes under this folder unless the user authorizes backend
  changes. Existing `geospatial_search` helpers may be imported; do not change
  their pinned model, collection contracts, source data, or compatibility hashes.
- `ui/`: TypeScript/Leaflet; `desktop/`: Electron launcher; `service/`: local
  Python API, imagery display, cache, and retrieval integration; `scripts/`:
  PowerShell setup/build/launch; `tests/`: synthetic and live browser checks.
- Original TIFFs and embedding bundles are read-only. Display reprojection and
  downsampling belong here. Keep PAM disabled; do not write sidecars or source
  overviews. Preserve validity masks, dataset-qualified IDs, acquisition dates,
  valid-base precedence, and base-only display/search.
- During zoom, cancel obsolete unfinished requests and keep decoded imagery
  available for Leaflet's scaling/fade/pruning. Keep map fetches and retention
  bounded; do not queue all statewide TIFFs or introduce online basemaps.
- Preserve authenticated loopback access, origin checks, path containment, cache
  generation checks, and single-service cache ownership. Use `DiskCache` for
  derived display artifacts and retain its configured cap including staging.
- Source/embedding artifacts remain under the existing job and its budget.
  Generated frontend cache, builds, logs, profiles, previews, and small state
  stay in ignored frontend folders. Use `temp/` here for temporary frontend work.
- Query inference and resident model preload must retain collection plus
  job-wide GPU locks. No CPU fallback, model replacement, automatic embedding
  pause, or termination of a user's worker. Browsing imagery does not need GPU
  inference. Keep the AMD/ROCm environment and existing model cache intact.

## Validation by change

Run commands from the repository root. Select checks relevant to the change;
documentation-only edits need link/command checks and `git diff --check`, not
GPU work or a complete rebuild.

```powershell
# Python service/cache/retrieval changes: offline synthetic regressions.
$env:PYTHONPATH = 'src;frontend/service'
.\.venv\Scripts\python.exe -B -m unittest discover -s frontend/tests -v
# UI changes: TypeScript validation and Vite bundle.
.\frontend\scripts\build.ps1
# Real imagery/UI integration; starts an owned service if none is open.
node .\frontend\tests\verify-ui.mjs
# If the desktop is already open, reuse it instead of starting another owner.
node .\frontend\tests\verify-ui.mjs --existing-service --overview
# Zoom changes: delayed replacements, scaling, pruning, rapid in/out.
node .\frontend\tests\verify-zoom.mjs
```

The zoom check automatically reuses an available desktop service through a
read-only proxy serving the current UI build. Otherwise it starts its own
service. Live checks require installed Edge and local imagery; `--overview`
also requires a prepared display overview. Tests write ignored reports and
screenshots under `tests/artifacts/`; they must not load a real GPU model, clear
user history, pause embedding, or close a borrowed service. Search/model/history
actions in the general UI check are intercepted fixtures. Inspect screenshots
when verifying a visual change. Stop only test-owned processes.

## Launch and package

```powershell
.\frontend\scripts\launch.ps1
.\frontend\scripts\build.ps1 -Package
```

The launcher builds changed UI source and opens the desktop. Packaging copies
service code and built UI into `dist/win-unpacked/`; rebuilding only `build/`
does not update a packaged app. Keep that directory together; it still depends
on the checkout and `.venv`. Coordinate with the user before closing their app
to replace a locked package. Reload/reopen to apply a new UI bundle.

Keep operating details in this folder's README and update PROJECT_STATE after
material architecture/behavior changes. Commit code, docs, and test programs;
keep real imagery, embeddings, caches, build output, credentials, and session
tokens out of Git. Follow the task's explicit commit/push instructions.
