# Current project state

Last verified: **2026-10-14**, on the existing Windows checkout. This is a
handoff snapshot, not a live status report. Read [../AGENTS.md](../AGENTS.md)
for permanent constraints. Check configuration, manifests, collection catalogs,
and active OS locks before acting. Update this document after material changes.

## What exists

Local semantic search over NYS aerial imagery, with an Electron desktop map,
native-chip inspector, saved searches, and GPU model preload. Agent-development
improvements are on `remote`; `main` remains the existing development baseline.
`prototyping` was previously merged and removed.

| Collection | Source TIFFs | Published vectors | Local state |
| --- | ---: | ---: | --- |
| Frozen NYS 2022 NAIP base | 3,805 | 11,985,388 | Complete |
| Separate 2021-release gap supplement | 75 | 229,533 | Complete |
| Combined search | 3,880 | 12,214,921 | Available |

The supplement fills two audited interior land gaps. Its eastern acquisitions
are September/November 2021; western acquisitions are April 29, 2022. Release
year and acquisition date are distinct. This does not establish complete
coverage of every coastal/open-water omission. Valid base pixels take display
precedence; Settings and CLI support strict base-only imagery/search.

## Architecture and model

- `src/geospatial_search/`: acquisition, lazy native TIFF windows, pinned model,
  atomic per-source NumPy/GeoParquet/run bundles, pause/resume, and retrieval.
- Base retrieval uses CPU Faiss IVF-PQ with original-vector rescoring. The small
  supplement uses bounded exact search; compatible scores merge across datasets.
  Qualified seen IDs support requests for additional matches. See
  [nys-search-index.md](nys-search-index.md).
- `frontend/ui/` is TypeScript/Leaflet; `frontend/desktop/` is Electron;
  `frontend/service/` is an authenticated loopback Python service using existing
  backend helpers. It builds display overviews/cache and reads source imagery.
- UI transport/types and tile scheduling now have separate modules. Service
  HTTP/security routing is in `http_server.py`; `server.py` retains application
  orchestration and re-exports the transport classes. Exact retrieval receives
  progress callbacks rather than replacing the backend module's global tqdm.
- Isolated development uses `.venv-dev` with pinned CPU test dependencies,
  marked synthetic TIFF/vector fixtures, and a separate mock query/model service.
  The four fingerprinted core source files and production configs are unchanged.
  See [development](DEVELOPMENT.md), [architecture](ARCHITECTURE.md), and
  [contracts](CONTRACTS.md). Windows CI covers offline and synthetic browser tiers.
- Map zoom retains and scales loaded tiles until replacements fade in. Model
  preload reuses one GPU process; history cleanup covers frontend and backend
  saved searches while preserving source data and embeddings.

Both configs pin `BiliSakura/GeoRSCLIP-ViT-H-14` to
`09af41b95fbb7b4bf884904e0d3c2777e83740bc`, using OpenCLIP `ViT-H-14` and
`laion2b_s32b_b79k` preprocessing. Native RGB is 0.6 m/pixel. Chips are 256 x 256
pixels (153.6 m square), stride 192 (25% overlap), at least 95% valid. Vectors
have 1,024 dimensions, are normalized in float32, and stored as float16.
Geographic metadata uses EPSG:4326, longitude first; source grids stay NAD83 UTM.

## Locations and environment

All paths below are relative to the repository root. Let **JOB** mean
`data/nys-naip-2022-rgb-060m/` and **SUPPLEMENT** mean
`JOB/supplements/naip-2021-release-gapfill/`.

| Artifact | Location |
| --- | --- |
| Immutable base TIFFs | `JOB/imagery/utm17`, `utm18`, `utm19` |
| Base manifest/provenance | `JOB/manifest.json`, `records/`, `metadata/` |
| Base bundles and index | `JOB/embeddings/georsclip-vith14-s256-st192/` (`shards/`, `search-index/`) |
| Supplement TIFFs/provenance | `SUPPLEMENT/imagery/utm18/`, `manifest.json`, `records/`, `metadata/` |
| Supplement bundles | `SUPPLEMENT/embeddings/georsclip-vith14-s256-st192/` |
| CLI search results | `JOB/search-results/` |
| Frontend saved searches/query vectors | `frontend/state/searches/`, `frontend/state/text-vectors/` |
| Frontend display cache/logs | `frontend/cache/`, `frontend/logs/` |
| Ignored working notes | `temp/`; frontend-specific notes may use `frontend/temp/` |

Use the existing `.venv` (Python 3.12) and Node dependencies in
`frontend/node_modules/`. The machine uses an AMD RX 9070 XT with ROCm-enabled
PyTorch exposed through `torch.cuda`; no silent CPU fallback. See
[rx-9070xt-embedding-setup.md](rx-9070xt-embedding-setup.md).
The job's shared cap is 300 decimal GB with a 5 GB reserve. Frontend display
cache has separate bounded accounting, currently 5 GB in `frontend/settings.json`.
Weights may use the existing external Hugging Face cache.
Use `.venv-dev` for synthetic checks, never install its CPU dependency snapshot
into `.venv`. Runtime observations do not authorize changes to an active owner.

## Working commands

Run from the repository root; GPU queries require idle embedding workers and
available shared locks. Unload a resident frontend model before CLI queries or
embedding. Download and embedding are already complete on this machine.

```powershell
.\frontend\scripts\launch.ps1
# Isolated CPU checks and synthetic UI; no production data/GPU required.
.\scripts\validate.ps1 -Mode Offline
.\scripts\validate.ps1 -Mode Docs
.\scripts\validate.ps1 -Mode SyntheticBrowser
.\frontend\scripts\dev-synthetic.ps1
# Small read-only metadata/lock observations; no model loading by default.
.\scripts\doctor.ps1
.\scripts\search-nys.ps1 -Query 'baseball field' -TopK 10 -Offline
.\scripts\search-nys.ps1 -Query 'baseball field' -BaseOnly -Offline
# Read-only progress snapshots; these do not start embedding.
.\scripts\embed-nys.ps1 -Status
.\scripts\embed-nys-gapfill.ps1 -Status

$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
$env:PYTHONPATH = 'src;frontend/service'
.\.venv\Scripts\python.exe -B -m unittest discover -s frontend/tests -v
.\frontend\scripts\build.ps1
node .\frontend\tests\verify-zoom.mjs
```

Pause/resume, acquisition, and smoke-test procedures are in
[nys-pipeline.md](nys-pipeline.md), [nys-naip-download.md](nys-naip-download.md),
and [nys-gapfill.md](nys-gapfill.md). Frontend test/packaging procedures are in
[../frontend/AGENTS.md](../frontend/AGENTS.md). Last relevant verification:
2026-10-14 isolated validation passed 81 backend and 58 frontend service tests
with no skips, scoped Ruff lint/format, TypeScript/Vite, dependency consistency,
and local documentation checks. Synthetic headless Edge smoke passed and its
screenshot was inspected. The earlier Windows package was not rebuilt for these
developer changes; rebuild packaging explicitly before using an updated package.
Real-imagery UI and delayed single-step/rapid zoom checks also passed, with
screenshots inspected, no browser runtime errors, and no real GPU inference.
PowerShell scripts parsed successfully. The explicit GPU smoke mode was added
but not run; no full embedding job was launched.

## Known limitations and discussed next steps

- Data paths are checkout-relative, not yet a configurable external job root.
  A fresh clone/worktree lacks local imagery, vectors, weights, environments,
  and generated assets. Do not move the existing dataset or assume a new
  worktree shares it. Synthetic development now works without those assets;
  production job relocation/configuration is still separate work.
  Supplement preparation also requires its ignored local
  audit; see the gap-fill guide.
- Embedding recipes fingerprint source bytes. `.gitattributes` preserves LF
  for pipeline Python files; avoid changing a running worker's checkout or
  fingerprinted code. Cold native display tiles can take longer than cached
  tiles. The packaged executable needs its full directory and this checkout.
- If Windows reports `WinError 4551` loading ROCm, use
  [../frontend/SEARCH_TROUBLESHOOTING.md](../frontend/SEARCH_TROUBLESHOOTING.md).
  Historical security-setting observations are not a current machine diagnosis.
- The decision-model feedback loop is planned, not implemented. Before building
  it, agree on relevance evaluation, additional-match requests, and stopping
  criteria. Read-only environment diagnostics are now implemented through
  `scripts/doctor.ps1`; they do not perform full integrity scans or reserve locks.
  Configurable production data-root support remains discussed future work.
