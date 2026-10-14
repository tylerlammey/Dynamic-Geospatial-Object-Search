# Development without production assets

Use Windows, Python 3.12, PowerShell, Node.js 22.14+ (or installed Node 24), and
Git. Read [AGENTS.md](../AGENTS.md) first. Fresh clones need no NYS imagery,
vectors, GPU drivers, or model weights for this workflow.

## Isolated setup

From the repository root:

```powershell
.\scripts\setup-dev.ps1
Push-Location frontend
try { npm.cmd ci --ignore-scripts --no-audit --no-fund } finally { Pop-Location }
.\scripts\validate.ps1 -Mode Offline
.\scripts\validate.ps1 -Mode Docs
```

Setup creates only ignored `.venv-dev/`. CPU PyTorch/torchvision come from the
official CPU index; the remaining development dependencies are pinned through
[requirements-dev.txt](../requirements-dev.txt) and
[constraints-dev.txt](../constraints-dev.txt). OpenCLIP is installed because
resume tests inspect distribution metadata even with inference mocked. No
checkpoint is downloaded. CPU tensor tests do not authorize CPU production
inference. The live `.venv`, ROCm packages, and frozen data remain intact.

This tested Windows/Python 3.12 snapshot is independent of fingerprinted
production dependencies. Update it only in `.venv-dev`, run `pip check` and
Offline validation, and record new verification. Do not apply it to `.venv`.
The tracked [npm lockfile](../frontend/package-lock.json) pins UI dependencies.
Installing with `--ignore-scripts` supports UI builds without an Electron runtime
download. Desktop launch still needs the explicit [frontend setup](../frontend/README.md).
Validation never installs dependencies implicitly.

## Validation tiers

| Mode | Checks | Requirements/effects |
| --- | --- | --- |
| `Offline` | Both synthetic Python suites, required Faiss coverage, scoped lint/format, TypeScript/Vite | `.venv-dev`, npm dependencies; ignored fixture/build outputs |
| `Docs` | Local Markdown links, repository command/module paths, changed whitespace | Development or existing interpreter; no external links fetched or commands executed |
| `SyntheticBrowser` | Synthetic HTTP/imagery and UI interaction via headless Edge | `.venv-dev`, Node, built UI, Edge; isolated owned service/profile/reports |
| `Browser` | Real-imagery UI and delayed/rapid zoom checks | Production `.venv`, Edge, local imagery; reuses available desktop service |
| `GpuSmoke` | Bounded image embedding and exact text search of that same new smoke collection | GPU environment, cached weights, imagery, free locks, shared job budget |

```powershell
.\scripts\validate.ps1 -Mode Offline -PythonExecutable '.venv-dev/Scripts/python.exe'
.\scripts\validate.ps1 -Mode SyntheticBrowser
# Include new Markdown before staging; unrelated untracked notes are ignored.
.\scripts\validate.ps1 -Mode Docs -DocsFiles docs/DEVELOPMENT.md,docs/ARCHITECTURE.md
# Focused diagnosis keeps existing unittest entry points.
$env:PYTHONPATH = 'src;frontend/service'
.\.venv-dev\Scripts\python.exe -B -m unittest discover -s tests -p test_doctor.py -v
```

Offline fails on skipped or zero discovered tests. Missing ANN dependencies
cannot give a misleading green result. Windows CI runs the same wrapper and
the synthetic browser check. Original focused unittest commands can still skip
optional index tests; report that coverage accurately.

GPU smoke is an explicit local tier, not a prerequisite for developer tools or
docs. It writes a unique bounded collection/results in the existing guarded
job, never starts statewide embedding, and never pauses another owner. Browser
checks do not load a real model; search/model actions are mocked, borrowed
services stay open, and only owned processes are stopped.

Ruff's explicit scope in [pyproject.toml](../pyproject.toml) covers developer
tools and new/refactored boundaries. Expand it incrementally with tests; do not
bulk-format fingerprinted pipeline source. Wrappers restore their temporary
environment changes and propagate failures.

## Synthetic UI and fixtures

```powershell
# Build, generate a unique fixture/state, and print a tokenized local URL.
.\frontend\scripts\dev-synthetic.ps1
# Ctrl+C stops this owned foreground service.

# Generate a reusable fixture; output must be a new directory.
.\.venv-dev\Scripts\python.exe -B scripts/create_synthetic_job.py --output temp/synthetic-job
.\scripts\validate.ps1 -Mode SyntheticBrowser
```

The generator writes two small native RGB/masked TIFFs and atomic bundles with
16-dimensional seeded unit vectors and a distinct synthetic model contract.
Output is allowed only beneath ignored `temp/` or `frontend/temp/`. Existing
directories and symlink/junction redirection are rejected. A completion marker
publishes last; the dev service verifies source/bundle provenance before use.

The separate dev service uses real HTTP authentication, imagery rendering,
cache, and isolated history/bookmarks with mocked model/text encoding. A
persistent banner identifies generated patterns and mock responses. Scores
exercise wiring and do not measure semantic relevance. Production desktop and
service launchers never select this entry point.

The dev launcher accepts `-Python` and `-SkipBuild`. The automated smoke supports
`FRONTEND_DEV_PYTHON` and `FRONTEND_EDGE_PATH`. Tokens remain in ignored state;
do not copy tokenized URLs/ready files into Git or project-state documentation.

## Read-only diagnostics

```powershell
.\scripts\doctor.ps1
.\scripts\doctor.ps1 -Json
# Optional GPU capability query, using the live interpreter explicitly.
.\scripts\doctor.ps1 -PythonExecutable '.venv/Scripts/python.exe' -GpuProbe
# Optional full read of the selected index file, never statewide imagery.
.\scripts\doctor.ps1 -VerifyIndexChecksum
```

Default interpreter selection is `.venv-dev`, then `.venv`, then `python`.
Diagnostics inspect package metadata without importing torch, read small
manifests/catalogs, probe existing lock bytes without creating files, compare
index snapshot/size, and observe disk free space and pinned cache-file presence.
They do not recursively scan job usage, load native indexes/models, download,
repair, pause workers, or change environments.

`-JobRoot`, `-CollectionDir`, `-Config`, and `-ModelCache` select observations.
Relative job/collection/config paths resolve against the repository. ModelCache
names the HF hub cache. Errors fail; missing production assets warn because
fresh clones normally lack them. `-Strict` also fails warnings. GPU probing
imports PyTorch and asks device availability without loading a model; index
checksumming can read a large file and is opt-in. Observations can race and
never reserve ownership. See [contracts](CONTRACTS.md) for the report format.

## Existing checkout and worktrees

Keep completed sources, provenance, and bundles at their existing paths. A new
worktree does not inherit ignored data, environments, or generated builds.
Repeat isolated setup there; do not copy production imagery or link shared
writable state just to pass tests. Configurable production external-data roots
remain separate work.

Follow root subagent/model-tier rules, assign independent file ownership, review
combined diffs, and validate integration. Historical PROJECT_STATE test counts
are context, not evidence that today's checkout passes.
