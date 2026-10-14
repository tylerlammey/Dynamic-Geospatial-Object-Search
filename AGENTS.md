# Dynamic Geospatial Object Search: Agent Guide

## Starting a new task

1. Read [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) for a substantial task.
   Inspect `git status --short` and preserve unrelated local changes.
2. Find the applicable scoped `AGENTS.md` before editing. Frontend work also
   follows [frontend/AGENTS.md](frontend/AGENTS.md).
3. Use the task map below to find relevant code, contracts, and checks. Read
   detailed operating guides when the task involves their pipeline operations.
4. Identify active owners before work involving data, environments, GPU, or
   services. [scripts/doctor.ps1](scripts/doctor.ps1) observes prerequisites
   without repairs, model loading, downloads, or job writes by default.

Explicit user instructions govern scope. Scoped guides supplement this root
guide. Project-state snapshots and discussed future work do not authorize new
work; live manifests, collection catalogs, configuration, and OS locks take
precedence over dated observations. A fresh clone/worktree has no production
dataset: start with [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Update docs/PROJECT_STATE.md after material changes to architecture, data,
models, operating commands, known limitations, or agreed next steps. Date local
observations; keep credentials, session tokens, PIDs, and temporary plans out
of that document. Keep durable engineering rules here or in scoped AGENTS.md
files, and temporary working notes under the ignored temp/ folder.

## Task map

| Task | Code and context | Checks |
| --- | --- | --- |
| Acquisition, budget, provenance | `download_nys_naip.py`; [download guide](docs/nys-naip-download.md) | Backend offline suite |
| Tiling, masks, embedding, pause/resume | `tiling.py`, `model.py`, `bundles.py`, `embed_georsclip.py`, `embedding_control.py`; [pipeline guide](docs/nys-pipeline.md) | Backend suite; explicit GPU smoke before a full GPU run |
| Retrieval and index | `search_text.py`, `search_index.py`, `federated.py`; [index guide](docs/nys-search-index.md) | Backend and affected service suites |
| Supplement | `gapfill*.py`, `*_gapfill.py`; [gap-fill guide](docs/nys-gapfill.md) | Backend and affected service suites |
| UI/map | `frontend/ui/`; [UI guide](frontend/ui/AGENTS.md) | TypeScript/build; applicable browser checks |
| Service/cache/history/workers | `frontend/service/`; [service guide](frontend/service/AGENTS.md) | Frontend suite; applicable browser checks |
| Developer tooling/CI | `scripts/`, `.github/`; [development guide](docs/DEVELOPMENT.md) | Offline validation and Docs mode |
| Documentation only | Authoritative guide for the subject | Docs mode; no GPU or full rebuild |

Code names without a directory above are under `src/geospatial_search/`.
See [architecture](docs/ARCHITECTURE.md) for request/ownership boundaries and
[contracts](docs/CONTRACTS.md) for validators and persisted formats.

## Subagents and delegation

Use subagents when a task can be divided easily into independent, bounded
subtasks and delegation is likely to reduce elapsed time or improve review
quality. Examples include independent investigations, separate test areas,
documentation review, and implementation in separate files. Keep small or
tightly coupled tasks with the parent.

Spawned agents must never use a higher model tier than their parent. Prefer
inheriting the parent's model. Explicit model selection requires a verified
equal or lower tier using the current runtime/provider's documented tier
mapping. Do not infer tier from names, version numbers, or reasoning settings.
If ordering is unavailable, inherit; if inheritance cannot be guaranteed, do
not spawn. Apply this rule recursively to every descendant.

Give each subagent an objective, relevant instructions, permitted files or
responsibility, dependencies, acceptance criteria, and required validation.
Assign separate file ownership for concurrent edits. Configuration, shared
contracts, dependencies, and PROJECT_STATE have one coordinated editor.
Preserve unrelated local changes. Coordinate GPU, services, environments, and
fingerprinted code through the parent; no competing owners or unapproved jobs.

Subagents report findings, changed files, validation, and unresolved issues.
The parent reviews/integrates the work, resolves conflicts, runs combined
validation, and owns the final result. Delegation does not expand user scope.

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

The source bytes of `embed_georsclip.py`, `tiling.py`, `model.py`, and
`bundles.py` participate in recipe compatibility. Formatting is a compatibility
change too. Do not run bulk formatters over these files or edit an active
worker's checkout. The supplement fingerprints additional adapter files;
consult `build_recipe` and the recorded recipe before changing them. Keep
readability refactors incremental and start outside the frozen core.

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

For coordinated developer validation, use `scripts/validate.ps1 -Mode Offline`;
it defaults to the isolated `.venv-dev`, includes both synthetic suites and the
UI build, and requires the ANN dependencies so index coverage cannot silently
skip. `-Mode Docs` checks local documentation links/command paths and whitespace.
`Browser` and `GpuSmoke` are explicit local tiers with additional prerequisites.
See [development commands](docs/DEVELOPMENT.md); validation never installs
dependencies. CPU PyTorch is allowed only for synthetic tensor tests, never as
a fallback for production inference.

## Completing a task

- Review the diff for unintended data, credentials, unrelated edits, and
  compatibility changes; run `git diff --check`.
- Run checks relevant to changed behavior. Add characterization/regression
  tests before changing masks, geometry, vector alignment, or ranking.
- Wait for local test/build commands to finish and inspect their exit status
  and final results. Starting a command is not evidence that checks passed.
- Update authoritative docs and PROJECT_STATE for material changes. Record
  current validation separately from historical results.
- Report behavior, checks run, skipped/unavailable tiers, and remaining limits.
  Commit/push only when authorized; stage explicit task-owned paths.

## GitHub validation after a push

After an authorized push, record the branch and full pushed commit SHA.

Documentation-only changes that cannot affect runtime, builds, dependencies,
data contracts, or validation do not require waiting for GitHub Actions.
Examples include prose edits to README/AGENTS files and additions to
PROJECT_IMPROVEMENTS.md. Review the complete pushed diff, run
`scripts/validate.ps1 -Mode Docs` and `git diff --check`, and report that local
checks passed and GitHub validation was not awaited. This exception applies
to this guidance update too; do not claim unobserved CI results passed.

Use the full verification below when any pushed change affects code, tests,
configuration, workflows, executable/generated inputs, or behavior-bearing
contracts, even if stored in Markdown. If the user explicitly requests CI
verification, wait for it regardless of the documentation-only exception.

For changes requiring GitHub verification, wait
for the expected GitHub Actions test workflows for that exact revision to
finish; inspect the applicable run, jobs, and required steps. A successful push
or an older green commit does not establish that the pushed changes passed.
Only completed successful required checks count as verified. Queued/running,
cancelled, skipped, missing, timed-out, or inaccessible checks must be reported
accurately and must not be called green. Explicitly optional checks remain
optional; explain their scope rather than implying they ran.

On failure, inspect the failed step/log, address issues within the authorized
task scope, run relevant local checks, and wait again after any new push. Each
new commit needs its own results; a rerun of an old SHA does not test new code.
For an authorized PR merge, also verify the applicable merge-result checks.

Use available connected tools, authenticated CLI/API access, or the GitHub UI.
Poll with bounded waits/backoff and keep the user informed during the wait.
If access or an external blocker prevents confirmation, report the exact SHA,
run URL/status when available, and limitation; do not claim CI passed. Include
the verified run link and result in the final handoff. This requirement does
not authorize committing, pushing, merging, changing workflows, or GPU work
beyond the user's existing scope.

## Git policy

Commit code, docs, configuration, dependencies, and legally shareable synthetic
tests. Keep imagery, data, output, checkpoints, caches, virtual environments,
credentials, and temp files out of Git. Preserve useful local provenance.
Check imagery licensing and coordinate-sharing conditions before publishing.
