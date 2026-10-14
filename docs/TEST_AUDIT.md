# Test audit

Reviewed **2026-10-14**, against `8ca50973` on `remote`. Scope: every current
Python test method (82 backend + 58 frontend service), all three browser
programs, the validation/documentation runners, setup, and the GitHub workflow.
Subtests exercise additional scenarios, so 140 methods is not a coverage metric.

This is a source/assertion review. The reviewed commit previously passed local
validation and the GitHub Windows workflow; no new GPU, live-imagery browser,
download, or production-state operation was performed for this audit. Proposed
test/runtime changes below have not been implemented by the audit. The only
engineering instruction change is waiting for local and exact-revision GitHub
results before reporting validation complete.

## Overall assessment

- Preserve the existing test cases. No exact duplicate methods justify deletion.
  Authentication, masks, ranking, budgets, and ownership deliberately recur at
  different boundaries: helper, integration, HTTP, and UI.
- The strongest coverage is synthetic file integrity, locks/pause, bounded
  storage, masks, exact scores, HTTP authentication, and destructive cleanup.
- The main problems are shallow assertions and missing adverse/boundary cases,
  not an excessive number of tests. Passing currently provides less evidence
  than several test names/reports suggest.
- The live UI program has an unsafe conditional real-query probe. Fix that
  before relying on its guarantee that no real GPU inference occurs.
- The clearest redundant execution is building the same UI twice in one CI job.
  Do not remove meaningful tests to save those few build steps.

## Complete Python inventory

Every method in these files was reviewed, including setup/teardown and relevant
production paths. Counts are discovered test methods, not individual assertions.

| File | Methods | Keep/change assessment |
| --- | ---: | --- |
| [test_download_nys_naip.py](../tests/test_download_nys_naip.py) | 23 | Keep budgets/provenance/concurrency; strengthen RGB, actual resume, transfer failures, and bounded synchronization |
| [test_embedding_control.py](../tests/test_embedding_control.py) | 4 | Keep OS ownership/pause; bound child waits and test additional failure transitions |
| [test_gapfill.py](../tests/test_gapfill.py) | 7 | Keep qualified federation and separate budget; add distinct RGB bands, mask boundary, provenance/compatibility rejection |
| [test_tiling_embedding.py](../tests/test_tiling_embedding.py) | 15 | Keep grid/bundle/resume; use independent spatial/grid oracles and failed-publication cases |
| [test_search_index.py](../tests/test_search_index.py) | 14 | Keep exact rescoring/integrity; add genuinely partial ANN selection and corruption/fallback boundaries |
| [test_search_progress.py](../tests/test_search_progress.py) | 3 | Keep; callback, wrapper, and CLI-renderer compatibility are distinct contracts |
| [test_doctor.py](../tests/test_doctor.py) | 5 | Keep read-only/missing/corrupt metadata checks; add held-lock and complete index/CLI observations |
| [test_synthetic_job.py](../tests/test_synthetic_job.py) | 5 | Keep deterministic fixture/provenance/isolation checks; expand redirected ancestors and children |
| [test_validation_tools.py](../tests/test_validation_tools.py) | 6 | Keep actual failure regressions; add full-wrapper failure propagation/environment restoration |
| **Backend total** | **82** | |
| [test_service.py](../frontend/tests/test_service.py) | 19 | Keep raster/cache/retrieval integration; split broad adapter case and expand cache-generation/restart/failure coverage |
| [test_model_residency.py](../frontend/tests/test_model_residency.py) | 9 | Keep owned lifecycle/real child; remove environment-specific PID inequality and add complete query subprocess integration |
| [test_indexed_retrieval.py](../frontend/tests/test_indexed_retrieval.py) | 6 | Keep option/serialization boundaries; rename one-shot-worker test and cover qualified IDs/base-only requests |
| [test_history.py](../frontend/tests/test_history.py) | 11 | Keep all cleanup safety cases; add actual redirected paths/process races where practical |
| [test_gapfill_display.py](../frontend/tests/test_gapfill_display.py) | 4 | Keep mixed display/import; strengthen cache reuse/source preservation and negative snapshot/base-only tests |
| [test_http_server.py](../frontend/tests/test_http_server.py) | 6 | Keep real socket transport tests; expand malformed requests and concurrency limits |
| [test_dev_server.py](../frontend/tests/test_dev_server.py) | 3 | Keep synthetic safety/integration; it exercises the separate development app, not production query inference |
| **Frontend service total** | **58** | |

## Browser and infrastructure inventory

| Program/check | Assessment |
| --- | --- |
| [verify-synthetic.mjs](../frontend/tests/verify-synthetic.mjs) | Keep CI smoke; generated TIFFs, real HTTP/display/UI, mock search/model. Expand adverse UI flows, zoom, and failure diagnostics |
| [verify-ui.mjs](../frontend/tests/verify-ui.mjs) | Keep local real-imagery rendering/intercepted interaction checks; remove conditional real query, verify borrowed code provenance, report only executed checks |
| [verify-zoom.mjs](../frontend/tests/verify-zoom.mjs) | Keep delayed/rapid retention tests; current-build proxy is useful. Add synthetic equivalent and bounded network/process cleanup |
| [run_tests.py](../scripts/run_tests.py) | Keep mandatory dependencies, fail-on-empty/skips, and actual exit status; repeated imports in separate suite processes are intentional |
| [check_docs.py](../scripts/check_docs.py) | Keep links/entry-point checks; clean-checkout whitespace currently misses committed changes; anchors/external links are not verified |
| [validate.ps1](../scripts/validate.ps1) | Keep explicit isolated/local GPU tiers; avoid duplicate same-job UI build and test full-wrapper failure/environment behavior |
| [setup-dev.ps1](../scripts/setup-dev.ps1) | Keep isolated/pinned CPU setup, pip consistency, and redirection refusal; do not install into production environment |
| [pyproject.toml](../pyproject.toml) | Keep scoped lint/format; its explicit list covers 12 files and is not repository-wide lint coverage |
| [validate.yml](../.github/workflows/validate.yml) | Keep Windows clean-clone checks; add committed-diff whitespace coverage and sanitized browser failure artifacts |
| `tsc --noEmit` / Vite | Keep both type checking and actual bundling; compiling does not establish UI behavior |

CI runs Offline, Docs, and SyntheticBrowser. Local Browser and GpuSmoke are
separate integration tiers. They must not become automatic production/GPU jobs
merely to increase the test count. CPU checks do not measure semantic relevance,
statewide ANN recall, installed GPU-driver compatibility, Electron packaging,
or every physical monitor/DPI layout. Use appropriate explicit evaluations for
changes involving those areas.

## Findings and recommendations

### 1. Urgent: real query in the live browser check

[verify-ui.mjs:80](../frontend/tests/verify-ui.mjs#L80) observes `status.busy`, then
sends a real authenticated `POST /api/search` expecting rejection. Ownership can
change between the observation and request; an accepted request can start real
GPU inference and write user search state. This violates the browser-test scope.
Remove that mutating probe and exercise rejection in deterministic isolated
service tests. Another status poll does not eliminate the race. Until corrected,
do not use this live check while an embedding/query owner can finish during it.

### 2. High: partial ANN behavior is not established

[test_search_index.py:24](../tests/test_search_index.py#L24) creates 600 vectors;
the builder uses two partitions. Production effort probes start at 64 and
candidate floors at 4,096, so default fixture searches probe every partition and
rescore every vector. Exact-result equality usefully verifies original-vector
rescoring but does not prove approximate candidate omission or recall.

The effort test at line 168 lowers candidate floors only, leaving full partition
probing. Add a native case with `nprobe < nlist` and a limited candidate budget,
assert returned scores against original vectors, and separately use controlled
native candidate IDs to verify omission/filtering/ties deterministically. Do
not require approximate retrieval to equal exact results when it omits a true
neighbor. Keep real-query recall benchmarking as its separate quality baseline.

### 3. High: missing adversarial saved-result and base-only coverage

[test_gapfill_display.py:19](../frontend/tests/test_gapfill_display.py#L19) tests
saved CLI imports mainly through a happy path with bundle/metadata readers
mocked. Add real small-bundle hydration and rejections for stale fingerprints,
unknown datasets, unpublished shards, wrong source/hash/window/chip identity,
missing rows, and mismatched bundle/catalog metadata.

Production `Application.saved()` checks the base run fingerprint, with no
adversarial tests for recorded supplement snapshots. The development app
overrides this method, so its history smoke does not validate the production
reader. Define and test the intended saved-result compatibility contract.

Existing base-only display tests use a healthy supplement and do not send a
base-only query with missing/corrupt supplement artifacts. Combined inventory
and status are inspected before base-only selection in several production
preflight paths. Add independent base-only display/search cases; review these
paths for unwanted supplement dependency. This is a likely runtime issue from
static review, not a reproduced product failure in this audit.

### 4. Medium: spatial/RGB/mask assertions need independent answers

- [Grid tests](../tests/test_tiling_embedding.py#L48) compare related helpers and
  coverage but omit the pinned stride offsets. A changed stride can still pass.
  Assert known offsets, such as length 1,000 -> `[0,192,384,576,744]`, plus the
  corresponding two-dimensional windows.
- [Geometry tests](../tests/test_tiling_embedding.py#L69) check valid polygons,
  internally consistent bounds, and point count without independently known
  longitude/latitude. Add numeric expected coordinates in all three UTM zones
  and native window transforms. Axis swaps can otherwise be self-consistent.
  Keep exact vertex count if it intentionally characterizes the frozen output;
  it should not substitute for spatial accuracy assertions.
- [Downloader compression tests](../tests/test_download_nys_naip.py#L345) verify
  masks/georeferencing/three bands without output RGB pixel assertions.
  Supplement compression uses the same value in all source bands. Use distinct
  RGB/NIR constants and appropriate JPEG tolerance so reordering/NIR substitution
  is detectable.
- Current valid-mask cases are well away from 95%. For 256-square chips, test
  the nearest accepted/rejected counts (62,260 / 62,259 valid pixels) and ensure
  vector/metadata rows remain aligned when neighboring chips are excluded.

### 5. Medium: some names assert more than their tests prove

- The "resume by hash" downloader case validates records but does not actually
  rerun processing. Repeat `process_tile`/the scheduler, forbid network/encoding,
  and verify complete outputs stay byte-identical.
- "Checksum rejected before native load" should spy on `faiss.read_index` and
  assert it was never called. Add corrupt-index behavior under automatic search,
  malformed mappings, and native layout mismatch.
- The "without requiring native library" fallback case is inside a class gated
  on those dependencies. Move dependency-independent fallback coverage outside
  that gate and simulate import failure; retain the required full CI suite.
- The "existing owned worker" forwarding case actually creates a mocked new
  one-shot process. Rename it, and assert qualified IDs and supplement opt-out
  survive request publication.

### 6. Medium: incidental process detail should not be mandatory

[test_model_residency.py:127](../frontend/tests/test_model_residency.py#L127)
requires a Windows worker PID to differ from the launcher PID. That is a venv
redirector detail rather than the required launch-ID/liveness contract. Keep the
real child, command exchange, successful unload, and exit assertions; allow both
PID layouts. The separate mock case already exercises differing-PID handling.

Current real child integration covers unload, while query publication/finishing
is largely mocked or in-process. Add a CPU-only child with a controlled adapter
covering request input, atomic result/status, failure, process exit, and a next
serialized query. It must not load GeoRSCLIP or touch production state.

### 7. Medium: cache/preservation checks can pass without proving the claim

- [Gap cache reuse](../frontend/tests/test_gapfill_display.py#L99) compares only
  cache byte count. Regenerating an identical PNG passes. Forbid original-source
  opening during the second call or assert source-read counts/cache hits.
- [Source preservation](../frontend/tests/test_gapfill_display.py#L96) checks
  only the TIFF's first four bytes. Compare the complete before/after hash.
- The [replacement failure case](../frontend/tests/test_service.py#L188) has a
  roomy cache. Add a near-cap failure: eviction happens before replacement, so
  the existing target can be removed before publication fails. Define and test
  which preservation/accounting guarantees apply.
- Add stale-generation writes after clear, event-controlled in-flight clear,
  restart reconstruction/LRU and abandoned-stage handling, concurrent writers,
  and a separate-process frontend cache owner.

### 8. Medium: publication and large-row boundaries are missing

Current ranking/bundle fixtures do not cross the 4,096-row block/row-group
boundary. Add ties and selected rows spanning that boundary. Test actual failed
bundle publication and failed index rebuild (budget/I/O/snapshot change), asserting
previous complete bundles and published index manifests survive. Manually
creating an abandoned staging directory does not exercise failure during writes.

### 9. Medium: local browser checks can test stale packaged resources

[verify-ui.mjs](../frontend/tests/verify-ui.mjs#L57) can borrow a packaged desktop
service and navigate directly to its packaged UI. The current checkout build
may never be exercised. The zoom program already proxies current UI assets;
use that approach for UI validation and verify/report borrowed-service provenance.
A borrowed old backend cannot establish that current service changes work.

### 10. Medium: reports and CI artifacts need to reflect actual execution

The live UI report always lists busy rejection even when its conditional branch
did not run, and claims retention whose substantive assertions are in the zoom
program. Populate report entries from executed assertions, recording conditional
checks accurately.

Fixed report filenames can leave an earlier `passed:true` after a failed rerun.
Use per-run/revision directories or invalidate old reports at startup. Synthetic
smoke writes its useful screenshot/report only on success and CI does not upload
them. Capture bounded failure diagnostics and publish selected synthetic reports
and screenshots on failure too. Sanitize tokens from exception/navigation URLs;
never upload ready files, profiles, user state, raw tokenized logs, or real imagery
from local checks without the applicable authorization/sharing review.

### 11. Medium: clean CI checkout does not verify committed whitespace

[check_docs.py](../scripts/check_docs.py#L98) calls working-tree and staged
`git diff --check`. In a clean Actions checkout both can be empty even when the
pushed commit introduced whitespace problems. Add an explicit fetched base/head
comparison for CI, keeping existing local diff checks. Documentation anchors,
external links, and command behavior remain outside the current checker; label
that scope honestly rather than treating it as complete documentation validation.

### 12. Medium: concurrency and browser waits need bounds

Downloader barriers have no timeout; a worker failure can hang the test. One
peak-concurrency assertion relies on a 25 ms sleep. Use bounded barriers/events
to establish overlap. Some child-process calls lack timeouts. Browser fetch,
WebSocket connection, and shutdown waits are also incompletely bounded, and
some spawn failures lack error listeners. Add clear prerequisite failures,
bounded cleanup, and actual exit acknowledgement for owned processes. Preserve
the rule that borrowed services are never stopped.

### 13. Medium: portable UI flows are mostly outside CI

The synthetic smoke covers one happy search/selection/history/mock-model path.
Search quality mapping, base-only toggles, navigation/clusters, cleanup
confirmation, error/empty-result recovery, model transitions/unload, responsive
layout, and delayed/rapid zoom remain mostly in local live checks or uncovered.
Port portable flows to synthetic CI. Use fixture pixels and visible coverage
assertions for zoom; loaded-image dimensions alone do not prove visible imagery.
Human screenshot inspection is still appropriate for visual changes.

### 14. Low: remove redundant build execution, preserve isolated fixtures

Offline and SyntheticBrowser both compile/bundle unchanged UI in the same job.
Use a checked already-built option or invoke the browser program directly after
the successful CI build. Standalone local browser validation should continue to
build by default. Independent ANN fixtures are valuable because corruption tests
mutate artifacts; sharing one mutable index between tests would create ordering
dependence. Push and PR checks may exercise different branch/merge snapshots and
should not be assumed redundant simply because both run.

### 15. Additional targeted coverage

- Doctor: actual held-lock owner, full valid/stale/corrupt index observations,
  optional checksum, strict/normal CLI exit codes, unchanged observed files.
- Full validation wrapper: early-stage failures halt later stages; temporary
  environment values restore even when a subprocess fails.
- Query adapter: corrupt cached vector fingerprint/shape/nonfinite values/norm;
  invalid typed query fields and boundaries; resident stdin/early-exit/timeouts.
- HTTP: malformed preview windows, bookmark inputs, cache-clear confirmation
  during overview, and exhausted handler capacity.
- History/worker: result/vector retention limits, actual redirected paths, and
  competing owners during cleanup; retain the existing adversarial unit cases.
- Download/supplement: truncated advertised transfer, mid-stream stop, rejected
  provider/provenance, retry exhaustion/nonretryable errors, compatibility fields.

## Suggested order

1. Remove the unsafe live-query probe and correct report/code-provenance claims.
2. Strengthen partial ANN, spatial/RGB/threshold, saved-result, and base-only tests.
3. Cover cache/publication/chunk boundaries and bounded process behavior.
4. Expand synthetic UI CI, preserve useful diagnostics, and verify pushed diffs.
5. Remove the repeated CI build after establishing an explicit fresh-build guard.

These are recommendations for separately scoped changes. No product or frozen
pipeline refactor is needed merely to delete apparent test duplication. Follow
[AGENTS.md](../AGENTS.md) for exact-commit CI waiting and authorized Git operations.
