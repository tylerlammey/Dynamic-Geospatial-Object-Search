# NYS vector search index

The default retrieval path uses **CPU Faiss IVF-PQ followed by exact rescoring
of the original vectors**. This is a derived index of the existing GeoRSCLIP
collection. It does not change the embedding model, preprocessing, chip grid,
original float16 vectors, GeoParquet, or imagery.

IVF selects promising groups of vectors. PQ stores short codes for candidate
selection. The returned candidate rows are then read from the original shards
and scored with their float32 dot products, using the same score convention as
the previous exact search. Candidate selection is approximate: rescoring gives
accurate scores for candidates, but does not guarantee every global neighbor
was found. Exact mode remains available for verification.

## Why this index

The completed NYS collection has **11,985,388 vectors, 1,024 dimensions, and
3,805 source shards**. Its original float16 vectors occupy 24.55 GB. This Windows
machine has about 32 GB RAM, an AMD RX 9070 XT, and roughly 17 GB available RAM
before searching. A full float32 HNSW index needs roughly 50–52 GB just for
vectors and graph edges. CPU IVF-PQ avoids a CUDA-only dependency and leaves
room for text encoding and the imagery frontend.

The published index uses **4,096 IVF partitions, 128 8-bit PQ subquantizers,
327,680 uniformly sampled training vectors, and eight CPU threads**. Training
uses seed 20261012 and 15 iterations. Its coarse quantizer uses inner product,
as do candidate search and original-vector rescoring. It occupies **1.65 GB**
and took **13 minutes 39 seconds** to validate, train, populate, and publish.
Training and publication run in bounded batches; publication accounts for its
temporary index file under the existing 300 GB job budget.

We compared the original exact scan, Faiss IVF-PQ128, Faiss IVF-SQ8, USearch
HNSW with int8 vectors, and LanceDB IVF-HNSW-SQ on a reproducible real-vector
sample. Full-corpus verification then checked the selected implementation
against the original exact search using 12 real text queries.

The small sample initially suggested a narrow rescoring pool was sufficient.
Full-state testing disproved that: it recovered only 77% of the exact top 100
`fighter jet` matches. The implementation therefore uses substantially larger
candidate pools. This is why the default was selected from full-corpus recall,
rather than the fastest small-sample timing alone.

Detailed local references and raw measurements stay under the ignored job directory at
`verification/search-selection/`; no vectors, imagery, or location records are
published with the code.

## Algorithm comparison

The screening sample contains 121,760 real vectors: 32 seeded random rows per
source TIFF. All algorithms used the same 12 compatible text vectors, requested
500 results, returned up to 4,000 candidates, and rescored those candidates
against the original sample vectors. IVF used 512 partitions and 128 probes;
HNSW used connectivity 16 and an expansion/candidate budget of 4,000. LanceDB
also used its scalar-quantized HNSW graph inside IVF partitions.

| Tested algorithm | Sample median retrieval | Mean recall@100 | Statewide index storage |
| --- | ---: | ---: | --- |
| Previous exact scan | 282.70 ms | 100% | Existing 24.55 GB vectors; no extra index |
| Faiss IVF-PQ128 | 6.07 ms | 98.08% | **1.65 GB measured** |
| Faiss IVF-SQ8 | 25.59 ms | 98.75% | Approximately 12.4 GB, estimated |
| USearch HNSW int8 | 11.88 ms | 97.25% | At least 12.3 GB for vectors, plus graph |
| LanceDB IVF-HNSW-SQ | 14.89 ms | 98.08% | Larger scalar/graph index plus copied database |

These are configured implementations, not a claim that every possible HNSW or
LanceDB configuration is slower. Their full-state latencies were not measured.
LanceDB would introduce a second database representation; uncompressed float32
vectors alone would occupy 49.1 GB. IVF-PQ was the fastest tested candidate
retriever with the smallest derived index. We then tuned and verified its
recall on the entire collection, retaining original shards and the exact
fallback. Sample timings do not predict full-state file I/O or fresh model
startup time.

## Statewide measurements

Measured on this Windows workstation on October 12, 2026, using the completed
11.99-million-vector collection. Three independent original full scans took
95.57, 143.21, and 101.61 seconds. Exact top-500 references for all 12 phrases
used the original ranking helpers; the remaining reference queries shared
source reads and validation. Each indexed query searched the same collection
and was checked against those references.

| Mode | Median retrieval | p95 retrieval | Mean recall@20 | Mean recall@100 | Mean recall@500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original exact scan | **101.61 s** | Three measured scans | 100% | 100% | 100% |
| Fast | 1.00 s | 3.52 s | 97.50% | 96.17% | 92.75% |
| **Balanced** | **6.28 s** | 16.80 s | **100%** | **99.75%** | 99.00% |
| Thorough | 10.51 s | 20.35 s | **100%** | **100%** | **99.78%** |

Balanced median retrieval is approximately **16.2 times faster** than the
original median; Thorough is approximately **9.7 times faster**. These medians
cover different query counts: three independently timed exact scans and twelve
indexed queries. They summarize representative local runs, rather than an
identically repeated latency distribution for each algorithm.

The lowest Balanced recall@100 was 97%, for `swimming pool`; its recall@500
was 88.6%. Thorough raised those to 100% and 97.4%. `fighter jet` recovered all
exact top-100 matches in both modes. Retrieval recall compares ranked chip IDs
with the exact vector scan; it is not object-detection recall or confidence.
Exact remains the option when full-corpus ranking must be guaranteed.

Index loading/checksums and original-bundle stat checks are included in these
retrieval timings. The first Fast query includes a cold process-local index
load; subsequent benchmark queries reuse that index. File-page cache state and
background workload affect latency. Actual frontend query workers load their
own index, and fresh phrases also load/encode with the pinned GPU model.
Metadata/source checksum verification and rendering add time beyond retrieval.
Real authenticated frontend checks with a cached text vector took 7.59 seconds
for five Balanced results and 10.52 seconds for eight additional Thorough
results. These include worker startup, a fresh index load, result metadata,
source verification, and worker shutdown; preview requests were checked
separately against the original pixels.

## Commands and modes

Install search dependencies into the existing environment without replacing
the GPU runtime. The normal embedding setup includes `requirements-search.txt`.
For an already configured environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-search.txt
.\scripts\build-search-index.ps1
.\scripts\build-search-index.ps1 -Status
.\scripts\search-nys.ps1 -Query 'fighter jet' -Offline -TopK 25
.\scripts\search-nys.ps1 -Query 'fighter jet' -Offline -SearchEffort Thorough
.\scripts\search-nys.ps1 -Query 'fighter jet' -Offline -Backend Exact
```

`faiss-cpu==1.15.1` is pinned. It runs on the CPU; GeoRSCLIP text encoding still
uses the existing pinned model and GPU runtime. A compatible existing index is
reused. Builds are foreground operations with percent, elapsed time, ETA, and
a separate ownership lock. No embedding run is started by the index builder.

| Effort | IVF partitions probed | Minimum candidate pool | Intended use |
| --- | ---: | ---: | --- |
| Fast | 64 | 4,096 | Explicit speed/recall tradeoff; lower recall on difficult text queries |
| Balanced | 512 | 131,072 | Default GUI and CLI retrieval |
| Thorough | 4,096 | 524,288 | Scan all compressed partitions and rescore more original candidates |
| Exact backend | All original vectors | All eligible rows | Guaranteed full-corpus ranking under the existing score convention |

These are caps for small collections. The candidate pool grows to at least
eight times the requested count plus excluded count. Probe count also grows
for larger requests. The API accepts 1–5,000 results; the CLI accepts 1–100
because it renders labelled chip PNGs and a contact sheet.

Thorough mode parallelizes the compressed IVF scan across up to eight CPU
threads. Original-vector rescoring uses one BLAS thread for the thousands of
small shard matrix-vector products, then restores the previous setting. This
avoids a measured slowdown from OpenBLAS's default 16-thread configuration.
Resolved source paths are cached, while original file stats are still checked
on every request.

Up to eight bounded readers fetch candidate vectors. If a request covers at
least 0.78125% of a source shard, one sequential read avoids scattered Windows
mapping page faults; sequential source reads are capped at 128 MiB each.
Sparser requests read mapped rows. Each source contributes only its top-k
candidates to the final merge, preserving exact ranking within the candidate
set and bounding temporary results. No full-state vector array is created.

`auto` uses a matching index and falls back to exact search when an index is
missing, stale, or the optional native dependency is unavailable. Explicit
`ann` requires a usable index; `exact` bypasses it. Corrupt native files or
changed original bundles fail verification instead of silently trusting the
index. Result JSON includes the backend, effort, candidate count, probe count,
and retrieval time.

## Frontend and future decision loop

The frontend uses the same retrieval function and exposes Balanced, Thorough,
and Exact. Existing searches, bookmarks, footprints, previews, pagination,
overlapping matches, map browsing, and GPU/collection ownership remain
compatible. A search still produces original `(shard_id, embedding_id)` IDs
and WGS84 geometry; native index row numbers do not escape into map results.

A future decision model can increase `top_k`, change effort, reformulate text,
or exclude already-seen IDs through the existing authenticated local endpoint:

```json
{
  "query": "fighter jet",
  "top_k": 200,
  "search_backend": "auto",
  "search_effort": "thorough",
  "exclude_ids": [["SOURCE_SHARD_ID", 7], ["ANOTHER_SHARD_ID", 7]]
}
```

The POST to `/api/search` returns a task ID. Poll `/api/search-progress?id=TASK_ID`,
then read `/api/results?id=RESULT_ID`, as for existing frontend searches.
Completion is acknowledged after the owned worker exits, so an immediate next
request does not race native process teardown.
Use real qualified IDs from prior results. Exclusions are validated against
the current snapshot; at most 20,000 unique seen IDs are accepted. Return count
means additional unseen matches, rather than counting excluded rows toward the
requested number. Scores remain similarity ranks, not detection confidence.
The decision model itself is deferred.

## Snapshot, storage, and verification

Index files live in the collection's `search-index/` folder, under the existing
NYS job directory. A checksummed native file and atomic `manifest.json` map
global index rows back to immutable source bundles. The manifest includes the
collection fingerprint, ordered shard mapping, file sizes/timestamps, native
checksum, dependency version, and training parameters.

The builder validates original bundle hashes and alignment before indexing,
then rechecks the snapshot and file stats before publication. Query processes
verify the native checksum before loading and recheck original bundle stats on
each search, including cached readers. Newly published shards invalidate the
snapshot, so resumed embedding can automatically use exact search until a new
index is built. Safely pause before querying a partial collection as usual.

Use `-Force` to publish a new generation after a collection changes. Interrupted
builds leave the previous published generation intact; rerun to rebuild from
scratch. Old derived generations are not automatically deleted. Original
embedding bundles and downloader provenance remain immutable.

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
$env:PYTHONPATH = 'src;frontend/service'
.\.venv\Scripts\python.exe -B -m unittest discover -s frontend/tests -v
node .\frontend\tests\verify-ui.mjs --overview
```

The full benchmark can be repeated with a JSON containing `run_fingerprint`
and a `queries` list of `{query, vector}` from the compatible text model:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m geospatial_search.benchmark_search_index `
  --queries 'data/nys-naip-2022-rgb-060m/verification/search-selection/text-queries.json'
```

By default it times three independent original full scans, then obtains the
remaining recall references with the unchanged ranking helpers while sharing
source validation and reads. Use `--timed-exact-queries 12` for independent
timings on all 12 phrases. A synthetic regression verifies shared references
match independent original scans exactly.
Use `--recompute-references` to remeasure the original scans after changing
machines or when collecting a fresh timing baseline.

It preserves exact top-500 reference IDs under the job directory to allow
candidate tuning without repeated full scans. Reference reuse requires an
identical collection snapshot and query payload. Text encoding, source
checksum verification for rendering, and PNG generation are excluded from
retrieval timings; these add time to an actual fresh GUI or CLI request.

Primary references: [Faiss index types and storage](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes),
[Faiss runtime tuning and native-file integrity](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning),
[USearch quantized HNSW](https://github.com/unum-cloud/USearch),
[LanceDB index tradeoffs](https://docs.lancedb.com/indexing/vector-index), and
[scoped BLAS thread controls](https://github.com/joblib/threadpoolctl).
