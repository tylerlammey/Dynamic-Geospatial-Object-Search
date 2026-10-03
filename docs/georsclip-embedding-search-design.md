# GeoRSCLIP Embedding and Geospatial Search Design

**Status:** proposed
**Scope:** 2024 Albany four-band orthophotography today; multiple cities, years,
and large-scale similarity/text search later.

## 1. Decision summary

Treat a *georeferenced image chip*, rather than an original JP2 tile, as the
searchable unit. Keep the existing JP2 files as immutable source imagery. Create
a canonical chip manifest with each chip's source-pixel window and geographic
footprint, then attach one normalized GeoRSCLIP embedding per chip.

Use this stack initially:

| Concern | Recommended system |
| --- | --- |
| Raster reading/chipping | Python + Rasterio/GDAL, reading JP2 windows directly |
| Embedding generation | Python + PyTorch/OpenCLIP on CUDA |
| Canonical spatial metadata | GeoParquet, partitioned by dataset/year/city/source tile |
| Embedding storage | Append-only, sharded `float16` memory-mappable arrays |
| Nearest-neighbor index (first release) | FAISS, with an exact GPU validation index and HNSW CPU index |
| Very-large, read-mostly production index | NVIDIA cuVS CAGRA on GPU, or FAISS GPU IVF-PQ when GPU memory is constrained |
| Query service | Python prototype; C++ FAISS/cuVS service only after the index profile is known |

Python is the appropriate first implementation language. The expensive work is
already in GDAL, PyTorch/CUDA, FAISS, and cuVS native code; replacing pipeline
orchestration with C++ will not materially improve batch embedding throughput.
C++ becomes worthwhile for a high-QPS, low-latency ANN service or if the Python
web/API layer becomes measurable bottleneck.

**Current hardware note:** the RX 9070 XT runs the PyTorch embedding job through
AMD ROCm, but it cannot use CUDA-only FAISS GPU builds or NVIDIA cuVS CAGRA. Use
exact CPU search or CPU FAISS HNSW for Albany and benchmark a ROCm-compatible
ANN engine only when the corpus becomes large enough to justify it.

## 2. What the current imagery means for this design

The Albany data is 158 JPEG 2000 (`.jp2`) aerial-orthophoto tiles. A source tile
is 6,000 x 4,000 pixels at 0.5 US survey foot/pixel (about 15 cm/pixel), and is
registered in NAD83(2011) / New York East State Plane, US survey feet. It has
four bands. The provider specifies bands 1, 2, and 3 for natural-color display
and bands 4, 1, and 2 for color infrared.

GeoRSCLIP ViT-H/14 is a three-channel CLIP model. Do **not** pass all four bands
to it or silently substitute NIR for a color channel: that changes the trained
input distribution. The baseline should use provider bands 1/2/3 as RGB and
preserve the original four-band JP2 for later use. NIR can support a separate
vegetation-oriented feature or a future properly trained four-channel model.

The selected Hugging Face redistribution, `BiliSakura/GeoRSCLIP-ViT-H-14`,
provides an OpenCLIP-compatible `GeoRSCLIP-ViT-H-14.pt` checkpoint. The current
implementation downloads that checkpoint once into the persistent Hugging Face
cache, creates the OpenCLIP `ViT-H-14` preprocessing transform, and records the
checkpoint SHA-256 plus a preprocessing fingerprint in `run.json`. The model
revision is not currently pinned; a production workflow should add an explicit
revision argument and record it alongside the checkpoint hash. The expected
ViT-H/14 projection width is normally 1,024, but the implementation records the
actual output dimension instead of hard-coding it.

## 3. The searchable record: one chip, one embedding

### 3.1 Why not one vector per original tile?

One 3,000 x 2,000 foot tile contains many unrelated places. Its one global
embedding would be vague: a query for a baseball field, rooftop solar, parking
lot, or dense houses would all return the same tile-level result. It also cannot
pinpoint a result on the map.

### 3.2 Baseline chip grid

Start with square 512 x 512 **source-pixel** chips, a 256-pixel stride (50%
overlap), and resize each chip to the model's 224 x 224 input. Each chip covers
256 x 256 feet. This produces approximately 345 chips per source tile and about
54,500 chips for the present Albany collection. It is a manageable baseline
that provides location specificity without creating an enormous corpus.

Do not use the 512 x 512 PNG preview as input; it is merely a visual reference
and has black padding. Read the native JP2 pixels instead.

This initial grid is a retrieval-quality experiment, not a permanent truth.
Evaluate it against a hand-labelled set of queries. If targets such as cars,
small roofs, or individual houses matter, add a smaller 224 x 224 pixel grid at
112-pixel stride. That can produce roughly 293,000 chips in Albany, so it should
be adopted only if the evaluation shows it earns its additional compute and
index size. Keep `scale_id` in every record so multiple grids can coexist.

### 3.3 Chip schema

Use a monotonically assigned `embedding_id` (`uint64`) as the vector index ID.
Use a stable, human-debuggable `chip_id` derived from immutable identity:

```text
albany_sp24 / <source_tile_stem> / s512_st256 / x0000256_y0000512
```

The canonical GeoParquet row should include at least:

| Field | Purpose |
| --- | --- |
| `embedding_id` (`uint64`) | Foreign key to the vector shard and ANN index |
| `chip_id`, `dataset_id`, `city`, `acquisition_year` | Stable identity and filtering |
| `source_uri`, `source_sha256`, `source_tile_id` | Reproducibility and source lookup |
| `pixel_x`, `pixel_y`, `pixel_width`, `pixel_height` | Exact native raster window |
| `geometry` (polygon), `bbox_*`, `centroid_lon`, `centroid_lat` | Map rendering, spatial predicates, and fast display |
| `crs_wkt` / EPSG when available | Prevents coordinate ambiguity |
| `scale_id`, `rgb_band_order`, `resize_method` | Image-construction contract |
| `model_id`, `model_revision`, `model_fingerprint`, `preprocess_fingerprint`, `embedding_dim` | Embedding compatibility contract |
| `embedding_shard`, `embedding_row` | Location in the append-only vector store |
| `created_at`, `pipeline_version`, `status`, `error` | Restartability and audit trail |

Store geometry in the source CRS for exact spatial work and also materialize the
WGS84 centroid/bounding box for web maps. Never store bare `x`/`y` values without
their CRS. Preserve the input JP2/world file and the source spatial reference as
the authoritative location record.

## 4. Data layout

```text
imagery/                         # source JP2 + sidecars; never modified
data/
  manifests/
    dataset=albany_sp24/year=2024/city=albany/part-*.parquet
  embeddings/
    model=georsclip-vith14-<model-fingerprint>/
      vectors-00000.f16          # row-major, normalized vectors
      vectors-00000.ids.u64
      vectors-00001.f16
      manifest.json
  indices/
    model=georsclip-vith14-<model-fingerprint>/
      faiss-hnsw-*.index
      or-cagra-*.index
  evaluations/
    retrieval-set-v1.parquet
    results-*.json
```

Do not make individual PNG/JPEG chips the primary data structure. That duplicates
imagery, creates millions of files at scale, and divorces the chip from its
geospatial transform. Generate them only for debugging, labeling, or an
optional web tile cache. The manifest's source raster plus pixel window is the
canonical chip definition.

The vector shard contains only dense numeric data; it is fast to memory-map and
simple to rebuild. GeoParquet holds rich metadata; it is queryable with DuckDB,
GeoPandas, Spark, or later imported into PostGIS. Keeping the two separate
avoids database-row overhead in the hot vector-search path.

## 5. Embedding pipeline

1. **Inventory and validate.** Read the image catalog/index, compute a source
   checksum, validate CRS/band count/nodata, and assign a `dataset_id`.
2. **Plan chips.** Generate only windows fully inside the source raster (or
   explicitly record padded edge chips). Compute each footprint using the
   raster's affine transform; do not infer it from the filename.
3. **Read and compose RGB.** Read the three natural-color bands from the JP2
   window through GDAL/Rasterio. Convert to the model's expected RGB ordering
   and deterministic 8-bit/float range. Record the policy for any nodata or
   very-dark border pixels.
4. **Preprocess exactly once.** Apply the selected model's documented resize,
   center crop, interpolation, and normalization. Pin the code/dependency
   versions and serialize a preprocessing fingerprint. A different crop or
   normalization makes vectors incompatible.
5. **Batch inference.** Use `model.eval()`, `torch.inference_mode()`, CUDA
   autocast (`float16` or `bfloat16` after validation), pinned-memory workers,
   and batches sized from GPU memory. JP2 decoding can become the limiter, so
   profile decode, host-to-device copy, and model time separately.
6. **Normalize and persist.** L2-normalize each image embedding in `float32`.
   Validate finite values and unit norm, then store the normalized result as
   `float16` with its ID. Cosine similarity is then a dot product.
7. **Index only completed rows.** Build/append the ANN index from the exact
   vector IDs successfully written. Failed chips remain visible in the manifest
   but never enter the index.
8. **Write progress atomically.** Commit shards atomically, keep a run ledger,
   and make reruns idempotent using `chip_id + model/revision/preprocess fingerprint`.

### Text query path

Encode a user query with the *same* GeoRSCLIP model revision's text encoder, then
L2-normalize it and search by inner product. Use a small, versioned prompt
template set during evaluation, for example `"an aerial image of {query}"`,
`"a satellite image of {query}"`, and an unadorned query. Average and normalize
the resulting text vectors only if the evaluation improves recall. Image and
text vectors must always come from the same model revision/version.

## 6. Approximate-nearest-neighbor design

### 6.1 Metric and correctness

Search `IndexFlatIP` over normalized vectors as the ground-truth evaluator:
inner product equals cosine similarity. Use it on GPU when the corpus fits. It
is also the benchmark against which every approximate index is measured.

The product must report retrieval quality (Recall@10/50 and query latency), not
just that an ANN index returns results. Build a labelled evaluation set that
includes text prompts, example image chips, expected spatial areas, and hard
negatives such as roads versus parking lots.

### 6.2 Index selection

| Corpus / update pattern | Index | Reason |
| --- | --- | --- |
| Up to roughly 1M vectors; additions expected | FAISS `IndexHNSWFlat` (CPU) | High recall, low operational complexity, supports incremental addition |
| Up to GPU memory; low latency is paramount | GPU exact `IndexFlatIP` | Simplest and highest-recall baseline; no ANN approximation |
| Tens to hundreds of millions, mostly static, NVIDIA GPU | cuVS CAGRA | GPU graph ANN intended for very low-latency, high-throughput search |
| Very large corpus where vector RAM is expensive | FAISS GPU IVF-PQ | Strong compression; tune `nlist`, `nprobe`, and PQ code size for recall/latency |

For this first Albany dataset, use exact GPU search if a CUDA GPU is available;
otherwise HNSW. Do **not** introduce a distributed vector database yet. At about
54,500 baseline vectors, even uncompressed vectors are small. The data model,
model fingerprints, and evaluation harness are the important investments.

At a common 1,024-dimensional output size, one normalized `float16` vector is
2 KiB: approximately 106 MiB for the 54,500-chip baseline and 572 MiB for the
293,000-chip small-chip variant. A 64-byte PQ code would reduce raw vector-code
storage substantially, at the cost of some recall. Measure using the actual
model output dimension before capacity planning.

### 6.3 Metadata and spatial filters without slowing ANN

The ANN engine should return integer IDs only. Fetch result rows in one batch
from a memory-mapped ID-to-metadata table or an embedded database, then load the
required JP2 window for display. This avoids a database call per neighbor.

For global similarity search, keep one index per `model_fingerprint` and
optionally per acquisition collection/year. For mandatory city, time, or
geographic-area filters, route to a pre-filtered index partition (for example,
city/year and coarse H3 cell) and merge top-k candidates. Post-filtering a
single global top-k can return too few valid results. Use PostGIS/GeoParquet for
precise map intersection after candidate retrieval; it is not the vector engine.

## 7. Runtime architecture

```text
JP2 + world-file / embedded CRS
          |
   Rasterio/GDAL window reader
          |
   deterministic RGB chip + geographic footprint ----> GeoParquet manifest
          |
   GeoRSCLIP ViT-H/14 image encoder (batched CUDA)
          |
   normalized f16 vector shard + ID map
          |
   FAISS / cuVS ANN index
          |
text or image query -> same encoder -> top-k IDs -> batched metadata lookup
                                                    -> map footprints + previews
```

Keep the ingestion and model code in Python. Both PyTorch and rasterio execute
their heavy work in optimized native/CUDA libraries. Add a C++ search service
only if profiling demonstrates a need, with this boundary:

* **Python:** ingestion, model loading/inference, experiments, batch jobs, and
  the initial API.
* **C++:** optional long-running FAISS/cuVS process exposing `search(vector,
  filters, k)` and returning IDs/scores; it memory-maps the ID map and loads the
  prebuilt index once.
* **No C++ model rewrite initially:** LibTorch would raise deployment and model
  compatibility costs while retaining essentially the same CUDA kernels.

## 8. Operational requirements

* Version datasets, source checksums, chip grid, RGB policy, model revision,
  preprocessing, embedding dimension, and index parameters together. Never mix
  embeddings across any of these boundaries.
* Keep the model cache locally and record the checkpoint hash and preprocessing
  fingerprint. Add and record an explicit model revision before production. The
  available model identifies itself as a redistribution; record its license/terms
  before a production deployment.
* Keep vector shards and ANN indexes rebuildable derived artifacts. Back up the
  source imagery and GeoParquet manifests first.
* Emit metrics: tiles/chips attempted and completed, decode ms/chip, inference
  ms/chip, vectors/sec, GPU utilization, invalid-vector count, index build
  duration, p50/p95 query latency, and Recall@k against exact search.
* Respect imagery licensing and avoid exposing source imagery or coordinates
  outside its permitted use.

## 9. Phased implementation plan

### Phase 0 — feasibility and evaluation (one tile)

Implement a reader for the first JP2; produce its 512/256 chip grid; load the
selected ViT-H/14 model revision; generate normalized vectors; and run exact search
for 20–50 representative text/image queries. Verify RGB ordering, chip
footprints, and prompt behavior visually.

**Exit criterion:** a reviewer can click every result and see the exact
corresponding map footprint/source window; vectors are reproducible in a rerun.

### Phase 1 — Albany baseline

Process all 158 tiles, write the manifest/vector shards, build exact GPU or HNSW
search, and create a small evaluation dataset. Test 512/256 against at least one
smaller-chip alternative before committing to the corpus size.

**Exit criterion:** measured recall/latency report, restartable processing, and
top-k map result view.

### Phase 2 — scalable retrieval

Add model-versioned index builds, filters/partition routing, image-query search,
and a simple API. Profile real load. Move static large collections to CAGRA or
IVF-PQ only when the corpus or latency data justifies it.

### Phase 3 — multi-city/time and domain adaptation

Add collection/year partitions and cross-temporal chip associations. Consider
NIR-aware learning or fine-tuning only after a labelled target task exists; it
requires an architecture/training change, so it is a new embedding space and a
new index, not a drop-in fourth channel.

## 10. Questions to resolve before implementation

1. What types and smallest real-world size of objects must be retrievable?
   This determines chip size and overlap more than any indexing choice.
2. Is search text-only, image-to-image, or both? GeoRSCLIP supports both, but
   each should be evaluated separately.
3. What hardware and target p95 latency/QPS are available? This decides whether
   GPU exact search, HNSW, CAGRA, or IVF-PQ is appropriate.
4. Are results allowed to span years/cities by default, and which geographic or
   time filters are mandatory? That determines index partitions.
5. Is the outcome semantic area retrieval or true object detection? Embeddings
   are excellent for semantic retrieval but do not replace an object detector
   when pixel-accurate object boxes are required.

## References

* GeoRSCLIP model card and feature-extraction example: <https://huggingface.co/BiliSakura/GeoRSCLIP-ViT-H-14>
* Selected GeoRSCLIP model and feature-extraction usage: <https://huggingface.co/BiliSakura/GeoRSCLIP-ViT-H-14>
* RS5M/GeoRSCLIP project and paper links: <https://github.com/om-ai-lab/RS5M>
