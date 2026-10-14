# Project improvements

A running list of known flaws, limitations, and areas for improvement across the
project. Add future findings here with concise reasoning, supporting evidence,
and a proposed improvement. These entries are recommendations, not authorization
to change the pipeline.

## Imagery acquisition and embedding (2026-10-14)

Reviewed from repository code and operating guides. No new imagery or GPU
experiment was run.

**Recommended order for future acquisition: download -> embed -> compress.**
Validate each downloaded TIFF, embed its original downloaded RGB windows,
publish and verify the embedding bundle, then compress the imagery for storage
and display before deleting the temporary source. This preserves more input
information for inference than download -> compress -> embed. It does not
require retaining the entire original statewide dataset: process a bounded
number of TIFFs at a time within the shared budget. The expected benefit is
avoiding additional pixel loss; improved search accuracy remains to be measured.

### Should embeddings have come before lossy compression?

**Preferably, for preserving the downloaded RGB signal.** The downloader reads
provider TIFFs, writes JPEG/YCBCR RGB at quality 83 (falling to 80, 77, or 75 to
fit a size ceiling), then deletes the temporary provider file. Embeddings read
the resulting compressed TIFFs. Native 0.6 m spacing and masks are preserved,
but pixel values are not. JPEG/YCBCR also reduces color detail through chroma
subsampling; see [GDAL's GeoTIFF documentation](https://gdal.org/en/stable/drivers/raster/gtiff.html).
Encoding before this local step avoids that additional information loss; it
cannot reverse any compression already present in the provider imagery.

**This does not establish that existing embeddings are inadequate.** Search
impact is unmeasured, and compression made statewide storage practical under
the 300 GB cap. Model preprocessing also resizes chips to 224 pixels. Compare
representative source/compressed chip pairs before considering re-embedding.

### Should embedding have overlapped downloading?

**Potentially, for earlier results and shorter total elapsed time.** Once a
complete source TIFF is downloaded and validated, a bounded GPU consumer could
embed its RGB windows before the source is deleted, while other TIFFs download.
Embedding compressed tiles as they arrive would overlap work but would not
avoid compression loss. Speedup depends on network, CPU, disk, and GPU contention.

The current pipeline requires `completion.json` and all 3,805 base tiles before
embedding. Running both existing launchers together is not a supported shortcut;
safe overlap needs explicit orchestration, source retention, and shared budgets.

### Known limitations and areas for improvement

| Limitation or evidence gap | Improvement and reasoning |
| --- | --- |
| Local lossy compression precedes inference; search degradation has not been measured. | Test identical windows and masks before/after compression at each supported quality. Measure embedding similarity, ranked-result changes, and human-labeled relevance, especially for small objects. Pixel error alone cannot establish search usefulness. |
| JPEG quality is chosen by file size rather than semantic fidelity. | Use the comparison to set a justified quality floor or selective lossless policy. Retrying qualities already reads the original source each time, so retries do not compound JPEG generations. |
| Downloaded source bytes are discarded after compression. | For future acquisition, retain each source until its embedding bundle is durable; optionally keep a budgeted representative source sample. Hashes and URLs preserve provenance but cannot reconstruct discarded pixels, and later model/grid experiments otherwise need another download. |
| Acquisition and embedding are separate, completion-gated stages. | Add a bounded per-source ready queue, backpressure, crash recovery, and cross-process storage reservations. Keep one GPU owner, reserve temporary/vector space, and finalize a stable collection snapshot before publishing its index. |
| RGB conversion discards NIR. | Keep RGB for the pinned model's contract; evaluate separate multispectral storage/model support only for tasks that need it. Retaining NIR alone would not improve this RGB encoder. |
| One chip scale, masked-edge exclusions, and overlapping TIFF buffers limit object coverage and distinct results. | Evaluate additional scales, edge handling, and geographic result deduplication against labeled examples. Whole-chip similarity is not object detection; changing scale or mask policy requires a separate compatible collection. |
| Existing ANN recall measures agreement with exact vector ranking. | Add human-labeled relevance and object-level coverage evaluation. High ANN recall cannot prove that compression, the embedding model, or the chip grid finds the intended objects. |

For the completed collection, preserve immutable imagery, provenance, vectors,
and the exact-search baseline. Prioritize a small controlled comparison over
a statewide rerun; any future source-based embeddings need separate recipes
recording the actual inference-source hash and its compressed display derivative.

Evidence: [downloader](src/geospatial_search/download_nys_naip.py),
[inventory gate](src/geospatial_search/tiling.py),
[embedding recipes](src/geospatial_search/embed_georsclip.py),
[acquisition guide](docs/nys-naip-download.md),
[pipeline guide](docs/nys-pipeline.md), and
[search-index benchmarks](docs/nys-search-index.md).

## Ground-truth retrieval benchmarks (2026-10-14)

Evaluate retrieval against imagery with instance annotations before expanding
country-wide acquisition. [RarePlanes](https://arxiv.org/html/2006.02963) is a
strong aircraft benchmark: approximately 14,700 annotated aircraft appearances
in real 0.31-0.39 m RGB imagery, with geographic polygons paired to each image.
It is [free to download under CC BY-SA 4.0](https://registry.opendata.aws/rareplanes/).
Use full-location RGB GeoTIFFs and background areas; the prepared crops all
contain aircraft and would make search evaluation artificially easy. Preserve
image/date identity because aircraft move and locations have repeat observations.
See the [product and annotation guide](https://www.iqt.org/library/rareplanes-public-user-guide).

Measure distinct-object recall at K returned chips, relevant-chip precision at
K, and recall versus inspected chips/area and elapsed time. Count an aircraft
once per image despite overlapping results, define visibility and evaluation
boundaries, and report mask/grid coverage losses separately from ranking losses.
Compare ANN with exact scoring to isolate index approximation from semantic
retrieval quality; chip retrieval does not establish object localization accuracy.

For broader classes, consider [xView](https://arxiv.org/html/1802.07856) (0.3 m;
[CC BY-NC-SA 4.0](https://challenge.xviewdataset.org/rules)). For a European
building benchmark, [SpaceNet 6 Rotterdam](https://spacenet.ai/rotterdam/) provides
0.5 m, 8-bit RGB GeoTIFFs and footprint labels under
[CC BY-SA 4.0](https://spacenet.ai/sn6-challenge/); hold out labeled RGB training
areas because its public test set is SAR-only. These are proposals: ingestion
needs separate dataset adapters/recipes for the current NYS/0.6 m assumptions;
no benchmark data has been acquired or evaluation implemented.
