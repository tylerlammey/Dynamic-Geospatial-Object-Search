# Object detection in desktop search

[LLMDet Large](https://huggingface.co/iSEE-Laboratory/llmdet_large) locates text
queries in ranked aerial image tiles. **Detect objects** is enabled in the
search toolbar. With detection enabled, the result count follows the
detections. Turn it off to request a fixed number of similarity matches.
The command-line GeoRSCLIP search also remains available.

## Set up and run

Complete the [base GPU setup](rx-9070xt-embedding-setup.md) first. From the
repository root, install the tested detector dependencies and cache the
pinned checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-detection.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m geospatial_search.detection_model --download
.\frontend\scripts\launch.ps1
```

The download is about 1.38 GB and does not load a GPU model. The desktop
can then run offline, reusing `model-cache/llmdet/`. On the benchmark
workstation the cache reuses the downloaded snapshot via hard links. Restart
the desktop to use the updated build.

LLMDet loads on the first detection search. Use **Load model to GPU** before
searching to keep the text encoder and, after its first use, the detector
resident across queries. **Unload model** releases both and their existing
collection and job GPU locks.

## Defaults and feedback

| Setting | Default | Purpose |
| --- | ---: | --- |
| Box score | 0.30 | Filter weak boxes before accepting a tile |
| Text threshold | 0.25 | Select tokens for a box label |
| Overlap suppression | 0.50 IoU | Keep the higher score for overlapping or substantially nested duplicates |
| GPU batch size | 2 | Stay within measured RX 9070 XT headroom |
| Fetch size | 32 tiles | Page through ranked candidates |
| Minimum probe | 64 tiles | Check before applying the recent-yield stop, if available |
| Recent window | 32 tiles | Measure how often recent tiles have qualifying boxes |
| Minimum recent yield | 5% | Continue while at least two of the last 32 tiles match |
| Candidate limit | 256 tiles | Bound the total work |

Adjust the box cutoff and tile limit in **Settings**. API callers can
override the other values through `detection` in the
[service contract](CONTRACTS.md). A smaller candidate limit takes precedence
over the minimum probe. A `min_detection_rate` of 0 disables the yield stop
for a bounded deeper scan.

Each evaluated tile with at least one qualifying box can appear, up to all
evaluated tiles. Search returns zero when none qualify. The strongest retained
box confidence orders the results, with GeoRSCLIP similarity breaking ties.
ANN retrieval and the feedback loop still inspect tiles in similarity order.
Similarity-only searches keep their original ranking.

The detector reads the exact published native RGB window and applies its
processor's resize. It clips boxes to the tile and keeps up to 50 after
suppressing duplicates. It suppresses a lower-scored box when IoU exceeds
the cutoff or when two comparably sized boxes are nearly nested: at least
90% of the smaller box overlaps, and its area is at least 25% of the larger
one. The second rule catches oversized duplicates that a standard IoU
cutoff can miss. An `nms_iou` of 1 turns both rules off.

The map hides detection tile outlines by default; **Tile outlines** restores
them. Small edge-cut boxes are dashed. When an edge-cut box covers at least
one quarter of its tile, the map and preview hide it by default. Use **Show
large partial boxes** in the inspector to review those estimates. The matching tile
and score remain available. A 256-pixel tile may contain only part of a large
site: the surrounding-imagery preview can show more of it, but LLMDet's
tile box does not measure the full site boundary.

The next request excludes all inspected tiles by dataset, shard, and row ID.
Once the minimum probe and a full recent window are available, the loop can
stop on low recent detection yield. It also stops at the candidate limit or
when retrieval finds no new candidates. Exact-only search scans once and
pages its bounded candidate pool. Overlapping tiles can contain the same
object, so result counts are tile counts. Missed detections and later ranks
can still contain objects, even when the loop stops on low yield.

## Measurement and limitations

LLMDet Large had the strongest box accuracy in the
[standalone benchmark](../benchmarks/model_selection/REPORT.md). Replaying its
105 labeled tile/query pairs showed that the previous 0.10 cutoff accepted
74/74 positive and 31/31 negative tiles. At 0.30 it accepted 73/74 positives
and 18/31 negatives. On the 19-pair exhaustive box subset, 0.30 and overlap
suppression reduced unmatched boxes from 76 to 5 at IoU 0.50, while matched
objects fell from 14/15 to 13/15. The lost match was a small boat. Lower the
cutoff for hard-to-see objects if you can review more false positives. The
labels are exploratory, and scores are not calibrated presence probabilities.

A second local audit on November 14 evaluated the first 96 base ANN
neighbors for each of nine queries (864 tile/query pairs) on the RX 9070 XT.
A visual review used 12 preset ranks per query from contact sheets without
model overlays. Four ambiguous tiles were left unscored; 69 positive and 35
negative tiles remained. These are locally reviewed labels, with overlapping
chips and a deliberately difficult negative group, not a statewide sample.

| Box cutoff | Reviewed positive tiles accepted | Reviewed negative tiles accepted | Boxes across all 864 pairs |
| --- | ---: | ---: | ---: |
| 0.10, previous behavior | 69/69 | 35/35 | 5,633 |
| 0.25 | 69/69 | 24/35 | 1,395 |
| **0.30** | **68/69** | **23/35** | **1,119** |
| 0.35 | 66/69 | 22/35 | 913 |
| 0.40 | 59/69 | 17/35 | 729 |
| 0.50 | 34/69 | 9/35 | 334 |

Rows other than the old baseline include within-tile overlap suppression at
IoU 0.50. The stronger default removes about 80% of boxes and rejects 12 of
the 35 reviewed negatives at the cost of one of 69 reviewed positives. It
still accepts 23 of the reviewed negatives. Example confusions included
wind turbines and shadows for helicopter, and boats and wakes for space
shuttle. Raising the cutoff further trades these errors against missed
boats, tennis courts, solar installations and other real objects. Extending
ANN retrieval cannot correct a confidently wrong box.

The runtime uses FP16 on the GPU, standard PyTorch operators, and the pinned
checkpoint's restored separate decoder heads. This addresses the
[Transformers tying bug](https://github.com/huggingface/transformers/issues/49529)
without editing the installed library. Batch size two leaves more headroom
than four when the text model is resident. Checking hundreds of tiles can
take minutes. The progress display shows the number checked and the yield.

The initial integration smoke on November 14, 2026 verified bounding boxes,
a real ANN refill, and resident model reuse. With GeoRSCLIP and LLMDet both
loaded, the two-query smoke peaked at 8.32 GiB allocated and 8.58 GiB reserved.

The follow-up used the live adaptive worker with the revised defaults. The
airplane search returned 254 tiles from 256 checked in eight rounds (93 s,
including first detector load and retrieval). The elephant search at the
default cutoff returned one candidate after 64 tiles; at 0.35 it returned
zero after 64 and stopped on low yield. The 0.45 helicopter run still returned
four candidates after 64. These checks verify variable output, deduplication of
candidate IDs, both stop paths, and resident model reuse. They do not label
every result. Peak allocated GPU memory was 8.32 GiB.

For the saved race-track example, the original top similarity tile had two
boxes with about 0.47 IoU, but the larger covered almost the entire smaller
box. On the RX 9070 XT the containment rule kept the stronger box and flagged
it as partial. A diagnostic run on a wider crop produced a broader box;
the production path continues to use native tiles. An Edge check using the
actual desktop service and saved query verified the confidence ranking and
the default/optional overlays on the real imagery.
