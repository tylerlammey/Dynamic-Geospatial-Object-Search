# Standalone model research and benchmark

Observed on 2026-10-26, Windows, AMD Radeon RX 9070 XT 16 GB, ROCm-enabled
PyTorch 2.9.1+rocm7.2.1, Transformers 5.17.0. Stages 2–4 remain **planned,
unimplemented**. No pipeline, service, retrieval defaults, recipes or model
environment were modified. [Reproduction and protocol](README.md),
[full revision/source pins](model_pins.json), [measured tables](results/TABLES.md).

## Recommendations from the measurements

**Lightweight decision candidate: Peekaboolean-450M, for further calibration and
labeling rather than immediate automatic rejection.** Warm batch-1 median/p95
was **21.7/23.4 ms**, versus approximately 2.2 seconds for both Qwen-based decision
models on this Windows ROCm stack. Batch 8 reached 169 tiles/s; batch 32 reached
202 tiles/s, with about 1.20 GiB peak allocated memory. Batch 64 was slightly
slower, so larger batches are not automatically better. BLIP was similarly fast,
but its image/text matching objective is a weaker fit to the requested decision
architecture. Peekaboolean's noncommercial weights remain a material constraint.

**Accuracy detector candidate: LLMDet large, with its checkpoint's per-layer
heads restored and verified.** It had the best exploratory box AP among the five
tested detector families: custom AP50 **0.680**, AP50:95 **0.362**. At box score .1,
it matched 14/15 audited objects; it found one of the two small boats, while the
other detectors found neither at IoU .5. Warm median/p95 was **240.7/267.1 ms**
for one tile and median 824.7 ms for four. This is a candidate recommendation,
not proof of high accuracy: only 19 tile/query pairs have exhaustive, approximate,
assistant-created boxes. Corrected measurements are in [llmdet.json](results/llmdet.json).

**Keep OWLv2 base ensemble as the fast direct-detection comparator.** It ran at
41.0/47.7 ms per tile and custom AP50:95 .340, close to LLMDet on this tiny audit.
Its speed makes an extra decision stage much harder to justify. SAM3-LiteText
was included as an additional current detector/presence-head candidate; its
direct presence head had only 57.1% recall at .5 on this exploratory test split.
It should not be assumed to be a safe lightweight gate.

**No tested gate has established the requested extremely high recall or reliable
arbitrary-query presence probabilities.** Peekaboolean's test recall at .5 was
32/35 (91.4%), with Brier .137 and ECE .143. Its learned unknown handling does not
turn the scalar into a verified probability. A small calibration-only Platt fit
worsened test Brier to .145 and ECE to .149. A threshold with no calibration misses
still missed one held-out positive. These findings argue against automating an
ANN stopping rule from these scores yet.

## The 2.2-second result and compatibility findings

The previous benchmark's exact script/model was not found in the inspected
checkout/temp/results, and has not been identified by the user. However,
Intern-Decision-0.8B and D3-lite independently reproduced approximately 2.2 s/tile.
This is an implementation-path bottleneck on the installed stack, not evidence
that all discriminative decision models require seconds.

The synchronized [Intern profile](results/intern-profile.json) attributes a median
**2,069 ms to 36 triangular solves**, inside 18 Gated DeltaNet layers, against
approximately 2,183 ms uninstrumented forward time. Vision took about 15 ms;
preparation/transfer took about 3 ms in the main benchmark. Native reference
convolution was about 11 ms in total. Nested profile times are inclusive and
must not be added. The missing optimized FLA kernel triggers the native
reference/triangular-solve path. A suitable ROCm/RDNA4 kernel is a real
optimization opportunity; CUDA performance claims do not establish its availability
on this Windows machine.

A benchmark-only finite Neumann inverse shortcut was tested with a residual
guard. Real activations made it unstable: the guard rejected all 18 inverses in
the last audited forward, so all 36 solves used the native fallback and latency
remained about 2.23 s. [Rejected optimization](results/solver-ablation.json).
It is excluded from model recommendations; the code and eight synthetic tests
retain the negative finding rather than silently using changed probabilities.

LLMDet initially emitted thin, misplaced boxes in both FP16 and an FP32 spot
check. The installed Transformers 5.17 implementation unconditionally ties
decoder heads despite the checkpoint's nonsharing flags. The same defect is
documented in [issue 49529](https://github.com/huggingface/transformers/issues/49529)
and [PR 49552](https://github.com/huggingface/transformers/pull/49552). The standalone
helper restores **42 tensors across 12 heads**, checks their exact values after
dtype conversion, and verifies independent storage plus the intended root/decoder
aliases. This repairs the intended checkpoint, not its learned weights. Custom
AP50:95 rose from .011 to .362. The invalid stock case is preserved in
[llmdet-stock.json](results/llmdet-stock.json), and is not ranked as the intended
model. No installed library or production file was patched.

OmDet's legacy timm config attempted a Hub repository lookup even in offline
mode. Materializing its exact declared local timm backbone configuration fixed
loading without fetching replacement weights. All nine candidates then completed
GPU inference. Native SAM3 boxes can extend beyond the tile; all evaluated
predictions use the same explicit clipping policy, preserving raw outputs.

Warm and first-use timing are very different. Cached checkpoint loading was
roughly 4–7 seconds per candidate, but the first SAM3-LiteText batch-1 call took
19.1 seconds; its first new batch-8 call took 109.1 seconds before settling near
1.02 s/batch. These first-use initialization/shape costs are retained in warm-up
records and excluded from warm quantiles. They should be budgeted separately
from weight-loading time. No warm-up iteration is counted as a timed trial.

The uncorrected LLMDet batch-8 probe allocated 16.14 GiB by PyTorch's counter and
took 8.37 s/batch, versus .82 s at batch 4. That timing cliff is consistent with
Windows GPU-memory pressure/overcommit, but physical residency was not separately
measured. Use batch 2–4 with headroom. The corrected run deliberately uses
1/2/4; the stock batch-8 result is a memory-limit observation, not an accuracy
benchmark. Allocated/reserved counters exclude some driver/display usage.

Source TIFF open/mask/RGB decode was measured separately: the first verifier run
had median 4.33 ms and p95 5.30 ms. These are local, mostly cached observations,
not guaranteed cold-storage latency. Pass the already decoded tile between
prospective stages rather than decoding it twice.

## Candidate selection

The decision comparison deliberately includes different discriminative mechanisms:

- **Peekaboolean-450M:** LFM2.5 vision/language backbone, merged LoRA and learned
  yes/no readout. One batched forward scores the question and unknown candidate;
  no generated text. It preserves the release's size-specific temperature. Its
  `noul` adds half of unknown mass to yes, so this scalar requires aerial
  calibration before it can mean object-presence probability. Noncommercial
  weights are a deployment constraint. [Model](https://huggingface.co/bykof/peekaboolean-450m),
  [source/report](https://github.com/bykof/peekaboolean).
- **Intern-Decision-0.8B:** trained decision-token candidate readout on Qwen3.5,
  one forward, allowed A/B logits, published temperature 2.747760550703. The
  benchmark restricts the output projection to the same two rows instead of
  computing an unused vocabulary distribution. [Model and inference contract](https://huggingface.co/internlm/Intern-Decision-0.8B).
- **D3-lite:** approximately 0.9B total parameters, learned 255-code head reduced
  to the two allowed options, noncausal full-attention layers alongside causal
  recurrent layers. The native adapter preserves its trained attention contract;
  it does not silently evaluate a different stock causal model. It uses no
  NVIDIA/remote optimized kernels. [Model and runtime](https://huggingface.co/vllm-sr/d3-lite).
- **BLIP ITM base:** cross-attention match/nonmatch softmax, a useful smaller
  control. It is not an object-presence-trained model; a caption match score is
  not proof of probability calibration. [Model](https://huggingface.co/Salesforce/blip-itm-base-coco).

All four accept arbitrary text with an image. They do not use autoregressive
generation. BLIP provides the contrasting matching objective, without treating
embedding similarity alone as the preferred decision architecture.

The detector comparison includes five candidates across different families:

| Candidate | Why included | Required input from original 256-pixel tiles |
|---|---|---|
| Grounding DINO base | Established text-conditioned detector | 800-pixel short side, max 1333 |
| LLMDet large | Higher-capacity accuracy challenger; training LLM absent at inference | 800-pixel short side, max 1333 |
| OWLv2 base ensemble | Independent patch detector; supports text and image queries | 960 square |
| OmDet-Turbo Swin tiny | Fast direct-detection comparator, native language cache | 640 square |
| SAM3-LiteText S0 | Concept detector plus direct presence-head challenger | 1008 square; full native mask branch also executed |

These are the released processors' required resize policies; original pixels
remain unchanged before that step. Scores and box coordinates are returned at
the original 256-pixel scale. Upsampling does not recover unseen small-object
detail. [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base),
[LLMDet](https://huggingface.co/iSEE-Laboratory/llmdet_large),
[OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble),
[OmDet-Turbo](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf),
[SAM3-LiteText](https://huggingface.co/vil-uob/sam3-litetext-s0).

Related research and unmeasured alternatives:

- Imajev 2B/4B has a trained decision head and explicit unknown state. Its shipped
  four answer-order averaging adds cost; the 2B adapter alone is not a complete
  model. Larger capacity should earn its residency/latency cost on aerial labels.
  [Imajev](https://huggingface.co/mohit67890/imajev-2b).
- SAM3 has a query-conditioned `presence_logits` head plus boxes. It remains
  detector-sized computation rather than an automatically lightweight gate.
  Its base checkpoint is gated/custom-license; no terms were accepted here.
  The measured SAM3-LiteText retains the large vision backbone, so text savings
  need not materially reduce per-new-tile cost when queries are cached.
  [SAM3 output contract](https://huggingface.co/docs/transformers/en/model_doc/sam3),
  [LiteText](https://huggingface.co/vil-uob/sam3-litetext-s0).
- Remote-sensing specialists LAE-DINO and OpenRSD merit a follow-up, especially
  oriented/small boxes, but their custom MMDetection/MMCV/MMRotate stacks require
  explicit Windows ROCm operator validation. No unmeasured leaderboard claim is
  substituted for local accuracy. [LAE-DINO](https://github.com/jaychempan/LAE-DINO),
  [OpenRSD](https://github.com/floatingstarZ/OpenRSD).
- YOLOE is a useful additional fast promptable detector, with separate AGPL
  licensing and runtime considerations. H2O-Lightning currently recommends a
  custom attention/runtime path; stock causal inference would be a different
  configuration. Hosted GDINO/DINO-X API results do not benchmark this GPU.
  [YOLOE](https://github.com/THU-MIG/yoloe),
  [H2O-Lightning](https://huggingface.co/h2oai/h2o-lightning-4b).
- TypeSafe Jev 1.14 now supports experimental image/box requests, so older claims
  that Jev has no vision are stale. It is a hosted comparison, not a local
  open-weight RX 9070 XT benchmark. No imagery was sent to hosted inference.
  [Official announcement](https://typesafe.ai/blog/jev-1-14).

## Evidence limits

The frozen sample has 108 image/query pairs on 96 existing embedded windows,
covering twelve original queries plus paired airplane/boat queries. All model
inputs use the same pixels. Labels were created by the assistant from native
review sheets before candidate predictions and are **unverified, exploratory**.
There are 105 presence labels (74 positive, 31 negative), three uncertain rows,
and only a small approximate box audit. This cannot establish the best detector
for NYS aerial imagery or validate extremely high recall for arbitrary objects.

Calibration/test sets keep sources and paired queries together. A separate
read-only footprint audit found no cross-split intersecting tiles. Nonetheless,
class priors differ: calibration has 39 positives/7 negatives, test 35/24.
Several query subsets contain only one class. Pooled temperature/Platt results
are an experiment, not a deployment calibration. Sparse ANN ranks do not measure
a contiguous last-N stopping policy. Low-score rejected tiles need labels too.

## Decision economics and adaptive feedback

For equal numbers of candidate tiles, let D be warm decision cost, T direct
detection cost, and a the fraction passed to detection. Serial cascade cost is
`D + a*T` per candidate, versus `T` directly. It saves compute only when
`a < 1-D/T`; if `D >= T`, it cannot win even when all tiles are rejected.
Compare matching batch regimes and include any load/unload or retrieval costs.
For K candidates, load overhead breaks even only when
`K*(T-D-a*T)` exceeds extra loading/switching cost. The measurements use cached
weights and make warm residency explicit.

Latency savings alone do not establish better search. Cascade object recall is
gate recall multiplied by detector recall **conditional on passing the gate**,
not two marginal recalls. Paired predictions must measure which detectable
objects the gate actually removes. The gate's tile-level false negatives can
remove multiple objects and can also falsely terminate retrieval. To compensate
by examining additional neighbors, under the simplifying constant-prevalence
assumption, the stronger useful-result break-even condition is `D/T+a < r`,
where r is gate recall. Later ANN neighbors usually violate that constant-prior
assumption, so this is only an analytic scenario.

For tail prevalence p, gate sensitivity r and false-positive rate f, the expected
hard accepted fraction is `r*p + f*(1-p)`, not p. A mean soft score estimates tail
prevalence only when calibrated for the query, rank stratum and imagery domain.
Unknown mass, scene ambiguity and low-resolution small objects require a
conservative pass/abstain policy. Calibrate with a geographically separated,
rank-stratified sample that includes rejected tiles, then audit a held-out set.
Use a fixed examination budget and conservative stopping evidence; no stopping
logic has been implemented here.

Even zero misses on approximately 299 **independent** positives only supports a
one-sided 95% lower bound near 99% recall. Overlapping tiles and per-query
guarantees need more care and generally more data. The current small, fallible
audit does not meet this requirement.

## Measured tradeoffs on this audit

The [paired cascade scenarios](results/cascade.json) use the independently run
detector predictions and remove them on rejected tiles. They are an offline
analysis, not implementation or validation of the proposed architecture.

For Peekaboolean followed by corrected LLMDet, the batch-4 isolated costs are
8.71 ms/decision tile and 206.18 ms/detection tile. Thus more than **4.22%** of
candidates must be rejected just to save warm computation, before loading,
extra ANN requests, partial batches or residency effects.

| Decision cutoff | Passed fraction of 108 audit pairs | Missed positive tile/query labels | Paired object recall @ IoU .5 | Small-object recall | Estimated warm speedup, matching batch 4 |
|---|---:|---:|---:|---:|---:|
| Direct LLMDet | 100% | — | 14/15 | 1/2 | 1.000× |
| .05 | 95.37% | 0/74 | 14/15 | 1/2 | 1.004× |
| .10 | 90.74% | 1/74 | 13/15 | 0/2 | 1.053× |
| .50 | 76.85% | 4/74 | 12/15 | 0/2 | 1.233× |

These cutoffs are illustrative scenarios. Zero observed misses at .05 does not
establish high recall on new queries. In particular, raising the gate to .10
removes the small boat that the accurate detector found. False-negative harm is
therefore directly observable, not inferred from independent marginal recalls.

At the .05 scenario, the fast direct detectors are better served by direct
detection: matching-batch arithmetic gives speedups .820× for OWLv2 and .791×
for OmDet (values below one mean slower). Intern/D3's current 2.2-second paths
cannot accelerate any tested detector even if every candidate is rejected.

If a sufficiently large queue permits decision batch 32 while LLMDet uses batch
4, the optimistic arithmetic improves to roughly 1.023× at .05. That assumes
full batches and isolated measured stage costs; the 6.4-second extra model load
alone needs about 1,400 candidates to amortize such a small saving. First-use
initialization and added ANN/disk work make cold-session break-even harder.
Low-prevalence ANN tails could provide larger savings, but this rank-sparse,
enriched convenience sample cannot measure that benefit or safe stopping.

## Simultaneous residency and scheduling

These are actual sequential, **all-pass** co-resident runs, including both
verifier and detector preparation/forward/postprocessing for every tile.
GeoRSCLIP was loaded with the existing pinned checkpoint and retained as its
normal FP32 model; it performed no query inference during these measurements.

| Resident models | Batch | Median combined ms/batch | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| Peekaboolean + SAM3-LiteText + GeoRSCLIP | 1 | 167.7 | 5.93 | 6.19 |
| Same | 4 | 553.3 | 6.72 | 7.36 |
| Peekaboolean + corrected LLMDet + GeoRSCLIP | 1 | 266.4 | 7.23 | 8.00 |
| Same | 2 | 448.8 | 9.18 | 10.60 |
| Same | 4 | 4013.7 | 13.03 | 15.72 |
| Peekaboolean + corrected LLMDet, GeoRSCLIP released | 4 | 1158.1 | 9.29 | 11.94 |

The two selected models can remain resident. Retaining the query encoder favors
batch 1–2 on this host; batch 4 causes a severe latency increase even though
allocated tensors appear to fit. Releasing the query encoder improves batch 4,
but the co-resident pair still costs more than the isolated-stage sum. These
observations are why the simple break-even estimates above are optimistic.
There is no measured production speedup to claim.

For a future implementation, benchmark the actual GPU memory budget, scheduling
and conditional batch fill before choosing residency policy. Avoid duplicate
image decoding, unnecessary full-vocabulary projections, unmerged LoRA and
repeated query tokenization. Cache language features only where the model's
fusion contract permits it. Preserve cold/first-shape setup separately from warm
latency. None of these recommendations changes application behavior here.

## Validation and remaining work

All nine final candidate runs completed on the requested GPU. Thirty-five
standalone CPU tests passed, covering calibration leakage, score metrics, box
matching/clipping, D3 attention/padding, the rejected inverse guard and exact
LLMDet head restoration. The repository Offline tier passed both synthetic
suites, scoped lint/format, TypeScript and Vite build. The sandbox's default
temporary-directory permissions prevented its first attempt; the successful
host run used temporary directories under this benchmark's ignored `local/`.
Browser and pipeline `GpuSmoke` tiers were not run; the UI/pipeline were not
changed, and standalone model GPU measurements are recorded separately.

The [split audit](results/split-audit.json) found 96 unique native windows,
80 source groups, no shared sources/tiles across splits and no intersecting
cross-split footprints. [Checkpoint hashes](results/checkpoint-inventory.json),
[runtime versions](results/runtime-observed.json) and
[display driver](results/display-driver.json) make the measured environment
explicit. No production data, recipe or installed environment was changed.

Before firm accuracy selection, obtain human-verified, query/rank-stratified
presence and exhaustive box labels on independent geographic blocks, including
small objects, clipped objects, subtype distinctions, absent objects and hard
negatives. Define those labeling rules before inspecting predictions. Fit
calibration and choose thresholds on a calibration split, then evaluate once on
held-out queries/regions and rejected tiles. Run a separate contiguous-rank
benchmark for ANN-tail feedback. Those are evaluation requirements, not
authorization to implement stages 2–4.
