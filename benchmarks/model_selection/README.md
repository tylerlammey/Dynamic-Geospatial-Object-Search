# Standalone model selection benchmark

Stages 2–4 (decision, ANN feedback, detection) remain **planned and unimplemented**.
This folder contains isolated model experiments; nothing in the application or
embedding/retrieval pipeline imports it. See [REPORT.md](REPORT.md) for findings.

The existing ROCm `.venv` is borrowed read-only. New adapter dependencies live
in this folder's ignored `dependencies/`; checkpoints in ignored `cache/`;
reviewed upstream sources in ignored `vendor/`; tile references, small audit
sheets, predictions and logs in ignored `local/`. No TIFFs are copied and no
embeddings are recalculated. Small audit sheets are review aids, not a chip dataset.
Summary measurements and source pins live in `results/` and [model_pins.json](model_pins.json).
No commits or publication are performed by these scripts.

Run from the repository root with Python 3.12. This host needs **GPU access
outside the restricted command sandbox**: discovery succeeds inside it, but an
ordinary allocation crashed with native `0xC0000005`; the same probe succeeded
on the host. Do not diagnose a candidate from that sandbox failure.

```powershell
# Observe owners first; benchmark also acquires real collection/GPU locks.
.\scripts\doctor.ps1
# Optional minimal owned GPU probe; needs working host GPU access.
.\.venv\Scripts\python.exe -u -B benchmarks/model_selection/probe_gpu.py
# Isolated small dependency overlay; leaves the ROCm environment untouched.
.\.venv\Scripts\python.exe -m pip install --no-deps --target benchmarks/model_selection/dependencies -r benchmarks/model_selection/requirements-benchmark.txt
# Review sources before use. Existing pins are reused; --refresh intentionally changes them.
.\.venv\Scripts\python.exe -B benchmarks/model_selection/fetch_sources.py
.\.venv\Scripts\python.exe -B benchmarks/model_selection/download_models.py peekaboolean intern d3 blip gdino owlv2 llmdet omdet sam3lite
# Reuse cached query vectors compatible with the collection. Force ANN; no exact fallback.
.\.venv\Scripts\python.exe -B benchmarks/model_selection/dataset.py --sheets
# Optional reproduction of the assistant's fallible visual audit, NOT human gold.
.\.venv\Scripts\python.exe -B benchmarks/model_selection/annotate_exploratory.py
.\.venv\Scripts\python.exe -u -B benchmarks/model_selection/suite.py
.\.venv-dev\Scripts\python.exe -B benchmarks/model_selection/evaluate.py
.\.venv-dev\Scripts\python.exe -B benchmarks/model_selection/analyze_cascade.py
.\.venv-dev\Scripts\python.exe -B benchmarks/model_selection/audit_split.py
.\.venv-dev\Scripts\python.exe -B benchmarks/model_selection/summarize_results.py
.\.venv\Scripts\python.exe -B benchmarks/model_selection/inventory.py
.\.venv-dev\Scripts\python.exe -B -m unittest discover -s benchmarks/model_selection -p 'test_*.py' -v
```

Never run two GPU experiments concurrently. The runner holds the production
job-wide GPU and base collection locks, refuses CPU fallback, and never sends
PAUSE/STOP signals or changes services. It reads only completed published rows.
If an owner is active, let it finish before running. Models are released at
subprocess exit, including native failures. Downloads do not acquire GPU locks.

`run.py --model MODEL` supports `--batches 1 4 8`, `--warmup 12`, `--repeats 25`,
`--dtype`, and `--tag`. Warm-up visits twelve query shapes before timing.
The suite defaults to batches 1/2/4 for LLMDet because batch 8 showed severe
memory pressure, and 1/4/8 for the other models. `--skip-predictions` marks a
timing-only follow-up, preserving the independent prediction audit. Additional
recorded grids use `--tag batch-grid --batches 16 32 64` for Peekaboolean and
`--tag batch8 --batches 8` for OmDet/SAM3-LiteText. Tagged results remain separate.
`--co-resident OTHER --tag residency` tests sequential inference with both models
loaded; `--with-retriever` retains the existing pinned GeoRSCLIP model as well.
It does not connect either model to the search pipeline. Compatibility
failures and OOMs are retained, never replaced by CPU times.

## Inputs and labels

`local/tiles.json` records globally qualified dataset/shard/embedding row IDs,
native source windows/checksums, collection recipe, index snapshot, cached-query
hash and selection diagnostics. `load_images` rechecks row identity and uses the
unchanged production `read_rgb` helper. The first audit-sheet build checks source
TIFF SHA-256 too. RGB remains native 256×256 at 0.6 m/pixel until each checkpoint's
required processor resize/normalization. Resize is recorded per model.

The sample starts with twelve cached arbitrary text queries, ANN ranks
1/2/5/20/100/250/500, and one random-source window per query (not a row-uniform
statewide draw). It adds twelve query pairs on existing windows for airplane and
small-boat audits. This is a convenience benchmark, not a representative survey
or a contiguous last-N feedback simulation. Acquisition/release metadata and
geographic footprints remain authoritative in the source collection.

The supplied labels are assistant-created before candidate predictions, with
ambiguous presence labels left null. Approximate boxes cover only a small subset;
one ambiguous pool tile is excluded from exhaustive box scoring. They are **not
human-verified ground truth**. A human should correct/export independent labels
before model selection for accuracy. Group entire source TIFFs when assigning
calibration/test splits; retain paired queries in the same split. The current
sample has no observed cross-split intersecting tile footprints, but adjacent
sources/provider buffers require checking on a larger evaluation. The class prior
differs markedly between the current calibration and test sets.

Metrics skip null labels/nonexhaustive boxes. Raw match/decision scores are never
assumed calibrated. Platt fitting uses only the calibration split; test labels
never choose its parameters. Custom COCO-style AP is explicitly distinguished
from official COCO evaluation; crowd, ignore regions and rotated boxes are absent.
Small objects use original area <32² pixels, regardless of model upsampling.
All evaluation scripts explicitly clip predicted boxes to the native image
boundary and discard empty clipped predictions; raw SAM3 outputs are preserved.
Annotation images/metrics are local experiments. Imagery attribution: Courtesy
USDA-FSA; distributed by NOAA Digital Coast.

## Timing contract

Each model gets a fresh process, `eval`, `inference_mode`, a fixed seed and eight
CPU threads. Batch sizes share identical source tiles and one text query per batch,
as an ANN page would. The prediction pass uses mixed-query batches of four.
The timer synchronizes before/after transfer, inference and postprocessing; GPU
events supplement forward wall time. Source TIFF open/decode is measured
separately; repeated warm stages start with the exact decoded PIL images in RAM.
Warm totals include CPU preparation, blocking ordinary H2D, forward and output
postprocessing. They exclude load/download/disk decode. There is no overlap/pinned
transfer. Quantiles describe batches; per-tile amortization is not batch-1 latency.
The reported p95 has 25 observations, so it is descriptive rather than a tight
tail estimate. VRAM is PyTorch allocated/reserved, not complete device/driver usage.
Load time uses cached checkpoints and warm OS caches, not a cold-disk guarantee.

The benchmark uses native Transformers operators and no NVIDIA custom kernels.
Intern/D3 use the upstream-equivalent flattened linear patch projection instead
of visual Conv3d; a synthetic equivalence regression covers this. Peekaboolean
merges LoRA and scores the duplicated yes/no question once, while retaining its
unknown candidate. Its scalar `noul=Pyes+0.5*Punknown` is a decision convention,
not literal calibrated object presence; raw yes/unknown components are saved.
BLIP is an image–text matching control, not a Jev-style presence model. Detector
box confidence is never relabeled as a calibrated tile probability.

The installed Transformers release silently shares LLMDet decoder heads that
the checkpoint keeps independent. `llmdet_heads.py` restores all per-layer head
tensors from the pinned checkpoint, verifies exact values and independent
storage, and preserves intended decoder/root aliases. `llmdet` measurements use
this corrected loading; `llmdet-stock` retains the invalid stock comparison.
This changes no installed package or checkpoint. See the report's upstream
issue references and the four head-restoration regression tests.

`profile_intern.py` attributes the ~2.2-second slowdown without replacing
operators. `intern-neumann --tag smoke-fast` records a rejected, guarded inverse
experiment: real activations failed the residual check and used native solves.
It is not an optimized/recommended model configuration.

Model/source licenses apply separately. Peekaboolean weights are noncommercial;
other tested model cards declare Apache-2.0, with underlying model terms still
applicable. Check current terms before deployment or redistributing weights.
Download only into this folder; do not install upstream Torch pins over ROCm.
