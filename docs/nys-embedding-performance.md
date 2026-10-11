# NYS embedding performance verification

Measured October 11, 2026 on this Windows computer: AMD Radeon RX 9070 XT
(16 GB), PyTorch 2.9.1+rocm7.2.1, unchanged pinned GeoRSCLIP ViT-H/14 checkpoint,
unchanged OpenCLIP RGB preprocessing, float16 AMP, and the selected 256/192 grid.

## Results and default

Batch sizes 16, 32, 64, 96, 128, 192, 256, 384, and 512 were tested. GPU timing
used warm-up followed by three or four completed image-encoding calls, including
CPU-to-GPU transfer, inference, return transfer, and normalization. Medians avoid
selecting a setting from one unusually fast call. Representative results:

| Batch size | GPU chips/sec | Peak GPU memory reserved |
| --- | ---: | ---: |
| 16 | 120 | 5.70 GB |
| 32 | 162 | 5.86 GB |
| 64 | 184 | 6.30 GB |
| 128 | 217 | 6.99 GB |
| 192 | 200 | 7.50 GB |
| 256 | 206 | 8.61 GB |
| 384 | 207 | 9.82 GB |
| 512 | 45 | 11.70 GB |

Holding AMP's context open across batches was also tested. It did not improve
the winning 128-image setting, so the existing per-batch implementation remains.
ROCm pinned-memory transfers remain disabled because the previous validation
found shutdown hangs with this allocator. The model weights, model revision,
input size, overlap, mask threshold, and normalized-vector semantics were kept.

The full production pipeline was then run twice on three complete TIFFs, one
from each UTM zone, including source checksums, masks, RGB/preprocessing,
coordinates, vector normalization, actual atomic bundle publication, and progress
status. Both runs processed **9,880 chips** with four reader threads:

| Setting | Runtime | Chips/sec | GPU inference and transfers | Bundle publication |
| --- | ---: | ---: | ---: | ---: |
| Previous batch 32 | 67.28 s | 146.85 | 64.66 s | 0.44 s |
| Selected batch 128 | 53.14 s | 185.92 | 50.30 s | 0.25 s |

That is **26.6% more throughput**, or **21.0% less elapsed time**, on this sample.
Batch 128 is now the default. Published bundle checksums and every vector/metadata
row ID were validated. Comparing 64 spread-out embeddings per TIFF between
batch settings gave a minimum cosine of **0.99999928** after float16 storage.
Small differences from GPU batch arithmetic remain within the same vector space.

## Where time goes

Production readers prepared 768 candidate chips, including SHA verification,
masks, RGB transforms, and coordinates, at approximately 547, 761, and 1,096
chips/sec with one, two, and four threads. In the complete-source test, reader
wait was only 0.16 s at batch 32 and 0.38 s at batch 128; inference and transfers
accounted for roughly 95% of elapsed time. Four bounded reader threads leave
preparation headroom for more expensive tiles.

The reader already calls native GDAL code, and the model uses native PyTorch
GPU kernels. Waiting for readers consumed less than 1% of total runtime in
this run. A C++ tiler is unlikely to materially improve total
throughput on this hardware. Rasterio documents that GDAL raster reads release
the GIL: [concurrent processing](https://rasterio.readthedocs.io/en/stable/topics/concurrency.html).

This environment has no Python Triton package. An untested `torch.compile`
deployment or a PyTorch/runtime upgrade is not part of the selected pipeline.
The existing inference-mode and mixed-precision paths are retained. PyTorch
documents the AMP context and cache controls in its
[automatic mixed precision guide](https://docs.pytorch.org/docs/stable/amp.html).

These are the fastest settings tested, rather than proof of a global optimum.
Sources were locally cached during comparison, and longer runs can encounter
different masks, thermal conditions, storage traffic, and Windows workloads.
Extrapolating 185.92 chips/sec across 11,985,388 candidates gives about **17.9
hours**, but the full run's rolling ETA is more useful than that extrapolation.
Use a smaller batch on another GPU if its memory or measured throughput requires
it. Changing batch size does not change the pinned model or require discarding
compatible completed bundles.

Detailed reports are local and ignored by Git under the NYS job's
`tiling/benchmarks/`; temporary benchmark scripts are under ignored `temp/`.
