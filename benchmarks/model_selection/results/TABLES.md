# Measured results

All latency numbers below are milliseconds; batching quantiles are per batch.

| Model | Status | B1 median/p95 | B4 median / tiles/s | B8 median / tiles/s | Load s | Peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|
| peekaboolean | ok | 21.7 / 23.4 | 34.8 / 114.3 | 47.7 / 169.0 | 6.40 | 1.48 |
| intern | ok | 2209.6 / 2306.3 | 2495.1 / 1.6 | 2659.1 / 3.0 | 5.69 | 2.03 |
| d3 | ok | 2204.9 / 2347.7 | 2321.6 / 1.7 | 2493.1 / 3.2 | 6.51 | 1.92 |
| blip | ok | 20.4 / 21.7 | 31.1 / 130.1 | 45.5 / 176.2 | 4.68 | 0.69 |
| gdino | ok | 157.5 / 175.3 | 500.3 / 8.5 | 928.3 / 8.5 | 5.00 | 3.76 |
| owlv2 | ok | 41.0 / 47.7 | 131.3 / 30.5 | 242.2 / 32.0 | 4.29 | 1.07 |
| llmdet | ok | 240.7 / 267.1 | 824.7 / 4.7 | unavailable | 5.19 | 8.44 |
| omdet | ok | 42.5 / 50.6 | 112.4 / 35.0 | 208.0 / 38.6 | 4.84 | 0.78 |
| sam3lite | ok | 147.5 / 149.3 | 524.8 / 7.6 | 1024.6 / 7.8 | 6.07 | 3.18 |

| Model | B1 CPU preparation | H2D | Forward wall / device | Postprocess |
|---|---:|---:|---:|---:|
| peekaboolean | 2.03 | 0.70 | 18.99 / 18.64 | 0.11 |
| intern | 2.42 | 0.91 | 2206.11 / 2205.85 | 0.13 |
| d3 | 2.08 | 0.81 | 2201.77 / 2201.49 | 0.20 |
| blip | 1.27 | 0.48 | 18.59 / 13.49 | 0.10 |
| gdino | 2.68 | 1.50 | 96.83 / 96.74 | 56.62 |
| owlv2 | 4.95 | 0.98 | 32.26 / 26.85 | 2.72 |
| llmdet | 2.70 | 1.49 | 230.72 / 230.63 | 5.42 |
| omdet | 1.74 | 0.79 | 25.58 / 12.50 | 16.38 |
| sam3lite | 3.21 | 0.98 | 142.37 / 116.72 | 0.89 |

## Exploratory assistant-label audit

These are NOT human-verified accuracy or calibration results.

| Model | Test n | Precision | Recall | FNR | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| peekaboolean | 59 | 0.780 | 0.914 | 0.086 | 0.137 | 0.143 |
| intern | 59 | 0.795 | 0.886 | 0.114 | 0.145 | 0.093 |
| d3 | 59 | 0.811 | 0.857 | 0.143 | 0.171 | 0.184 |
| blip | 59 | 0.800 | 0.914 | 0.086 | 0.157 | 0.153 |

| Detector | Custom AP50 | Custom AP50:95 | Small recall50 @ .1 |
|---|---:|---:|---:|
| gdino | 0.612 | 0.322 | 0.000 |
| owlv2 | 0.596 | 0.340 | 0.000 |
| llmdet | 0.680 | 0.362 | 0.500 |
| omdet | 0.401 | 0.198 | 0.000 |
| sam3lite | 0.561 | 0.310 | 0.000 |
