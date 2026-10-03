# Running GeoRSCLIP embeddings on an AMD Radeon RX 9070 XT

This repository is on Windows with an RX 9070 XT. Use AMD's ROCm PyTorch
distribution, not the ordinary PyPI `torch` package: ROCm exposes the Radeon as
the PyTorch `cuda` device, which the embedding program requires.

## 1. Environment and AMD runtime

The repository-root `.venv` has been created with Python 3.12.10 and is ignored
by Git. Activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install a current AMD Radeon driver and the ROCm Core SDK for Windows first.
AMD's current Windows guide requires Python 3.12 and, in its ROCm 7.2.1 example,
Radeon driver 26.2.2 or newer. Confirm the latest compatible driver and release
for the RX 9070 XT before changing either dependency:

<https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html>

Then, from the activated environment, install the ROCm Core SDK packages and
the matching PyTorch wheels published by AMD. The following is AMD's documented
ROCm 7.2.1 Windows sequence. Replace all of the versions together only when the
AMD guide supplies a newer matching set.

```powershell
python -m pip install --upgrade pip

python -m pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

python -m pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl

python -m pip install -r requirements.txt
```

Verify that PyTorch sees the Radeon before running the model:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m torch.utils.collect_env
```

`torch.cuda.is_available()` must print `True`. Despite the name, that is the
correct PyTorch API for ROCm/HIP. Do not fall back to CPU for ViT-H/14.

## 2. Model download and cache

The program downloads this exact Hugging Face repository's checkpoint directly:

```text
BiliSakura/GeoRSCLIP-ViT-H-14
```

The repository does not contain the Transformers files (`config.json` and
`model.safetensors`) needed by the model-card example. It contains the
OpenCLIP-compatible `GeoRSCLIP-ViT-H-14.pt` checkpoint instead. The program
downloads that file automatically on the first run and keeps it in the
persistent Hugging Face cache, normally:

```text
C:\Users\Tyler\.cache\huggingface\hub
```

Restarting Windows or running the program again reuses that cache; it does not
download the model again unless the cached files are removed or the model ID or
filename is changed. The current CLI does not pin a Hugging Face revision, so
the recorded checkpoint SHA-256 is the reproducibility identifier. To keep the
cache on another drive, pass an explicit persistent directory such as
`--model-cache-dir D:\hf-cache`. Do not put a multi-gigabyte cache under Git;
if it lives in the repository, add it to `.gitignore`.

The output records the Hugging Face model ID, downloaded filename, checkpoint
SHA-256, and preprocessing fingerprint so incompatible vector spaces cannot be
mixed.

* <https://huggingface.co/Zilun/GeoRSCLIP>
* <https://github.com/om-ai-lab/RS5M>

## Citation

The Hugging Face model card requests citation of the original GeoRSCLIP work.
The repository's [`CITATION.cff`](../CITATION.cff) contains the machine-readable
version; cite Zhang et al. (2024), *RS5M and GeoRSCLIP: A Large-Scale
Vision-Language Dataset and a Large Vision-Language Model for Remote Sensing*,
IEEE Transactions on Geoscience and Remote Sensing, 62, 1-23,
<https://doi.org/10.1109/TGRS.2024.3449154>.

## 3. Generate Albany chips and vectors

The default grid is 512 x 512 native pixels at a 256-pixel stride. At Albany's
0.5-foot GSD, every chip covers 256 x 256 feet. It reads the original JP2
windows, uses natural-color bands 1/2/3 as RGB, and lets the model preprocess
them to its 224-pixel input. The fourth NIR band is preserved in the source
imagery but is not passed into this three-channel pretrained model.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m geospatial_search.embed_georsclip `
  --imagery-dir imagery/city_Albany_sp24 `
  --output-dir data/embeddings/albany_sp24/georsclip-vith14-s512-st256 `
  --dataset-id albany_sp24 `
  --batch-size 32
```

The RX 9070 XT has 16 GB VRAM. Start at batch size 32. For an out-of-memory
error, retry with a new output directory and batch size 16, then 8. If there is
plenty of headroom, benchmark 48 or 64 and retain the best end-to-end
chips/second. Float16 autocast is enabled by default; `--disable-amp` is only a
diagnostic fallback for a numerical or ROCm-kernel problem.

Before running Albany, perform a smoke test on a directory containing one JP2
tile and a distinct output directory. This validates JP2 decoding, chip
footprints, GPU visibility, and model loading without committing to the full
run.

## 4. Outputs and next search step

| File | Meaning |
| --- | --- |
| `embeddings.f16.npy` | Row-major L2-normalized float16 vectors; memory-mappable |
| `chips.geoparquet` | Chip footprints, source-pixel windows, IDs, and WGS84 display bounds |
| `run.json` | Dataset/grid/model/revision/vector contract |

`embedding_id` is the row number in `embeddings.f16.npy`; use it as the ID when
building a FAISS or other ANN index. The GeoParquet file resolves result IDs to
map footprints and source windows, so the vector index stays compact and fast.

The heavy work already runs in ROCm/PyTorch and GDAL native code. On this data,
JPEG 2000 decoding may limit throughput, so profile end-to-end speed before
considering C++. CUDA-only FAISS GPU packages and NVIDIA cuVS cannot run on this
AMD GPU; begin with CPU FAISS HNSW or exact search for this Albany-scale corpus.

## 5. Text-query smoke test

Render the ten nearest image chips for a natural-language query:

```powershell
$env:PYTHONPATH = "src"
python -m geospatial_search.search_text --query "baseball field"
```

It writes ten labelled PNGs, a `contact-sheet.png`, and `results.json` under a
new timestamped folder in `data/search-results/`. The query encoder uses the
same cached GeoRSCLIP checkpoint as the embedding job.
