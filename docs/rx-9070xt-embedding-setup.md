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
filename is changed. The NYS configuration pins revision
`09af41b95fbb7b4bf884904e0d3c2777e83740bc` and records the checkpoint SHA-256.
To keep the
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

## 3. NYS embedding and search

The source imagery is now NYS 2022 NAIP RGB GeoTIFFs at 0.6 m/pixel.
The selected 256-pixel chip and 192-pixel stride provide 25% overlap.
Use the shared configuration and launchers in the
[NYS pipeline guide](nys-pipeline.md).

```powershell
.\scripts\setup-nys.ps1 -Embeddings
.\scripts\embed-nys.ps1 -Plan
.\scripts\embed-nys.ps1 -SmokeTest -Offline
# Start the statewide collection after smoke verification:
.\scripts\embed-nys.ps1 -Offline
.\scripts\embed-nys.ps1 -Pause
.\scripts\search-nys.ps1 -Query 'baseball field' -Offline
.\scripts\embed-nys.ps1 -Unpause -Offline
```

The RX 9070 XT has 16 GB VRAM. Batch size 128 is the measured default; its
GPU benchmark reserved about 7 GB. On out-of-memory, resume the same compatible
collection with `-BatchSize 64`, then 32, 16, or 8. For a smaller GPU, start at
32. See the [benchmark results](nys-embedding-performance.md). AMP is
enabled by default; changing it requires a distinct embedding output because
it is part of the numerical compatibility contract. Completed source bundles
are checksummed and skipped during resume.

Keep Windows awake. The embedding worker runs in the background; closing its
monitor leaves it running. Reattach with `scripts\embed-nys.ps1 -Monitor`.
Pause waits for the current tile to be saved and the GPU to be released;
search accepts the resulting partial collection automatically. Sources, plans,
embeddings, status, and search results all remain inside the existing NYS
folder. Preview/result coordinates use WGS84; source pixel grids remain in
NAD83 UTM. GPU-specific PyTorch installations may need different wheels on
another computer; the pipeline also supports a compatible NVIDIA CUDA build.
