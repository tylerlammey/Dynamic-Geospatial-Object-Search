param([switch]$Embeddings)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
$environmentPath = Join-Path $repositoryRoot '.venv'
$pythonExecutable = Join-Path $environmentPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    & py -3.12 -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 with the Python launcher, then rerun setup.' }
}
$requirementsPath = Join-Path $repositoryRoot 'requirements-nys-naip.txt'
if ($Embeddings) {
    & $pythonExecutable -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Install the GPU PyTorch wheel in .venv first; see docs\rx-9070xt-embedding-setup.md.' }
    $requirementsPath = Join-Path $repositoryRoot 'requirements.txt'
}
& $pythonExecutable -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'NYS dependencies are ready.'
