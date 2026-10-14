param([string]$PythonExecutable)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
$environmentPath = Join-Path $repositoryRoot '.venv-dev'
$devPython = Join-Path $environmentPath 'Scripts\python.exe'
foreach ($path in @($environmentPath, (Join-Path $environmentPath 'Scripts'), $devPython)) {
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'Development environment cannot be a symlink or junction; refusing to modify a redirected environment.'
        }
    }
}
if (-not (Test-Path -LiteralPath $devPython)) {
    if ($PythonExecutable) {
        & $PythonExecutable -m venv $environmentPath
    } else {
        & py -3.12 -m venv $environmentPath
    }
    if ($LASTEXITCODE -ne 0) { throw 'Creating .venv-dev failed; install Python 3.12 or pass -PythonExecutable.' }
}
& $devPython -c 'import sys; assert sys.version_info[:2] == (3,12), "Development checks require Python 3.12"'
if ($LASTEXITCODE -ne 0) { throw 'The isolated development environment must use Python 3.12.' }
& $devPython -c 'import os,sys; assert sys.prefix != sys.base_prefix and os.path.samefile(sys.prefix,sys.argv[1]), "Expected isolated .venv-dev"' $environmentPath
if ($LASTEXITCODE -ne 0) { throw 'Refusing dependency changes outside the isolated .venv-dev.' }
$constraintsPath = Join-Path $repositoryRoot 'constraints-dev.txt'
# This installer is deliberately separate from setup-nys.ps1 and the live .venv.
# Use the CPU index only for CPU wheels; resolve ordinary dependencies on PyPI.
& $devPython -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu -c $constraintsPath 'torch==2.9.1+cpu' 'torchvision==0.24.1+cpu'
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
& $devPython -m pip install -r (Join-Path $repositoryRoot 'requirements-dev.txt')
if ($LASTEXITCODE -ne 0) { throw 'Development dependency installation failed.' }
& $devPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Development dependency consistency check failed.' }
& $devPython -c 'import torch; assert torch.version.hip is None and torch.version.cuda is None, "Expected the CPU-only test build"'
if ($LASTEXITCODE -ne 0) { throw 'The isolated environment does not contain the expected CPU-only torch build.' }
Write-Host 'Isolated Python test dependencies are ready in .venv-dev. Install frontend dependencies with npm ci --ignore-scripts in frontend, then run scripts/validate.ps1 -Mode Offline.'
