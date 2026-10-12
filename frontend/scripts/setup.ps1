param([switch]$PrepareImagery, [switch]$NoLaunch)
$ErrorActionPreference = 'Stop'
$FrontRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $FrontRoot
$Python = if ($env:NYS_FRONTEND_PYTHON) { $env:NYS_FRONTEND_PYTHON } else { Join-Path $RepositoryRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'The existing project Python environment is required. See frontend/README.md.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
& $Python -B -c "import rasterio, numpy, PIL, shapely, pyproj; print('Read-only Python dependency check passed')"
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies are missing. Use an isolated frontend environment; do not modify the live worker environment.' }
$env:npm_config_cache = Join-Path $FrontRoot '.tools/npm-cache'
$env:ELECTRON_CACHE = Join-Path $FrontRoot '.tools/electron-cache'
$env:ELECTRON_BUILDER_CACHE = Join-Path $FrontRoot '.tools/builder-cache'
Push-Location $FrontRoot
try {
    & npm.cmd ci --ignore-scripts --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'Frontend npm install failed.' }
    & node (Join-Path $FrontRoot 'node_modules/electron/install.js')
    if ($LASTEXITCODE -ne 0) { throw 'Electron runtime setup failed.' }
} finally { Pop-Location }
& (Join-Path $PSScriptRoot 'build.ps1')
if ($PrepareImagery) {
    & (Join-Path $PSScriptRoot 'prepare-imagery.ps1')
    if (-not $NoLaunch) { & (Join-Path $PSScriptRoot 'launch.ps1') }
}
if (-not $PrepareImagery -or $NoLaunch) {
    Write-Host 'Setup complete. Open the app with .\frontend\scripts\launch.ps1'
}
