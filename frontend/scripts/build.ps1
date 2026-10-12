param([switch]$Package)
$ErrorActionPreference = 'Stop'
$FrontRoot = Split-Path -Parent $PSScriptRoot
$env:npm_config_cache = Join-Path $FrontRoot '.tools/npm-cache'
$env:ELECTRON_CACHE = Join-Path $FrontRoot '.tools/electron-cache'
$env:ELECTRON_BUILDER_CACHE = Join-Path $FrontRoot '.tools/builder-cache'
$env:CSC_IDENTITY_AUTO_DISCOVERY = 'false'
Push-Location $FrontRoot
try {
    if ($Package) { & npm.cmd run package } else { & npm.cmd run build }
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
