param(
    [string]$Python,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$FrontRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $FrontRoot
if (-not $Python) { $Python = Join-Path $RepoRoot '.venv-dev/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Development Python is missing. Follow docs/DEVELOPMENT.md to create .venv-dev, or supply -Python.'
}
$SessionName = 'synthetic-' + [guid]::NewGuid().ToString('N')
$JobPath = Join-Path $FrontRoot "temp/$SessionName-job"
$StatePath = Join-Path $FrontRoot "temp/$SessionName-state"
$PreviousPythonPath = $env:PYTHONPATH
$PreviousPam = $env:GDAL_PAM_ENABLED
try {
    $env:PYTHONPATH = (Join-Path $RepoRoot 'src') + ';' + (Join-Path $FrontRoot 'service')
    $env:GDAL_PAM_ENABLED = 'NO'
    if (-not $SkipBuild) { & (Join-Path $PSScriptRoot 'build.ps1') }
    & $Python -B (Join-Path $RepoRoot 'scripts/create_synthetic_job.py') --output $JobPath
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic fixture generation failed.' }
    Write-Host 'Serving synthetic development data. Open the printed local URL; Ctrl+C stops this owned service.'
    & $Python -B (Join-Path $FrontRoot 'service/dev_server.py') --job $JobPath --state $StatePath
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic development service failed.' }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
    $env:GDAL_PAM_ENABLED = $PreviousPam
}
