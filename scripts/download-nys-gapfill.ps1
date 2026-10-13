param([switch]$Prepare,[switch]$Pilot,[switch]$Monitor,[switch]$Status,[switch]$Background,[ValidateRange(1,4)][int]$Workers=4)
$ErrorActionPreference='Stop'
$repositoryRoot=Split-Path $PSScriptRoot -Parent
$pythonExecutable=Join-Path $repositoryRoot '.venv/Scripts/python.exe'
$supplementRoot=Join-Path $repositoryRoot 'data/nys-naip-2022-rgb-060m/supplements/naip-2021-release-gapfill'
if ($Monitor -or $Status) {
    & (Join-Path $PSScriptRoot 'watch-nys-naip.ps1') -OutputDir $supplementRoot -Once:$Status -Label 'NYS gap fill | NOAA 2021 release | 0.6 m/pixel | shared 300 GB cap'
    return
}
$savedPythonPath=$env:PYTHONPATH
try {
    $env:PYTHONPATH=Join-Path $repositoryRoot 'src'
    $arguments=@('-B','-m','geospatial_search.download_gapfill','--workers',$Workers.ToString())
    if ($Prepare) {$arguments+='--prepare'}
    if ($Pilot) {$arguments+='--pilot'}
    if ($Background) {
        $runtime=Join-Path $supplementRoot 'runtime'
        New-Item -ItemType Directory -Path $runtime -Force | Out-Null
        $stamp=[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')
        $process=Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $repositoryRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $runtime "download-$stamp.log") -RedirectStandardError (Join-Path $runtime "download-$stamp.err.log")
        Write-Host "Gap-fill worker started. Monitor: .\scripts\download-nys-gapfill.ps1 -Monitor"
        Write-Host "Logs: $runtime"
    } else {
        & $pythonExecutable @arguments
        if ($LASTEXITCODE -ne 0) {throw 'Gap-fill download failed; see its status and logs.'}
    }
} finally {$env:PYTHONPATH=$savedPythonPath}
