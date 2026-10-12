param(
    [ValidateRange(1,16)][int]$Threads = 8,
    [ValidateRange(1,65536)][int]$Partitions = 4096,
    [ValidateRange(256,2000000)][int]$TrainingVectors = 327680,
    [switch]$Force,
    [switch]$Status,
    [string]$CollectionDir = 'data\nys-naip-2022-rgb-060m\embeddings\georsclip-vith14-s256-st192'
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
if (-not [IO.Path]::IsPathRooted($CollectionDir)) { $CollectionDir = Join-Path $repositoryRoot $CollectionDir }
$CollectionDir = [IO.Path]::GetFullPath($CollectionDir)
if ($Status) {
    $statusPath = Join-Path $CollectionDir 'search-index\status.json'
    if (-not (Test-Path -LiteralPath $statusPath)) { Write-Host 'No index build status yet.'; return }
    Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json | Format-List
    return
}
$pythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Project Python is missing; run setup first.' }
$savedPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repositoryRoot 'src'
    $arguments = @('-B','-m','geospatial_search.build_search_index','--collection-dir',$CollectionDir,
        '--threads',$Threads.ToString(),'--partitions',$Partitions.ToString(),'--training-vectors',$TrainingVectors.ToString())
    if ($Force) { $arguments += '--force' }
    $savedInvocationPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $pythonExecutable @arguments
        $buildExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedInvocationPreference }
    if ($buildExitCode -ne 0) { throw 'Search index build failed; existing embeddings and previously published indexes are unchanged.' }
} finally { $env:PYTHONPATH = $savedPythonPath }
