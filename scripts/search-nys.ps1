param(
    [Parameter(Mandatory=$true)][string]$Query,
    [string]$CollectionDir,
    [ValidateRange(1,100)][int]$TopK = 10,
    [switch]$Offline,
    [switch]$AllowPartial
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
$pythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$pythonArguments = @('-B','-m','geospatial_search.search_text','--query',$Query,'--top-k',$TopK.ToString())
if ($CollectionDir) {
    if (-not [IO.Path]::IsPathRooted($CollectionDir)) { $CollectionDir = Join-Path $repositoryRoot $CollectionDir }
    $pythonArguments += @('--collection-dir',[IO.Path]::GetFullPath($CollectionDir))
}
if ($Offline) { $pythonArguments += '--offline' }
if ($AllowPartial) { $pythonArguments += '--allow-partial' }
$savedPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repositoryRoot 'src'
    & $pythonExecutable @pythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Search exited with code $LASTEXITCODE" }
} finally { $env:PYTHONPATH = $savedPythonPath }
