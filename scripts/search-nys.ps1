param(
    [Parameter(Mandatory=$true)][string]$Query,
    [string]$CollectionDir,
    [ValidateRange(1,100)][int]$TopK = 10,
    [switch]$Offline,
    [switch]$AllowPartial,
    [ValidateSet('auto','exact','ann')][string]$Backend = 'auto',
    [ValidateSet('fast','balanced','thorough')][string]$SearchEffort = 'balanced'
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
$pythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Project Python is missing; run setup first.' }
$pythonArguments = @('-B','-m','geospatial_search.search_text','--query',$Query,'--top-k',$TopK.ToString(),
    '--backend',$Backend.ToLowerInvariant(),'--search-effort',$SearchEffort.ToLowerInvariant())
if ($CollectionDir) {
    if (-not [IO.Path]::IsPathRooted($CollectionDir)) { $CollectionDir = Join-Path $repositoryRoot $CollectionDir }
    $pythonArguments += @('--collection-dir',[IO.Path]::GetFullPath($CollectionDir))
}
if ($Offline) { $pythonArguments += '--offline' }
if ($AllowPartial) { $pythonArguments += '--allow-partial' }
$savedPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repositoryRoot 'src'
    # Windows PowerShell can turn harmless native stderr into terminating
    # errors when output is redirected. Use the actual process exit code.
    $savedInvocationPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $pythonExecutable @pythonArguments
        $searchExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedInvocationPreference }
    if ($searchExitCode -ne 0) { throw "Search exited with code $searchExitCode" }
} finally { $env:PYTHONPATH = $savedPythonPath }
