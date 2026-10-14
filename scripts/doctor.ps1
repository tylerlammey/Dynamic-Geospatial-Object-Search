param(
    [string]$PythonExecutable,
    [string]$JobRoot,
    [string]$CollectionDir,
    [string]$Config,
    [string]$ModelCache,
    [switch]$Json,
    [switch]$GpuProbe,
    [switch]$VerifyIndexChecksum,
    [switch]$Strict
)
$ErrorActionPreference = 'Stop'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonExecutable) {
    foreach ($candidate in @('.venv-dev/Scripts/python.exe', '.venv/Scripts/python.exe')) {
        $path = Join-Path $RepositoryRoot $candidate
        if (Test-Path -LiteralPath $path) { $PythonExecutable = $path; break }
    }
    if (-not $PythonExecutable) { $PythonExecutable = 'python' }
} elseif (-not [IO.Path]::IsPathRooted($PythonExecutable) -and (Test-Path -LiteralPath (Join-Path $RepositoryRoot $PythonExecutable))) {
    $PythonExecutable = Join-Path $RepositoryRoot $PythonExecutable
}
$pythonCommand = Get-Command $PythonExecutable -CommandType Application -ErrorAction Stop
$arguments = @('-B', (Join-Path $PSScriptRoot 'doctor.py'))
foreach ($pair in @(@('--job-root', $JobRoot), @('--collection-dir', $CollectionDir), @('--config', $Config), @('--model-cache', $ModelCache))) {
    if ($pair[1]) { $arguments += $pair }
}
if ($Json) { $arguments += '--json' }
if ($GpuProbe) { $arguments += '--gpu-probe' }
if ($VerifyIndexChecksum) { $arguments += '--verify-index-checksum' }
if ($Strict) { $arguments += '--strict' }
& $pythonCommand.Source @arguments
if ($LASTEXITCODE -ne 0) { throw 'Diagnostics reported errors or strict-mode warnings; inspect the observations above.' }
