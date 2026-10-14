param(
    [ValidateSet('Offline','Docs','Browser','SyntheticBrowser','GpuSmoke')][string]$Mode = 'Offline',
    [string]$PythonExecutable,
    [string[]]$DocsFiles = @(),
    [string]$SmokeQuery = 'baseball field'
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
if (-not $PythonExecutable) {
    $environmentName = if ($Mode -in @('Browser','GpuSmoke')) { '.venv' } else { '.venv-dev' }
    $PythonExecutable = Join-Path $repositoryRoot "$environmentName\Scripts\python.exe"
    if ($Mode -eq 'Docs' -and -not (Test-Path -LiteralPath $PythonExecutable)) {
        $PythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
    }
} elseif (-not [IO.Path]::IsPathRooted($PythonExecutable)) {
    $PythonExecutable = Join-Path $repositoryRoot $PythonExecutable
}
if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Interpreter missing: $PythonExecutable. Offline/Docs: scripts/setup-dev.ps1; Browser/GpuSmoke: production setup guide. Validation does not install dependencies."
}
function Invoke-CheckedNative {
    param([string]$Executable, [string[]]$Arguments)
    # Fail before invocation if a tool is absent; a stale LASTEXITCODE is not success.
    # Runners may have several Node installations on PATH; invoke one path only.
    $nativeCommand = Get-Command $Executable -CommandType Application -TotalCount 1 -ErrorAction Stop
    # unittest progress uses stderr; Windows PowerShell must use the exit status.
    $savedInvocationPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $nativeCommand.Source @Arguments
        $nativeExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedInvocationPreference }
    if ($nativeExitCode -ne 0) { throw "$Executable exited with code $nativeExitCode." }
}
$savedEnvironment = @{}
foreach ($name in @('PYTHONPATH','PYTHONDONTWRITEBYTECODE','FRONTEND_DEV_PYTHON')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name,'Process')
}
Push-Location $repositoryRoot
try {
    $env:PYTHONPATH = (Join-Path $repositoryRoot 'src') + [IO.Path]::PathSeparator + (Join-Path $repositoryRoot 'frontend/service')
    $env:PYTHONDONTWRITEBYTECODE = '1'
    switch ($Mode) {
        'Docs' {
            $arguments = @('-B', (Join-Path $PSScriptRoot 'check_docs.py'))
            if ($DocsFiles.Count) { $arguments += '--files'; $arguments += $DocsFiles }
            Invoke-CheckedNative $PythonExecutable $arguments
        }
        'Offline' {
            Invoke-CheckedNative $PythonExecutable @('-B', (Join-Path $PSScriptRoot 'run_tests.py'), 'backend')
            Invoke-CheckedNative $PythonExecutable @('-B', (Join-Path $PSScriptRoot 'run_tests.py'), 'frontend')
            Invoke-CheckedNative $PythonExecutable @('-m', 'ruff', 'check', '.')
            Invoke-CheckedNative $PythonExecutable @('-m', 'ruff', 'format', '--check', '.')
            & (Join-Path $repositoryRoot 'frontend/scripts/build.ps1')
            Write-Host 'Offline checks passed: both synthetic suites (no skips), scoped lint/format, TypeScript and UI build. Browser and GPU checks were not requested.'
        }
        'Browser' {
            Write-Host 'Browser checks require Edge, production .venv, installed frontend dependencies, and local imagery; reuse existing ownership where available.'
            & (Join-Path $repositoryRoot 'frontend/scripts/build.ps1')
            Invoke-CheckedNative 'node' @((Join-Path $repositoryRoot 'frontend/tests/verify-ui.mjs'), '--auto-service')
            Invoke-CheckedNative 'node' @((Join-Path $repositoryRoot 'frontend/tests/verify-zoom.mjs'))
        }
        'SyntheticBrowser' {
            Write-Host 'Synthetic browser checks require Edge and the isolated .venv-dev; fixtures need no real imagery or model weights.'
            & (Join-Path $repositoryRoot 'frontend/scripts/build.ps1')
            $env:FRONTEND_DEV_PYTHON = $PythonExecutable
            Invoke-CheckedNative 'node' @((Join-Path $repositoryRoot 'frontend/tests/verify-synthetic.mjs'))
        }
        'GpuSmoke' {
            $livePython = Join-Path $repositoryRoot '.venv/Scripts/python.exe'
            if ([IO.Path]::GetFullPath($PythonExecutable) -ne [IO.Path]::GetFullPath($livePython)) {
                throw 'GpuSmoke uses the existing production .venv through the pipeline launchers.'
            }
            # One unique collection for this explicit smoke request, never statewide output.
            $collection = Join-Path $repositoryRoot ('data/nys-naip-2022-rgb-060m/embeddings/smoke-validation-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))
            Write-Host "Explicit GPU smoke collection: $collection. Existing GPU owners must release locks first."
            & (Join-Path $PSScriptRoot 'embed-nys.ps1') -SmokeTest -Foreground -Offline -OutputDir $collection
            if ($LASTEXITCODE -ne 0) { throw 'GPU embedding smoke failed.' }
            & (Join-Path $PSScriptRoot 'search-nys.ps1') -Query $SmokeQuery -CollectionDir $collection -Backend exact -TopK 3 -Offline
            if ($LASTEXITCODE -ne 0) { throw 'Text search of the new smoke collection failed.' }
        }
    }
} finally {
    Pop-Location
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name,$savedEnvironment[$name],'Process')
    }
}
