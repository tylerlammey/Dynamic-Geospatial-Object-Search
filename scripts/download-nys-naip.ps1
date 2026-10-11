param(
    [switch]$Background,
    [switch]$Monitor,
    [switch]$Status,
    [switch]$Pilot,
    [switch]$Prepare,
    [ValidateRange(1,4)][int]$Workers = 4,
    [string]$OutputDir = 'data\nys-naip-2022-rgb-060m'
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
if (-not [IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repositoryRoot $OutputDir }
$jobRoot = [IO.Path]::GetFullPath($OutputDir)
$pythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$pipelinePath = Join-Path $repositoryRoot 'src\geospatial_search\download_nys_naip.py'
$runtimePath = Join-Path $jobRoot 'runtime'
$temporaryPath = Join-Path $jobRoot 'temporary'
$watchPath = Join-Path $PSScriptRoot 'watch-nys-naip.ps1'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw "Python environment missing: $pythonExecutable" }
if ($Monitor -or $Status) {
    & $watchPath -OutputDir $jobRoot -Once:$Status
    return
}
New-Item -ItemType Directory -Path $runtimePath,$temporaryPath -Force | Out-Null
$arguments = @('-B', ('"{0}"' -f $pipelinePath), '--output-dir', ('"{0}"' -f $jobRoot), '--workers', $Workers.ToString())
if ($Pilot) { $arguments += '--pilot' }
if ($Prepare) { $arguments += '--prepare' }
& $pythonExecutable -B $pipelinePath --output-dir $jobRoot --check-running | Out-Null
$alreadyRunning = $LASTEXITCODE -eq 0
if ($LASTEXITCODE -notin @(0,1)) { throw 'Could not check the existing worker lock.' }
if ($Pilot -or $Prepare) {
    if ($alreadyRunning) { throw 'The statewide worker is active. Use -Monitor or -Status to inspect it.' }
    $foregroundArguments = @('-B', $pipelinePath, '--output-dir', $jobRoot, '--workers', $Workers.ToString())
    if ($Pilot) { $foregroundArguments += '--pilot' }
    if ($Prepare) { $foregroundArguments += '--prepare' }
    & $pythonExecutable @foregroundArguments
    if ($LASTEXITCODE -ne 0) { throw "Worker exited with code $LASTEXITCODE" }
    return
}
if ($alreadyRunning) {
    Write-Host 'The download is already running; attaching to its progress.'
} else {
    # Calling run.ps1 is an explicit request to start/resume this job.
    $stopPath = Join-Path $jobRoot 'STOP'
    if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath }
    $logSuffix = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)
    $consoleLog = Join-Path $runtimePath ('console-' + $logSuffix + '.log')
    $errorLog = Join-Path $runtimePath ('errors-' + $logSuffix + '.log')
    $launchTime = [DateTimeOffset]::UtcNow
    $workerEnvironment = @{
        PYTHONDONTWRITEBYTECODE = '1'
        PYTHONUNBUFFERED = '1'
        TEMP = $temporaryPath
        TMP = $temporaryPath
        GDAL_TMPDIR = $temporaryPath
    }
    $savedEnvironment = @{}
    try {
        foreach ($environmentName in $workerEnvironment.Keys) {
            $savedEnvironment[$environmentName] = [Environment]::GetEnvironmentVariable($environmentName,'Process')
            [Environment]::SetEnvironmentVariable($environmentName,$workerEnvironment[$environmentName],'Process')
        }
        $process = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments `
            -WorkingDirectory $jobRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $consoleLog -RedirectStandardError $errorLog
    } finally {
        foreach ($environmentName in $savedEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($environmentName,$savedEnvironment[$environmentName],'Process')
        }
    }
    $ready = $false
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        $checkpointPath = Join-Path $jobRoot 'status.json'
        if (Test-Path -LiteralPath $checkpointPath) {
            $checkpoint = Get-Content -LiteralPath $checkpointPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ([DateTimeOffset]::Parse($checkpoint.updated_utc) -ge $launchTime) {
                if ($checkpoint.state -in @('running','verifying','complete')) { $ready = $true; break }
                if ($checkpoint.state -eq 'failed') { throw "Worker failed: $($checkpoint.error). Log: $errorLog" }
            }
        }
        if ($process.HasExited) { throw "Worker exited before startup. See $errorLog" }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) { throw "Worker startup was not confirmed; inspect $errorLog and status.json before retrying." }
    Write-Host 'Started the resumable background download.'
}
if ($Background) { & $watchPath -OutputDir $jobRoot -Once; return }
& $watchPath -OutputDir $jobRoot
