param(
    [switch]$Plan,
    [switch]$SmokeTest,
    [Alias('Unpause')][switch]$Resume,
    [switch]$Pause,
    [ValidateRange(1,86400)][int]$PauseTimeoutSeconds = 300,
    [switch]$Status,
    [switch]$Monitor,
    [switch]$Background,
    [switch]$Foreground,
    [switch]$Offline,
    [ValidateRange(1,512)][Nullable[int]]$BatchSize,
    [ValidateRange(1,8)][Nullable[int]]$Workers,
    [string]$OutputDir
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
$pythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$watchPath = Join-Path $PSScriptRoot 'watch-nys-embeddings.ps1'
$embeddingRoot = [IO.Path]::GetFullPath((Join-Path $repositoryRoot 'data\nys-naip-2022-rgb-060m\embeddings'))
if ($SmokeTest -and $Background -and -not $OutputDir) {
    if ($Resume) { throw 'Resuming a smoke test requires its -OutputDir.' }
    $OutputDir = Join-Path $embeddingRoot ('smoke-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))
}
$selectedOutput = if ($OutputDir) { $OutputDir } else { Join-Path $embeddingRoot 'georsclip-vith14-s256-st192' }
if (-not [IO.Path]::IsPathRooted($selectedOutput)) { $selectedOutput = Join-Path $repositoryRoot $selectedOutput }
$selectedOutput = [IO.Path]::GetFullPath($selectedOutput)
if (-not $selectedOutput.StartsWith($embeddingRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Embedding OutputDir must be a collection inside data\nys-naip-2022-rgb-060m\embeddings.'
}
if ($Pause -and ($Resume -or $Plan -or $SmokeTest -or $Status -or $Monitor -or $Foreground -or $Background)) {
    throw 'Use -Pause separately from launch/monitor/status switches.'
}
if ($Status -or $Monitor) {
    & $watchPath -OutputDir $selectedOutput -Once:$Status
    return
}
if (($Plan -or $Foreground) -and $Background) { throw 'Use -Background separately from -Plan or -Foreground.' }
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Run scripts\setup-nys.ps1 and follow the GPU setup guide first.' }
$savedEnvironment = @{}
foreach ($name in @('PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTHONUNBUFFERED')) { $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    $env:PYTHONPATH = Join-Path $repositoryRoot 'src'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUNBUFFERED = '1'
    if ($Pause) {
        & $pythonExecutable -B -m geospatial_search.embedding_control --output-dir $selectedOutput --timeout $PauseTimeoutSeconds
        if ($LASTEXITCODE -ne 0) { throw 'Pause has not completed; check status/logs before searching.' }
        return
    }
    $arguments = @('-B','-m','geospatial_search.embed_georsclip')
    if ($null -ne $BatchSize) { $arguments += @('--batch-size',$BatchSize.ToString()) }
    if ($null -ne $Workers) { $arguments += @('--workers',$Workers.ToString()) }
    if ($Offline) { $arguments += '--offline' }
    if ($Plan -or ($SmokeTest -and -not $Background) -or $Foreground) {
        if ($OutputDir -or $Foreground) { $arguments += @('--output-dir',$selectedOutput) }
        if ($Plan) { $arguments += '--plan' }
        if ($SmokeTest) { $arguments += '--smoke-test' }
        if ($Resume) { $arguments += '--resume' }
        & $pythonExecutable @arguments
        if ($LASTEXITCODE -ne 0) { throw "Embedding exited with code $LASTEXITCODE; completed bundles are resumable." }
        return
    }
    # Advisory check; the worker's OS lock is the authoritative race-safe guard.
    if ($SmokeTest) { $arguments += '--smoke-test' }
    & $pythonExecutable -B -m geospatial_search.embedding_control --output-dir $selectedOutput --check-running | Out-Null
    $busyCode = $LASTEXITCODE
    if ($busyCode -notin @(0,1)) { throw 'Could not check the embedding collection lock.' }
    if ($busyCode -eq 0) {
        Write-Host 'A process already owns this collection. Attaching to its progress.'
    } else {
        $statusPath = Join-Path $selectedOutput 'status.json'
        $catalogPath = Join-Path $selectedOutput 'collection.json'
        if (Test-Path -LiteralPath $catalogPath) {
            $catalog = Get-Content -LiteralPath $catalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($catalog.state -eq 'complete') {
                Write-Host 'This collection is complete.'
                & $watchPath -OutputDir $selectedOutput -Once
                $global:LASTEXITCODE = 0
                return
            }
        }
        $runtime = Join-Path $selectedOutput 'runtime'
        New-Item -ItemType Directory -Path $runtime -Force | Out-Null
        $arguments += @('--output-dir',$selectedOutput)
        if ($Resume -or (Test-Path -LiteralPath (Join-Path $selectedOutput 'run.json')) -or (Test-Path -LiteralPath $statusPath)) { $arguments += '--resume' }
        $stamp = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + ([Guid]::NewGuid().ToString('N').Substring(0,8))
        $stdoutPath = Join-Path $runtime "embedding-$stamp.log"
        $stderrPath = Join-Path $runtime "embedding-$stamp.err.log"
        # Windows Start-Process requires explicit quoting for paths with spaces.
        $quotedArguments = foreach ($argument in $arguments) {
            if ($argument.Contains('"')) { throw 'Arguments cannot contain double quotes.' }
            '"' + $argument + '"'
        }
        $launchTime = [DateTime]::UtcNow
        $process = Start-Process -FilePath $pythonExecutable -ArgumentList $quotedArguments -WorkingDirectory $repositoryRoot `
            -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $ready = $false
        for ($attempt = 0; $attempt -lt 120; $attempt++) {
            Start-Sleep -Milliseconds 250
            if (Test-Path -LiteralPath $statusPath) {
                try { $checkpoint = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { continue }
                if ([DateTimeOffset]::Parse($checkpoint.updated_utc).UtcDateTime -ge $launchTime) {
                    if ($checkpoint.state -eq 'failed') { throw "Embedding failed: $($checkpoint.error). See $stderrPath" }
                    $ready = $true
                    break
                }
            }
            if ($process.HasExited) {
                & $pythonExecutable -B -m geospatial_search.embedding_control --output-dir $selectedOutput --check-running | Out-Null
                if ($LASTEXITCODE -eq 0) { $ready = $true; Write-Host 'Another launcher started the worker. Attaching.'; break }
                $detail = Get-Content -LiteralPath $stderrPath -Tail 12 -ErrorAction SilentlyContinue | Out-String
                throw "Embedding exited during startup. See $stderrPath`n$detail"
            }
        }
        if (-not $ready) { throw "Worker startup is taking longer than expected. It was left running; use -Monitor or inspect $stderrPath before restarting." }
        Write-Host "Embedding worker started. Logs: $runtime"
    }
    if ($Background) { & $watchPath -OutputDir $selectedOutput -Once }
    else { & $watchPath -OutputDir $selectedOutput }
    $global:LASTEXITCODE = 0
} finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name,$savedEnvironment[$name],'Process') }
}
