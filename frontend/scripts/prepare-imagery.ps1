param([switch]$Force)
$ErrorActionPreference = 'Stop'
$FrontRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $FrontRoot
$Python = if ($env:NYS_FRONTEND_PYTHON) { $env:NYS_FRONTEND_PYTHON } else { Join-Path $RepositoryRoot '.venv/Scripts/python.exe' }
$LogRoot = Join-Path $FrontRoot 'logs'
$TempRoot = Join-Path $FrontRoot 'temp'
New-Item -ItemType Directory -Path $LogRoot, $TempRoot -Force | Out-Null
$Session = $null
$OwnedServer = $null
$ReadyPath = $null

function Get-FrontendStatus($Candidate) {
    if ($Candidate.port -lt 1 -or $Candidate.port -gt 65535 -or -not $Candidate.token) { throw 'Invalid frontend session.' }
    Invoke-RestMethod -Uri ('http://127.0.0.1:' + $Candidate.port + '/api/status') -Headers @{Authorization = 'Bearer ' + $Candidate.token} -TimeoutSec 10
}

try {
    # Reuse the desktop/browser service so there is only one owner of the cache.
    foreach ($File in (Get-ChildItem -LiteralPath $LogRoot -Filter '*.json' | Sort-Object LastWriteTime -Descending)) {
        if ($File.Name -notmatch '^(desktop|browser|prepare)-[a-f0-9-]+\.json$') { continue }
        try {
            $Candidate = Get-Content -LiteralPath $File.FullName -Raw | ConvertFrom-Json
            $Status = Get-FrontendStatus $Candidate
            $Session = $Candidate
            break
        } catch {}
    }
    if (-not $Session) {
        if (-not (Test-Path -LiteralPath $Python)) { throw 'The existing project Python environment is required. See frontend/README.md.' }
        $RunId = [guid]::NewGuid().ToString('N')
        $ReadyPath = Join-Path $LogRoot ('prepare-' + $RunId + '.json')
        $env:PYTHONDONTWRITEBYTECODE = '1'
        $env:GDAL_PAM_ENABLED = 'NO'
        $env:TEMP = $TempRoot
        $env:TMP = $TempRoot
        $ErrorLog = Join-Path $LogRoot ('prepare-' + $RunId + '.err.log')
        $OwnedServer = Start-Process -FilePath $Python -ArgumentList @('-B', ('"' + (Join-Path $FrontRoot 'service/server.py') + '"'), '--frontend', ('"' + $FrontRoot + '"'), '--ready-file', ('"' + $ReadyPath + '"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $LogRoot ('prepare-' + $RunId + '.out.log')) -RedirectStandardError $ErrorLog
        for ($Attempt = 0; $Attempt -lt 900 -and -not (Test-Path -LiteralPath $ReadyPath); $Attempt++) {
            if ($OwnedServer.HasExited) { throw ('Local service stopped. See ' + $ErrorLog) }
            Start-Sleep -Milliseconds 100
        }
        if (-not (Test-Path -LiteralPath $ReadyPath)) { throw 'Local service startup timed out.' }
        $Session = Get-Content -LiteralPath $ReadyPath -Raw | ConvertFrom-Json
        $Status = Get-FrontendStatus $Session
    }
    if ($Status.imagery_error) { throw $Status.imagery_error }
    $ApiRoot = 'http://127.0.0.1:' + $Session.port
    $Headers = @{Authorization = 'Bearer ' + $Session.token}
    $Active = $Status.overview.state -in @('starting', 'preparing', 'publishing')
    if ($Status.overview.ready -and -not $Force -and -not $Active) {
        Write-Host 'Photographic imagery is already prepared. Use -Force to rebuild the display mosaic.'
        return
    }
    if ($Active) { Write-Host 'Attached to the current photographic imagery preparation.' }
    else {
        Write-Host 'Preparing local photographic overviews. First preparation typically takes 10-20 minutes; cached work is reused.'
        Write-Host 'Generated imagery stays in frontend/cache/. Original TIFFs remain unchanged.'
        try {
            Invoke-RestMethod -Method Post -Uri ($ApiRoot + '/api/overview/prepare') -Headers $Headers -ContentType 'application/json' -Body '{"confirm":"prepare-local-overview"}' -TimeoutSec 30 | Out-Null
        } catch { throw ('Cannot start preparation: ' + $_.ErrorDetails.Message + ' ' + $_.Exception.Message) }
    }
    if ($OwnedServer) { Write-Host 'Ctrl+C stops this preparation; run the same command to resume cached work.' }
    else { Write-Host 'Ctrl+C stops this progress monitor; preparation continues in the open app.' }
    while ($true) {
        $Status = Get-FrontendStatus $Session
        $Overview = $Status.overview
        if ($Overview.state -eq 'complete' -and $Overview.ready) {
            Write-Host ('Photographic imagery ready: ' + $Overview.completed_sources + ' source overviews and the statewide display pyramid.')
            break
        }
        if ($Overview.state -in @('cancelled', 'error', 'idle')) { throw ('Preparation stopped: ' + $Overview.error + ' Run this script again while embedding is idle or paused to resume.') }
        $Total = [int]$Overview.total_sources
        $Completed = [int]$Overview.completed_sources
        $Percent = if ($Total -gt 0) { [int](100 * $Completed / $Total) } else { 0 }
        $ProgressArguments = @{Activity = 'Prepare photographic imagery'; Status = ($Overview.state + ': ' + $Completed + '/' + $Total + ' sources'); PercentComplete = $Percent}
        if ($Overview.eta_seconds -gt 0) { $ProgressArguments.SecondsRemaining = [int]$Overview.eta_seconds }
        Write-Progress @ProgressArguments
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Progress -Activity 'Prepare photographic imagery' -Completed
    # Never stop a service belonging to an already-open app.
    if ($OwnedServer) {
        if (-not $Session -and $ReadyPath -and (Test-Path -LiteralPath $ReadyPath)) {
            try { $Session = Get-Content -LiteralPath $ReadyPath -Raw | ConvertFrom-Json } catch {}
        }
        if ($Session) {
            try { Invoke-RestMethod -Method Post -Uri ('http://127.0.0.1:' + $Session.port + '/api/shutdown') -Headers @{Authorization = 'Bearer ' + $Session.token} -ContentType 'application/json' -Body '{}' -TimeoutSec 10 | Out-Null } catch {}
        }
        if (-not $OwnedServer.WaitForExit(15000)) { $OwnedServer.Kill() }
    }
}
