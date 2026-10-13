param(
    [switch]$Once,
    [ValidateRange(0,1000000)][int]$MaxUpdates = 0,
    [string]$OutputDir = 'data\nys-naip-2022-rgb-060m',
    [string]$Label = 'New York 2022 NAIP | 0.6 m/pixel | compressed RGB | 300 GB folder cap'
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
if (-not [IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repositoryRoot $OutputDir }
$jobRoot = [IO.Path]::GetFullPath($OutputDir)
$statusPath = Join-Path $jobRoot 'status.json'
$easternZone = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$updates = 0
$lastSummary = ''
$shownWorkerBars = 0

function Format-RemainingTime([double]$Seconds) {
    $span = [TimeSpan]::FromSeconds([Math]::Max(0,$Seconds))
    if ($span.TotalHours -ge 1) { return ('{0}h {1:00}m' -f [Math]::Floor($span.TotalHours),$span.Minutes) }
    if ($span.TotalMinutes -ge 1) { return ('{0}m {1:00}s' -f [Math]::Floor($span.TotalMinutes),$span.Seconds) }
    return ('{0}s' -f [Math]::Ceiling($span.TotalSeconds))
}

Write-Host $Label
Write-Host "Output: $jobRoot\imagery"
if (-not $Once) { Write-Host 'Ctrl+C closes this monitor; the background download continues.' }
try {
    while ($true) {
        if (-not (Test-Path -LiteralPath $statusPath)) {
            if ($Once) { Write-Host 'Waiting for the worker to create status.json.'; break }
            Start-Sleep -Seconds 1
            continue
        }
        # status.json is replaced atomically by the worker.
        try { $status = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json }
        catch { if ($Once) { throw }; Start-Sleep -Milliseconds 250; continue }
        $percent = if ($status.total_tiles -gt 0) { 100.0 * $status.completed_tiles / $status.total_tiles } else { 0 }
        $etaText = 'estimating'
        $remaining = -1
        if ($null -ne $status.eta_seconds) {
            $remaining = [int][Math]::Min([int]::MaxValue,[Math]::Max(0,$status.eta_seconds))
            $etaText = Format-RemainingTime $remaining
            if ($status.estimated_completion_utc) {
                $finishUtc = [DateTimeOffset]::Parse($status.estimated_completion_utc).UtcDateTime
                $finish = [TimeZoneInfo]::ConvertTimeFromUtc($finishUtc,$easternZone)
                $etaText += ' | finish ' + $finish.ToString('MMM d h:mm tt') + ' Eastern'
            }
        } elseif ($status.state -notin @('running','verifying')) { $etaText = 'not running' }
        $summary = '{0}: {1}/{2} tiles ({3:N2}%) | ETA {4} | imagery {5:N2} GB | folder {6:N2}/300 GB' -f `
            $status.state,$status.completed_tiles,$status.total_tiles,$percent,$etaText,`
            ($status.compressed_imagery_bytes/1e9),($status.folder_bytes/1e9)
        if ($status.configured_workers) { $summary += ' | workers ' + $status.configured_workers }
        $workerAlive = $null -ne (Get-Process -Id $status.pid -ErrorAction SilentlyContinue)
        if ($status.state -in @('running','verifying') -and -not $workerAlive) {
            Write-Host 'The worker has stopped; status.json is its last checkpoint.'
            Write-Host $summary
            Write-Host 'Run scripts\download-nys-naip.ps1 with the same OutputDir to verify completed files and resume.'
            break
        }
        if ($Once) { Write-Host $summary; if ($status.current_tile) { Write-Host ('Tile: '+$status.current_tile) }; break }
        Write-Progress -Id 1 -Activity 'New York statewide imagery' -Status $summary `
            -PercentComplete ([int][Math]::Min(100,$percent)) -SecondsRemaining $remaining
        if ($status.active_workers -and $status.active_workers.Count -gt 0) {
            $workerIndex = 0
            foreach ($tileWorker in $status.active_workers) {
                $workerPhase = $tileWorker.phase.Replace('_',' ')
                Write-Progress -Id (2+$workerIndex) -ParentId 1 -Activity $workerPhase -Status $tileWorker.tile `
                    -PercentComplete ([int][Math]::Min(100,[Math]::Max(0,$tileWorker.phase_percent)))
                $workerIndex++
            }
            for ($oldBar = $workerIndex; $oldBar -lt $shownWorkerBars; $oldBar++) {
                Write-Progress -Id (2+$oldBar) -Activity 'Tile worker' -Completed
            }
            $shownWorkerBars = $workerIndex
        } elseif ($status.phase) {
            for ($oldBar = 1; $oldBar -lt $shownWorkerBars; $oldBar++) {
                Write-Progress -Id (2+$oldBar) -Activity 'Tile worker' -Completed
            }
            $shownWorkerBars = 1
            $phase = $status.phase.Replace('_',' ')
            $phasePercent = [int][Math]::Min(100,[Math]::Max(0,$status.phase_percent))
            $detail = if ($status.current_tile) { $status.current_tile } else { 'Checking existing outputs' }
            Write-Progress -Id 2 -ParentId 1 -Activity $phase -Status $detail -PercentComplete $phasePercent
        }
        if ($summary -ne $lastSummary -and $updates % 5 -eq 0) {
            Write-Host $summary
            $lastSummary = $summary
        }
        $updates++
        if ($status.state -notin @('running','verifying')) {
            Write-Host $summary
            if ($status.error) { Write-Host ('Reason: '+$status.error) }
            break
        }
        if ($MaxUpdates -gt 0 -and $updates -ge $MaxUpdates) { break }
        Start-Sleep -Seconds 2
    }
} finally {
    for ($oldBar = 0; $oldBar -lt [Math]::Max(1,$shownWorkerBars); $oldBar++) {
        Write-Progress -Id (2+$oldBar) -Activity 'Tile worker' -Completed
    }
    Write-Progress -Id 1 -Activity 'New York statewide imagery' -Completed
}
