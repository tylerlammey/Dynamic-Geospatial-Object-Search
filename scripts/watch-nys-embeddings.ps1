param(
    [switch]$Once,
    [ValidateRange(0,1000000)][int]$MaxUpdates = 0,
    [string]$OutputDir = 'data\nys-naip-2022-rgb-060m\embeddings\georsclip-vith14-s256-st192'
)
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path $PSScriptRoot -Parent
if (-not [IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repositoryRoot $OutputDir }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$statusPath = Join-Path $OutputDir 'status.json'
$easternZone = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$updates = 0
Write-Host "NYS GeoRSCLIP embeddings | 153.6 m chips | 25% overlap`nCollection: $OutputDir"
if (-not $Once) { Write-Host 'Ctrl+C closes this monitor; the background worker continues. Use embed-nys.ps1 -Pause to stop safely.' }
try {
    while ($true) {
        if (-not (Test-Path -LiteralPath $statusPath)) { Write-Host 'No embedding status yet. Start it with scripts\embed-nys.ps1.'; break }
        try { $status = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json }
        catch { if ($Once) { throw }; Start-Sleep -Milliseconds 250; continue }
        $percent = [Math]::Max(0.0,[Math]::Min(100.0,[double]$status.percent_complete))
        $running = $status.state -in @('loading','running','pausing')
        $alive = $status.pid -and ($null -ne (Get-Process -Id $status.pid -ErrorAction SilentlyContinue))
        $remaining = -1
        $eta = if ($running) { 'estimating' } else { 'not running' }
        if ($status.state -eq 'running' -and $null -ne $status.eta_seconds) {
            $remaining = [int][Math]::Min([int]::MaxValue,[Math]::Max(0,$status.eta_seconds))
            $eta = $status.eta_hms
            $finishUtc = [DateTimeOffset]::Parse($status.updated_utc).UtcDateTime.AddSeconds($remaining)
            $finish = [TimeZoneInfo]::ConvertTimeFromUtc($finishUtc,$easternZone)
            $eta += ' | finish ' + $finish.ToString('MMM d h:mm tt') + ' Eastern'
        }
        $runtime = if ($status.elapsed_hms) { $status.elapsed_hms } else { '00:00:00' }
        $summary = '{0}: {1:N2}% | {2:N0}/{3:N0} windows | runtime {4} | ETA {5} | saved vectors {6:N0} | tiles {7}/{8} | folder {9:N2}/300 GB' -f `
            $status.state,$percent,$status.processed_candidate_chips,$status.total_candidate_chips,$runtime,$eta,`
            $status.durable_embedded_chips,$status.completed_sources,$status.total_sources,($status.folder_bytes/1e9)
        if ($running -and -not $alive) {
            Write-Host "Worker stopped; this is its last checkpoint.`n$summary"
            $latestError = Get-ChildItem -LiteralPath (Join-Path $OutputDir 'runtime') -Filter '*.err.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latestError) { Write-Host "Check log: $($latestError.FullName)" }
            Write-Host 'Rerun scripts\embed-nys.ps1 with the same OutputDir to verify and resume saved tiles.'
            break
        }
        if ($Once) { Write-Host $summary; if ($status.current_tile) { Write-Host ('Tile: '+$status.current_tile) }; if ($status.error) { Write-Host ('Reason: '+$status.error) }; break }
        Write-Progress -Id 1 -Activity 'New York statewide embeddings' -Status $summary -PercentComplete ([int]$percent) -SecondsRemaining $remaining
        if ($updates % 5 -eq 0 -or -not $running) { Write-Host $summary }
        $updates++
        if (-not $running) { if ($status.error) { Write-Host ('Reason: '+$status.error) }; break }
        if ($MaxUpdates -gt 0 -and $updates -ge $MaxUpdates) { break }
        Start-Sleep -Seconds 2
    }
} finally { Write-Progress -Id 1 -Activity 'New York statewide embeddings' -Completed }
