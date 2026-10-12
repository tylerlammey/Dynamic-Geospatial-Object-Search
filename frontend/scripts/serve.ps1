param([switch]$Browser)
$ErrorActionPreference = 'Stop'
$FrontRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $FrontRoot
$Python = if ($env:NYS_FRONTEND_PYTHON) { $env:NYS_FRONTEND_PYTHON } else { Join-Path $RepositoryRoot '.venv/Scripts/python.exe' }
$LogRoot = Join-Path $FrontRoot 'logs'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$TempRoot = Join-Path $FrontRoot 'temp'
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:GDAL_PAM_ENABLED = 'NO'
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
if (-not (Test-Path -LiteralPath (Join-Path $FrontRoot 'build/index.html'))) { & (Join-Path $PSScriptRoot 'build.ps1') }
$Ready = Join-Path $LogRoot ('browser-' + [guid]::NewGuid().ToString('N') + '.json')
$OwnedServer = Start-Process -FilePath $Python -ArgumentList @('-B', ('"' + (Join-Path $FrontRoot 'service/server.py') + '"'), '--frontend', ('"' + $FrontRoot + '"'), '--ready-file', ('"' + $Ready + '"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $LogRoot 'browser-service.out.log') -RedirectStandardError (Join-Path $LogRoot 'browser-service.err.log')
try {
    for ($Attempt = 0; $Attempt -lt 600 -and -not (Test-Path -LiteralPath $Ready); $Attempt++) {
        if ($OwnedServer.HasExited) { throw 'Local service stopped; see frontend/logs/browser-service.err.log.' }
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $Ready)) { throw 'Local service startup timed out.' }
    $Session = Get-Content -LiteralPath $Ready -Raw | ConvertFrom-Json
    $Address = 'http://127.0.0.1:' + $Session.port + '/#' + $Session.token
    if ($Browser) { Start-Process $Address }
    Write-Host ('Local development URL: ' + $Address)
    Write-Host 'Press Ctrl+C here to stop only this frontend service.'
    $OwnedServer.WaitForExit()
} finally {
    if (Test-Path -LiteralPath $Ready) {
        $Session = Get-Content -LiteralPath $Ready -Raw | ConvertFrom-Json
        try { Invoke-RestMethod -Method Post -Uri ('http://127.0.0.1:' + $Session.port + '/api/shutdown') -Headers @{Authorization = 'Bearer ' + $Session.token} -ContentType 'application/json' -Body '{}' | Out-Null } catch {}
    }
}
