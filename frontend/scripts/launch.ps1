$ErrorActionPreference = 'Stop'
$FrontRoot = Split-Path -Parent $PSScriptRoot
$Electron = Join-Path $FrontRoot 'node_modules/electron/dist/electron.exe'
if (-not (Test-Path -LiteralPath $Electron)) { throw 'Run .\frontend\scripts\setup.ps1 first.' }
$NeedsBuild = -not (Test-Path -LiteralPath (Join-Path $FrontRoot 'build/index.html'))
if (-not $NeedsBuild) {
    $BuiltAt = (Get-Item -LiteralPath (Join-Path $FrontRoot 'build/index.html')).LastWriteTimeUtc
    $NeedsBuild = @(Get-ChildItem -LiteralPath (Join-Path $FrontRoot 'ui') -Recurse -File | Where-Object { $_.LastWriteTimeUtc -gt $BuiltAt }).Count -gt 0
}
if ($NeedsBuild) { & (Join-Path $PSScriptRoot 'build.ps1') }
# Electron is the interactive application; SW_HIDE also hides its main window.
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
$AppProcess = Start-Process -FilePath $Electron -ArgumentList @(('"' + $FrontRoot + '"')) -WorkingDirectory $FrontRoot -WindowStyle Normal -PassThru
$AllowedExecutables = @($Electron, (Join-Path $FrontRoot 'dist/win-unpacked/NYS Imagery Search.exe'))
for ($Attempt = 0; $Attempt -lt 240; $Attempt++) {
    $AppWindow = Get-Process electron,'NYS Imagery Search' -ErrorAction SilentlyContinue | Where-Object {
        $AllowedExecutables -contains $_.Path -and $_.MainWindowTitle -eq 'NYS Imagery Search' -and $_.MainWindowHandle -ne 0
    } | Select-Object -First 1
    if ($AppWindow) {
        Write-Host 'NYS Imagery Search is open. Use the mouse wheel to zoom into local aerial imagery.'
        return
    }
    $AppProcess.Refresh()
    if ($AppProcess.HasExited -and $AppProcess.ExitCode -ne 0) { throw ('Desktop startup failed (' + $AppProcess.ExitCode + '). See frontend/logs/desktop-launch.log and desktop-service.log.') }
    Start-Sleep -Milliseconds 250
}
throw 'No visible NYS Imagery Search window appeared within 60 seconds. See frontend/logs/desktop-launch.log and desktop-service.log.'
