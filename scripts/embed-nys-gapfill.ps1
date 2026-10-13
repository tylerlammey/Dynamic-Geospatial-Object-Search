param([switch]$Plan,[switch]$SmokeTest,[switch]$Monitor,[switch]$Status,[switch]$Pause,[switch]$Resume,[switch]$Background,[switch]$Offline,[ValidateRange(1,512)][int]$BatchSize=128,[string]$OutputDir)
$ErrorActionPreference='Stop'
$repositoryRoot=Split-Path $PSScriptRoot -Parent
$pythonExecutable=Join-Path $repositoryRoot '.venv/Scripts/python.exe'
$embeddingRoot=Join-Path $repositoryRoot 'data/nys-naip-2022-rgb-060m/supplements/naip-2021-release-gapfill/embeddings'
if (-not $OutputDir) {
    $OutputDir=if($SmokeTest){Join-Path $embeddingRoot ('smoke-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))}else{Join-Path $embeddingRoot 'georsclip-vith14-s256-st192'}
}
if (-not [IO.Path]::IsPathRooted($OutputDir)) {$OutputDir=Join-Path $repositoryRoot $OutputDir}
$OutputDir=[IO.Path]::GetFullPath($OutputDir)
if (-not $OutputDir.StartsWith($embeddingRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'OutputDir must stay inside gap-fill embeddings.'}
if ($Monitor -or $Status) {& (Join-Path $PSScriptRoot 'watch-nys-embeddings.ps1') -OutputDir $OutputDir -Once:$Status;return}
$savedPythonPath=$env:PYTHONPATH
try {
    $env:PYTHONPATH=Join-Path $repositoryRoot 'src'
    if ($Pause) {& $pythonExecutable -B -m geospatial_search.gapfill_control --output-dir $OutputDir;if($LASTEXITCODE -ne 0){throw 'Gap-fill pause has not completed.'};return}
    $arguments=@('-B','-m','geospatial_search.embed_gapfill','--output-dir',$OutputDir,'--batch-size',$BatchSize.ToString())
    if ($Plan) {$arguments+='--plan'}
    if ($SmokeTest) {$arguments+='--smoke-test'}
    if ($Offline) {$arguments+='--offline'}
    if ($Resume -or (Test-Path -LiteralPath (Join-Path $OutputDir 'run.json'))) {$arguments+='--resume'}
    if ($Background) {
        & $pythonExecutable -B -m geospatial_search.gapfill_control --output-dir $OutputDir --check-running | Out-Null
        if ($LASTEXITCODE -eq 0) {Write-Host 'Gap-fill collection is already owned. Attaching to progress.';& (Join-Path $PSScriptRoot 'watch-nys-embeddings.ps1') -OutputDir $OutputDir -Once;return}
        $runtime=Join-Path $OutputDir 'runtime';New-Item -ItemType Directory -Path $runtime -Force | Out-Null
        $stamp=[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')
        $quotedArguments=foreach($argument in $arguments){'"'+$argument+'"'}
        Start-Process -FilePath $pythonExecutable -ArgumentList $quotedArguments -WorkingDirectory $repositoryRoot -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $runtime "embedding-$stamp.log") -RedirectStandardError (Join-Path $runtime "embedding-$stamp.err.log") | Out-Null
        $awakeArguments=@('-B','-m','geospatial_search.gapfill_control','--keep-awake','--output-dir',('"'+$OutputDir+'"'))
        Start-Process -FilePath $pythonExecutable -ArgumentList $awakeArguments -WorkingDirectory $repositoryRoot -WindowStyle Hidden | Out-Null
        Write-Host 'Gap-fill embedding started. Monitor: .\scripts\embed-nys-gapfill.ps1 -Monitor'
    } else {
        $savedInvocationPreference=$ErrorActionPreference
        try {$ErrorActionPreference='Continue';& $pythonExecutable @arguments;$embeddingExitCode=$LASTEXITCODE}
        finally {$ErrorActionPreference=$savedInvocationPreference}
        if($embeddingExitCode -ne 0){throw 'Gap-fill embedding failed; inspect its status and logs.'}
    }
} finally {$env:PYTHONPATH=$savedPythonPath}
