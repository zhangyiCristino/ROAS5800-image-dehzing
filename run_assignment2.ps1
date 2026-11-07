param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'assignment2\run.ps1') -NoBrowser:$NoBrowser
exit $LASTEXITCODE
