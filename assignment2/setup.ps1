$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$assignmentEnv = Join-Path $PSScriptRoot '.venv'
$assignmentPython = Join-Path $assignmentEnv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $assignmentPython)) {
    & py -3.12 -m venv $assignmentEnv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
}
& $assignmentPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'GPU dependencies could not be installed.' }
& $assignmentPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Assignment 2 dependencies could not be installed.' }
& $assignmentPython (Join-Path $PSScriptRoot 'scripts\fetch_sources.py') --repos --ipc-weights
if ($LASTEXITCODE -ne 0) { throw 'Official model assets could not be prepared or verified.' }
Write-Host 'Assignment 2 is ready. Run run.ps1 to open the local interface.'
