$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        & $pythonLauncher.Source -3.12 -m venv (Join-Path $PSScriptRoot '.venv')
    } else {
        $basePython = Get-Command python -ErrorAction SilentlyContinue
        if (-not $basePython) { throw 'Install Python 3.12 before running setup.' }
        & $basePython.Source -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)"
        if ($LASTEXITCODE -ne 0) { throw 'The python command must point to Python 3.12.' }
        & $basePython.Source -m venv (Join-Path $PSScriptRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install it, then run this script again.' }
}
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip.' }
& $venvPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Could not install the tested PyTorch build.' }
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the assignment dependencies.' }
Write-Host 'Environment ready. Run .\.venv\Scripts\python.exe scripts/prepare_assets.py, then .\run.ps1.'
