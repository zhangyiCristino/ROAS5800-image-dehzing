$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) {
    $assignmentPython = $localPython
} elseif (Test-Path -LiteralPath $workspacePython) {
    $assignmentPython = $workspacePython
} else {
    Write-Host 'Please run setup.ps1 once before starting the interface.'
    exit 1
}
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:TORCH_COMPILE_DISABLE = '1'
$env:XFORMERS_DISABLED = '1'
# Reuse the already-running assignment page instead of starting a second server.
try {
    $localRequest = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:7860/config')
    $localRequest.Proxy = $null
    $localRequest.Timeout = 2000
    $localResponse = $localRequest.GetResponse()
    $localReader = New-Object System.IO.StreamReader($localResponse.GetResponseStream())
    $pageConfig = $localReader.ReadToEnd() | ConvertFrom-Json
    $localReader.Dispose()
    $localResponse.Dispose()
    if ($pageConfig.title -like '*ROAS5800*') {
        & $assignmentPython -c "import webbrowser; webbrowser.open('http://127.0.0.1:7860')"
        exit 0
    }
} catch {
    # No existing assignment server; start one below.
}
& $assignmentPython (Join-Path $PSScriptRoot 'app.py') --open
exit $LASTEXITCODE
