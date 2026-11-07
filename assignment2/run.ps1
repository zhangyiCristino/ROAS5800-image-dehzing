param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$repoRoot = Split-Path $PSScriptRoot -Parent
$workspaceRoot = Split-Path $repoRoot -Parent
$pythonCandidates = @(
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    (Join-Path $repoRoot '.venv\Scripts\python.exe'),
    (Join-Path $workspaceRoot '.venv\Scripts\python.exe')
)
$assignmentPython = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $assignmentPython) { throw 'Run assignment2\setup.ps1 first.' }
try {
    $request = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:7861/config')
    $request.Proxy = $null
    $request.Timeout = 2000
    $response = $request.GetResponse()
    $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
    $config = $reader.ReadToEnd() | ConvertFrom-Json
    $reader.Dispose()
    $response.Dispose()
    if ($config.title -like '*Assignment 2*') {
        if (-not $NoBrowser) { & $assignmentPython -c "import webbrowser; webbrowser.open('http://127.0.0.1:7861')" }
        exit 0
    }
} catch {
    # Start a new Assignment 2 instance if no matching local instance exists.
}
$uiArguments = @((Join-Path $PSScriptRoot 'app.py'), '--port', '7861')
if (-not $NoBrowser) { $uiArguments += '--open' }
& $assignmentPython @uiArguments
exit $LASTEXITCODE
