$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Local virtual environment not found. Create it first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv; .\venv\Scripts\pip install -r requirements.txt" -ForegroundColor White
    exit 1
}

& $VenvPython (Join-Path $ProjectRoot "main.py") @args
