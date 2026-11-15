# start-web.ps1 - start the local Amharic speech-to-text web page
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\start-web.ps1
#
# Then open the link it prints (http://127.0.0.1:5000). It opens automatically.
# Press Ctrl+C in this window to stop the server.

$ErrorActionPreference = "Stop"

$Root      = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy    = Join-Path $Root ".venv\Scripts\python.exe"
$ModelFile = Join-Path $Root "models\shook-tiny-stage2\model.safetensors"

Set-Location $Root

if (-not (Test-Path $VenvPy)) {
    Write-Host "The virtual environment is missing." -ForegroundColor Red
    Write-Host "Run this first:  powershell -ExecutionPolicy Bypass -File .\setup.ps1"
    exit 1
}

if (-not (Test-Path $ModelFile)) {
    Write-Host "The model is missing from models\shook-tiny-stage2." -ForegroundColor Red
    Write-Host "Run this first:  powershell -ExecutionPolicy Bypass -File .\setup.ps1"
    exit 1
}

& $VenvPy app.py
