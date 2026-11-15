# start-web.ps1 - start the local Amharic speech-to-text web page
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\start-web.ps1
#
# Then open the link it prints (http://127.0.0.1:5000). It opens automatically.
# Press Ctrl+C in this window to stop the server.

$ErrorActionPreference = "Stop"

$Root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

Set-Location $Root

if (-not (Test-Path $VenvPy)) {
    Write-Host "The virtual environment is missing." -ForegroundColor Red
    Write-Host "Run this first:  powershell -ExecutionPolicy Bypass -File .\setup.ps1"
    exit 1
}

$Missing = @()
foreach ($ModelDir in @("shook-tiny-stage2", "wav2vec2-large-xlsr-53-amharic")) {
    $ModelFile = Join-Path $Root "models\$ModelDir\model.safetensors"
    if (-not (Test-Path $ModelFile)) { $Missing += $ModelDir }
}

if ($Missing.Count -gt 0) {
    Write-Host "These models are missing from .\models\:" -ForegroundColor Red
    foreach ($Name in $Missing) { Write-Host "  - $Name" }
    Write-Host "Run this first:  powershell -ExecutionPolicy Bypass -File .\setup.ps1"
    exit 1
}

& $VenvPy app.py
