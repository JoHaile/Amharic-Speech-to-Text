# setup.ps1 - one-time setup for offline Amharic speech-to-text (CPU only)
#
# Creates a virtual environment, installs CPU-only PyTorch plus the other
# dependencies, and downloads the Shook Tiny Amharic model into .\models\.
#
# Anonymous download: no Hugging Face account, token, login or API key is used
# or requested anywhere in this script.
#
# Safe to re-run. It never deletes an existing model and skips work already done.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"

$Root      = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv      = Join-Path $Root ".venv"
$VenvPy    = Join-Path $Venv "Scripts\python.exe"
$ModelDir  = Join-Path $Root "models\shook-tiny-stage2"
$ModelFile = Join-Path $ModelDir "model.safetensors"
$RepoId    = "b1n1yam/shook-tiny-amharic-stage2-polish"

Write-Host "=== Amharic ASR (Shook Tiny) setup ===" -ForegroundColor Cyan
Write-Host "Project: $Root"

# --- 1. Virtual environment ------------------------------------------------
if (Test-Path $VenvPy) {
    Write-Host "[1/4] Virtual environment already exists - reusing it."
} else {
    Write-Host "[1/4] Creating virtual environment (.venv)..."
    python -m venv $Venv
    if (-not (Test-Path $VenvPy)) {
        throw "Failed to create the virtual environment. Is Python 3.10+ on PATH?"
    }
}

# --- 2. pip ---------------------------------------------------------------
Write-Host "[2/4] Upgrading pip..."
& $VenvPy -m pip install --upgrade pip --quiet

# --- 3. Dependencies ------------------------------------------------------
# CPU-only PyTorch from the official CPU index. No CUDA, no NVIDIA packages.
Write-Host "[3/4] Installing CPU-only PyTorch (this can take a few minutes)..."
& $VenvPy -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0+cpu"

Write-Host "      Installing remaining dependencies..."
& $VenvPy -m pip install `
    "transformers==5.22.2" `
    "huggingface_hub==2.4.0" `
    "safetensors==0.8.0" `
    "soundfile==0.14.0" `
    "librosa==1.0.0" `
    "numpy==2.5.4" `
    "psutil==7.2.2"

# --- 4. Model download ----------------------------------------------------
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

if (Test-Path $ModelFile) {
    $sizeMb = [math]::Round((Get-Item $ModelFile).Length / 1MB, 1)
    Write-Host "[4/4] Model already present ($sizeMb MB) - skipping download."
} else {
    Write-Host "[4/4] Downloading model files anonymously (no token)..."
    Write-Host "      Repo: $RepoId"

    $py = @'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="b1n1yam/shook-tiny-amharic-stage2-polish",
    local_dir="models/shook-tiny-stage2",
    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
    ignore_patterns=["runs/*", "*.tfevents*", "training_args.bin"],
    token=False,          # anonymous, public files only
)
print("download complete")
'@
    $tmp = Join-Path $env:TEMP "shook_download.py"
    Set-Content -Path $tmp -Value $py -Encoding UTF8
    Push-Location $Root
    try {
        & $VenvPy $tmp
        if ($LASTEXITCODE -ne 0) { throw "Model download failed (see the error above)." }
    } finally {
        Pop-Location
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path $ModelFile)) {
    throw "Model weights are missing after setup: $ModelFile"
}

# --- Verify: load fully offline -------------------------------------------
Write-Host ""
Write-Host "Verifying the model loads offline (no network)..."
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$check = @'
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

proc = AutoProcessor.from_pretrained("models/shook-tiny-stage2", local_files_only=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "models/shook-tiny-stage2", local_files_only=True, dtype=torch.float32
)
model.to("cpu").eval()
print("OK - loaded offline on CPU")
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
'@
$tmpCheck = Join-Path $env:TEMP "shook_verify.py"
Set-Content -Path $tmpCheck -Value $check -Encoding UTF8
Push-Location $Root
try {
    & $VenvPy $tmpCheck
    if ($LASTEXITCODE -ne 0) { throw "Offline load check failed." }
} finally {
    Pop-Location
    Remove-Item $tmpCheck -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Transcribe a file with:"
Write-Host "  .\.venv\Scripts\python.exe transcribe.py `"audio\your_file.wav`""
Write-Host "Interactive mode:"
Write-Host "  .\.venv\Scripts\python.exe transcribe.py"
