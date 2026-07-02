# Windows local setup — same Python stack as RunPod (no apt / native C++ build).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $env:SDX_SECRETS_FILE) { $env:SDX_SECRETS_FILE = "D:\Development\secret.txt" }
if (-not $env:SDX_DATA) { $env:SDX_DATA = "D:\Development\sdx-data" }
if (-not $env:SDX_PRETRAINED) { $env:SDX_PRETRAINED = Join-Path $Root "pretrained" }
if (-not $env:SDX_RESULTS) { $env:SDX_RESULTS = Join-Path $Root "results" }

Write-Host "SDX_SECRETS_FILE=$env:SDX_SECRETS_FILE"
Write-Host "SDX_DATA=$env:SDX_DATA"

python -m pip install -U pip wheel setuptools
python -m pip install -r runpod/requirements-runpod.txt

if ($env:SDX_SKIP_CUDA_WHEELS -ne "1") {
  $cudaOk = $false
  try {
    $cudaOk = python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $cudaOk = $true }
  } catch { $cudaOk = $false }
  if (-not $cudaOk) {
    Write-Host "Installing CUDA 12.8 wheels..."
    python -m pip install --force-reinstall -r requirements-cuda128.txt
  }
}

if ($env:SDX_SKIP_EDITABLE_INSTALL -ne "1") {
  python -m pip install -e ".[demo]"
}

New-Item -ItemType Directory -Force -Path $env:SDX_DATA, $env:SDX_PRETRAINED, $env:SDX_RESULTS | Out-Null

python -m toolkit.training.env_health
Write-Host "Setup complete."
