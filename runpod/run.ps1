$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $env:SDX_SECRETS_FILE) { $env:SDX_SECRETS_FILE = "D:\Development\secret.txt" }
if (-not $env:SDX_DATA) { $env:SDX_DATA = Join-Path $Root "data" }
if (-not $env:SDX_PRETRAINED) { $env:SDX_PRETRAINED = Join-Path $Root "pretrained" }
if (-not $env:SDX_RESULTS) { $env:SDX_RESULTS = Join-Path $Root "results" }
python (Join-Path $Root "scripts\run_pipeline.py") @args
exit $LASTEXITCODE
