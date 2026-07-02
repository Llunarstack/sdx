$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $env:SDX_SECRETS_FILE) { $env:SDX_SECRETS_FILE = "D:\Development\secret.txt" }
if (-not $env:SDX_DATA) { $env:SDX_DATA = Join-Path $Root "data" }
$SmokeData = Join-Path $env:SDX_DATA "integration_smoke"
python (Join-Path $Root "scripts\integration_smoke.py") --data-root $SmokeData --secrets $env:SDX_SECRETS_FILE @args
exit $LASTEXITCODE
