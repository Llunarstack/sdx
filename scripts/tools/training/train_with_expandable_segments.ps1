# Run training with CUDA allocator expandable segments (reduces fragmentation OOM).
# IMPORTANT: PYTORCH_CUDA_ALLOC_CONF must be set before Python loads torch — use this wrapper, not plain `python train.py`.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PyArgs = @()
)
$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
if (-not $env:CUDA_DEVICE_MAX_CONNECTIONS) {
    $env:CUDA_DEVICE_MAX_CONNECTIONS = "8"
}
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Push-Location $RepoRoot
try {
    python train.py @PyArgs
} finally {
    Pop-Location
}
