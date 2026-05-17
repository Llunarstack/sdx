# Multi-GPU local DDP launcher (standalone). Repo root inferred from script location.
param(
    [Parameter(Position = 0)]
    [int]$Ngpu = 2,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
if ($Ngpu -lt 1) {
    throw "Ngpu must be >= 1 (got $Ngpu)"
}
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Push-Location $RepoRoot
try {
    python -m torch.distributed.run --standalone --nproc_per_node=$Ngpu train.py @ExtraArgs
} finally {
    Pop-Location
}
