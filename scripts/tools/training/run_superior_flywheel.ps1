# One-click Superior Stack flywheel (Windows PowerShell).
# Usage:
#   .\scripts\tools\training\run_superior_flywheel.ps1 -BaseCkpt results\best.pt -WorkDir flywheel_run

param(
    [Parameter(Mandatory = $true)]
    [string]$BaseCkpt,
    [string]$WorkDir = "flywheel_run",
    [string]$LocalRagJsonl = "",
    [string]$VitCkpt = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $RepoRoot

$cmd = @(
    "python", "-m", "scripts.tools", "run_flywheel",
    "--base-ckpt", $BaseCkpt,
    "--work-dir", $WorkDir
)

if ($LocalRagJsonl -ne "") {
    $cmd += @("--local-rag-jsonl", $LocalRagJsonl)
}
if ($VitCkpt -ne "") {
    $cmd += @("--vit-ckpt", $VitCkpt)
}
if ($DryRun) {
    $cmd += "--dry-run"
}

Write-Host "Running: $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Length - 1)]
exit $LASTEXITCODE
