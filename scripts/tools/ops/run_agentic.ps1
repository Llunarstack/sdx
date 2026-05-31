# One-click Agentic Stack (Windows PowerShell).
# Usage:
#   .\scripts\tools\ops\run_agentic.ps1 -Ckpt results\best.pt -Prompt "neon alley at night"

param(
    [Parameter(Mandatory = $true)]
    [string]$Ckpt,
    [Parameter(Mandatory = $true)]
    [string]$Prompt,
    [string]$Out = "agent_out.png",
    [string]$WorkDir = "agentic_run",
    [string]$LocalRagJsonl = "",
    [string]$VitCkpt = "",
    [string]$Mode = "generate",
    [int]$Variants = 3,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $RepoRoot

$cmdName = switch ($Mode.ToLower()) {
    "evolve" { "agentic_evolve" }
    "flywheel" { "agentic_flywheel" }
    "roles" { "agentic_roles" }
    default { "agentic_generate" }
}

$cmd = @(
    "python", "-m", "scripts.tools", $cmdName,
    "--ckpt", $Ckpt,
    "--prompt", $Prompt,
    "--work-dir", $WorkDir
)

if ($Mode.ToLower() -eq "flywheel") {
    $cmd = @(
        "python", "-m", "scripts.tools", "agentic_flywheel",
        "--base-ckpt", $Ckpt,
        "--prompt", $Prompt,
        "--work-dir", $WorkDir
    )
}

if ($Out -ne "" -and $cmdName -ne "agentic_flywheel") {
    $cmd += @("--out", $Out)
}
if ($LocalRagJsonl -ne "") {
    $cmd += @("--local-rag-jsonl", $LocalRagJsonl)
}
if ($VitCkpt -ne "") {
    $cmd += @("--vit-ckpt", $VitCkpt)
}
if ($Mode.ToLower() -eq "evolve") {
    $cmd += @("--variants", "$Variants")
}
if ($DryRun) {
    $cmd += "--dry-run"
}

Write-Host "Running: $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Length - 1)]
exit $LASTEXITCODE
