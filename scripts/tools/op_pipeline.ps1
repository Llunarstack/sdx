<#
OP pipeline (Windows-friendly):
1) Run dataset preflight coverage checks
2) Normalize/boost captions into a training-ready manifest
3) Optionally run training (if -TrainArgs is provided)
4) Optionally run quick qualitative eval on a checkpoint

Examples:
  .\scripts\tools\op_pipeline.ps1 -ManifestIn .\data\manifest.jsonl
  .\scripts\tools\op_pipeline.ps1 -ManifestIn .\data\manifest.jsonl -TrainArgs @("--passes","2","--image-size","512","--model","DiT-XL/2-Text")
  .\scripts\tools\op_pipeline.ps1 -ManifestIn .\data\manifest.jsonl -EvalCkpt .\results\run1\best.pt -Preset sdxl -OpMode portrait
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestIn,

    [Parameter()]
    [string]$WorkDir = "op_pipeline_run",

    [Parameter()]
    [string[]]$TrainArgs = @(),

    [Parameter()]
    [string]$EvalCkpt = "",

    [Parameter()]
    [string]$Preset = "",

    [Parameter()]
    [string]$OpMode = "",

    [Parameter()]
    [float]$MinHardStyle = 0.10,

    [Parameter()]
    [float]$MinPerson = 0.20,

    [Parameter()]
    [float]$MinSpatial = 0.03,

    [Parameter()]
    [float]$MinAnatomy = 0.10,

    [Parameter()]
    [float]$MinConceptBleed = 0.02
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$null = $repoRoot

$manifestInAbs = Resolve-Path $ManifestIn
$workAbs = Resolve-Path $WorkDir
if (-not $workAbs) { $workAbs = (Join-Path (Get-Location) $WorkDir) ; New-Item -ItemType Directory -Force -Path $workAbs | Out-Null }

$normManifest = Join-Path $workAbs "manifest_norm.jsonl"
$preflightReport = Join-Path $workAbs "preflight.json"

Write-Host "OP pipeline: preflight coverage..."
$preflightArgs = @(
    "-m", "scripts.tools", "op_preflight",
    "--manifest", $manifestInAbs,
    "--out", $preflightReport,
    "--min-hard-style", $MinHardStyle.ToString(),
    "--min-person", $MinPerson.ToString(),
    "--min-spatial", $MinSpatial.ToString(),
    "--min-anatomy", $MinAnatomy.ToString(),
    "--min-concept-bleed", $MinConceptBleed.ToString()
)
& python @($preflightArgs) | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Preflight failed. Fix dataset coverage and rerun." }

Write-Host "OP pipeline: normalize/boost captions..."
& python -m scripts.tools.normalize_captions --in $manifestInAbs --out $normManifest | Out-Host
if ($LASTEXITCODE -ne 0) { throw "normalize_captions failed." }

if ($TrainArgs.Count -gt 0) {
    Write-Host "OP pipeline: training..."
    $trainCmd = @("train.py")
    $trainCmd += $TrainArgs

    # Ensure train.py has a manifest input.
    # If user already provided --manifest-jsonl, we won't override.
    $hasManifest = ($TrainArgs -contains "--manifest-jsonl")
    if (-not $hasManifest) {
        $trainCmd += @("--manifest-jsonl", $normManifest)
    }

    & python $trainCmd | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Training failed." }
}

if ($EvalCkpt -and $EvalCkpt.Trim().Length -gt 0) {
    Write-Host "OP pipeline: eval prompts..."
    $evalOutDir = Join-Path $workAbs "eval_prompts"

    $evalArgs = @(
        "-m", "scripts.tools.eval_prompts",
        "--ckpt", $EvalCkpt,
        "--out-dir", $evalOutDir
    )
    if ($Preset) { $evalArgs += @("--preset", $Preset) }
    if ($OpMode) { $evalArgs += @("--op-mode", $OpMode) }

    & python @($evalArgs) | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Eval failed." }
}

Write-Host "OP pipeline finished."

