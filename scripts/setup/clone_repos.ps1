# Clone reference repos used by or cited in SDX (optional; runtime deps are pip-only).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$External = Join-Path $Root "external"
if (-not (Test-Path $External)) { New-Item -ItemType Directory -Path $External | Out-Null }

# Meta DiT (reference implementation cited in README)
$DitPath = Join-Path $External "DiT"
if (-not (Test-Path $DitPath)) {
  Write-Host "Cloning facebookresearch/DiT..."
  git clone --depth 1 https://github.com/facebookresearch/DiT.git $DitPath
  Write-Host "Done: $DitPath"
} else { Write-Host "Already exists: $DitPath" }

# ControlNet: structural conditioning reference
$ControlNetPath = Join-Path $External "ControlNet"
if (-not (Test-Path $ControlNetPath)) {
  Write-Host "Cloning lllyasviel/ControlNet..."
  git clone --depth 1 https://github.com/lllyasviel/ControlNet.git $ControlNetPath
  Write-Host "Done: $ControlNetPath"
} else { Write-Host "Already exists: $ControlNetPath" }

# FLUX: modern diffusion reference (Black Forest Labs)
$FluxPath = Join-Path $External "flux"
if (-not (Test-Path $FluxPath)) {
  Write-Host "Cloning black-forest-labs/flux..."
  git clone --depth 1 https://github.com/black-forest-labs/flux.git $FluxPath
  Write-Host "Done: $FluxPath"
} else { Write-Host "Already exists: $FluxPath" }

# Stability generative-models: SD3 / official reference
$GenPath = Join-Path $External "generative-models"
if (-not (Test-Path $GenPath)) {
  Write-Host "Cloning stability-AI/generative-models..."
  git clone --depth 1 https://github.com/Stability-AI/generative-models.git $GenPath
  Write-Host "Done: $GenPath"
} else { Write-Host "Already exists: $GenPath" }

# PixArt-alpha: T5 + DiT, efficient text-to-image (PixArt-alpha evolution)
$PixArtAlphaPath = Join-Path $External "PixArt-alpha"
if (-not (Test-Path $PixArtAlphaPath)) {
  Write-Host "Cloning PixArt-alpha/PixArt-alpha..."
  git clone --depth 1 https://github.com/PixArt-alpha/PixArt-alpha.git $PixArtAlphaPath
  Write-Host "Done: $PixArtAlphaPath"
} else { Write-Host "Already exists: $PixArtAlphaPath" }

# PixArt-sigma: 4K T2I, weak-to-strong DiT
$PixArtSigmaPath = Join-Path $External "PixArt-sigma"
if (-not (Test-Path $PixArtSigmaPath)) {
  Write-Host "Cloning PixArt-alpha/PixArt-sigma..."
  git clone --depth 1 https://github.com/PixArt-alpha/PixArt-sigma.git $PixArtSigmaPath
  Write-Host "Done: $PixArtSigmaPath"
} else { Write-Host "Already exists: $PixArtSigmaPath" }

# Z-Image: S3-DiT single-stream diffusion transformer (Tongyi-MAI)
$ZImagePath = Join-Path $External "Z-Image"
if (-not (Test-Path $ZImagePath)) {
  Write-Host "Cloning Tongyi-MAI/Z-Image..."
  git clone --depth 1 https://github.com/Tongyi-MAI/Z-Image.git $ZImagePath
  Write-Host "Done: $ZImagePath"
} else { Write-Host "Already exists: $ZImagePath" }

# SiT: Scalable Interpolant Transformers, flow matching + DiT backbone
$SiTPath = Join-Path $External "SiT"
if (-not (Test-Path $SiTPath)) {
  Write-Host "Cloning willisma/SiT..."
  git clone --depth 1 https://github.com/willisma/SiT.git $SiTPath
  Write-Host "Done: $SiTPath"
} else { Write-Host "Already exists: $SiTPath" }

# Lumina-T2X: Lumina-T2I and Next-DiT, next-gen DiT scaling (Alpha-VLLM)
$LuminaPath = Join-Path $External "Lumina-T2X"
if (-not (Test-Path $LuminaPath)) {
  Write-Host "Cloning Alpha-VLLM/Lumina-T2X..."
  git clone --depth 1 https://github.com/Alpha-VLLM/Lumina-T2X.git $LuminaPath
  Write-Host "Done: $LuminaPath"
} else { Write-Host "Already exists: $LuminaPath" }

Write-Host "Reference repos are in $External. SDX does not import them; use pip install -r requirements.txt for runtime deps."
