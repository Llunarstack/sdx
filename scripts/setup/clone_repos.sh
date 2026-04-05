#!/usr/bin/env bash
# Clone reference repos used by or cited in SDX (optional; runtime deps are pip-only).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL="${ROOT}/external"
mkdir -p "$EXTERNAL"

# Meta DiT (reference implementation cited in README)
if [ ! -d "${EXTERNAL}/DiT" ]; then
  echo "Cloning facebookresearch/DiT..."
  git clone --depth 1 https://github.com/facebookresearch/DiT.git "${EXTERNAL}/DiT"
  echo "Done: ${EXTERNAL}/DiT"
else echo "Already exists: ${EXTERNAL}/DiT"; fi

# ControlNet: structural conditioning reference
if [ ! -d "${EXTERNAL}/ControlNet" ]; then
  echo "Cloning lllyasviel/ControlNet..."
  git clone --depth 1 https://github.com/lllyasviel/ControlNet.git "${EXTERNAL}/ControlNet"
  echo "Done: ${EXTERNAL}/ControlNet"
else echo "Already exists: ${EXTERNAL}/ControlNet"; fi

# FLUX: modern diffusion reference (Black Forest Labs)
if [ ! -d "${EXTERNAL}/flux" ]; then
  echo "Cloning black-forest-labs/flux..."
  git clone --depth 1 https://github.com/black-forest-labs/flux.git "${EXTERNAL}/flux"
  echo "Done: ${EXTERNAL}/flux"
else echo "Already exists: ${EXTERNAL}/flux"; fi

# Stability generative-models: SD3 reference
if [ ! -d "${EXTERNAL}/generative-models" ]; then
  echo "Cloning Stability-AI/generative-models..."
  git clone --depth 1 https://github.com/Stability-AI/generative-models.git "${EXTERNAL}/generative-models"
  echo "Done: ${EXTERNAL}/generative-models"
else echo "Already exists: ${EXTERNAL}/generative-models"; fi

# PixArt-alpha: T5 + DiT, efficient text-to-image
if [ ! -d "${EXTERNAL}/PixArt-alpha" ]; then
  echo "Cloning PixArt-alpha/PixArt-alpha..."
  git clone --depth 1 https://github.com/PixArt-alpha/PixArt-alpha.git "${EXTERNAL}/PixArt-alpha"
  echo "Done: ${EXTERNAL}/PixArt-alpha"
else echo "Already exists: ${EXTERNAL}/PixArt-alpha"; fi

# PixArt-sigma: 4K T2I, weak-to-strong DiT
if [ ! -d "${EXTERNAL}/PixArt-sigma" ]; then
  echo "Cloning PixArt-alpha/PixArt-sigma..."
  git clone --depth 1 https://github.com/PixArt-alpha/PixArt-sigma.git "${EXTERNAL}/PixArt-sigma"
  echo "Done: ${EXTERNAL}/PixArt-sigma"
else echo "Already exists: ${EXTERNAL}/PixArt-sigma"; fi

# Z-Image: S3-DiT single-stream diffusion transformer (Tongyi-MAI)
if [ ! -d "${EXTERNAL}/Z-Image" ]; then
  echo "Cloning Tongyi-MAI/Z-Image..."
  git clone --depth 1 https://github.com/Tongyi-MAI/Z-Image.git "${EXTERNAL}/Z-Image"
  echo "Done: ${EXTERNAL}/Z-Image"
else echo "Already exists: ${EXTERNAL}/Z-Image"; fi

# SiT: Scalable Interpolant Transformers, flow matching + DiT backbone
if [ ! -d "${EXTERNAL}/SiT" ]; then
  echo "Cloning willisma/SiT..."
  git clone --depth 1 https://github.com/willisma/SiT.git "${EXTERNAL}/SiT"
  echo "Done: ${EXTERNAL}/SiT"
else echo "Already exists: ${EXTERNAL}/SiT"; fi

# Lumina-T2X: Lumina-T2I and Next-DiT (Alpha-VLLM)
if [ ! -d "${EXTERNAL}/Lumina-T2X" ]; then
  echo "Cloning Alpha-VLLM/Lumina-T2X..."
  git clone --depth 1 https://github.com/Alpha-VLLM/Lumina-T2X.git "${EXTERNAL}/Lumina-T2X"
  echo "Done: ${EXTERNAL}/Lumina-T2X"
else echo "Already exists: ${EXTERNAL}/Lumina-T2X"; fi

echo "Reference repos are in ${EXTERNAL}. SDX does not import them; use pip install -r requirements.txt for runtime deps."
