#!/usr/bin/env bash
# One-shot environment bootstrap for a RunPod (or any Ubuntu CUDA) pod.
#
#   bash scripts/setup/runpod_setup.sh
#
# Assumes a PyTorch base image (torch already installed with CUDA). If torch is
# missing or CPU-only, follow up with:
#
#   pip install --force-reinstall -r requirements-cuda128.txt
set -euo pipefail

cd "$(dirname "$0")/../.."

# opencv-python needs libGL/libglib at import time; slim CUDA images omit them.
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 || \
        echo "WARN: apt install failed (non-root?). If 'import cv2' fails, install libgl1 + libglib2.0-0."
fi

pip install -r requirements.txt

# Persist HF model downloads (T5, VAE, CLIP) on the network volume so they
# survive pod restarts. Only set if not already configured by the pod template.
if [ -d /workspace ] && [ -z "${HF_HOME:-}" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    grep -qxF 'export HF_HOME=/workspace/.cache/huggingface' ~/.bashrc 2>/dev/null || \
        echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
    echo "HF_HOME set to /workspace/.cache/huggingface (persists across pod restarts)."
fi

echo
echo "=== Verifying environment ==="
python -m toolkit.training.env_health

echo
echo "Setup complete. Train with e.g.:"
echo "  python train.py --data-path /workspace/images --flow-matching-training --epochs 20 --results-dir /workspace/results"
