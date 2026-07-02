#!/usr/bin/env bash
# Install every Python dependency SDX needs on RunPod.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> pip upgrade"
python -m pip install -U pip wheel setuptools

echo "==> pip install runpod stack"
python -m pip install -r runpod/requirements-runpod.txt

# RunPod images often ship CPU-only torch from PyPI, or no torch at all.
# Install CUDA 12.8 wheels when GPU stack is missing or CPU-only.
need_cuda_wheels=0
if ! python -c "import torch" 2>/dev/null; then
  need_cuda_wheels=1
elif ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  need_cuda_wheels=1
fi

if [ "${SDX_SKIP_CUDA_WHEELS:-0}" = "1" ]; then
  echo "(skip) SDX_SKIP_CUDA_WHEELS=1 — not installing requirements-cuda128.txt"
elif [ "$need_cuda_wheels" = "1" ]; then
  echo "==> Installing CUDA 12.8 PyTorch / xformers wheels"
  python -m pip install --force-reinstall -r requirements-cuda128.txt
else
  echo "CUDA PyTorch already present: $(python -c 'import torch; print(torch.__version__)')"
fi

# Editable install registers console scripts (sdx-demo, sdx-sample) without duplicating deps.
if [ "${SDX_SKIP_EDITABLE_INSTALL:-0}" != "1" ]; then
  echo "==> pip install -e .[demo]"
  python -m pip install -e ".[demo]"
fi

echo "Python dependencies OK."
