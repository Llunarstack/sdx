#!/usr/bin/env bash
# CUDA allocator fragmentation: expandable segments — must export before Python imports torch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
cd "$ROOT"
exec python train.py "$@"
