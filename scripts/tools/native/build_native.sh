#!/usr/bin/env bash
# Build SDX native C++ (optional CUDA) and Rust JSONL tools when cargo exists.
# Usage: bash scripts/tools/native/build_native.sh
# SDX_BUILD_CUDA=0  — CPU-only C++ (no sdx_cuda_hwc_to_chw)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CPP="$ROOT/native/cpp"
BUILD="$CPP/build"
USE_CUDA="${SDX_BUILD_CUDA:-ON}"
if [[ "${SDX_BUILD_CUDA:-}" == "0" ]]; then USE_CUDA=OFF; fi

echo "==> CMake configure ($CPP) SDX_BUILD_CUDA=$USE_CUDA"
cmake -S "$CPP" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DSDX_BUILD_CUDA="$USE_CUDA"
echo "==> CMake build"
cmake --build "$BUILD" --parallel

if command -v cargo >/dev/null 2>&1; then
  echo "==> cargo build: sdx-jsonl-tools"
  (cd "$ROOT/native/rust/sdx-jsonl-tools" && cargo build --release)
else
  echo "(skip) cargo not found"
fi

echo "Done."
