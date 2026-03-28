#!/usr/bin/env bash
# Build SDX native C++ (optional CUDA) and Rust tools when cargo exists.
# Usage: bash scripts/tools/native/build_native.sh
# SDX_BUILD_CUDA=0  force CPU-only C++
# SDX_BUILD_CUDA=1  force CUDA (requires nvcc)
# unset: auto - ON if command -v nvcc, else OFF
#
# If the previous configure used a different SDX_BUILD_CUDA, this script removes
# native/cpp/build so CMake does not reuse a stale cache.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CPP="$ROOT/native/cpp"
BUILD="$CPP/build"
if [[ "${SDX_BUILD_CUDA:-}" == "0" ]]; then
  USE_CUDA=OFF
elif [[ "${SDX_BUILD_CUDA:-}" == "1" ]]; then
  USE_CUDA=ON
elif command -v nvcc >/dev/null 2>&1; then
  USE_CUDA=ON
else
  USE_CUDA=OFF
  echo "(info) nvcc not on PATH - C++ build without CUDA. SDX_BUILD_CUDA=1 to force."
fi

line=""
CACHED=""
if [[ -f "$BUILD/CMakeCache.txt" ]]; then
  line="$(grep -E '^SDX_BUILD_CUDA:' "$BUILD/CMakeCache.txt" | head -1 || true)"
  case "$line" in
    SDX_BUILD_CUDA:*=ON*)
      CACHED=ON
      ;;
    SDX_BUILD_CUDA:*=OFF*)
      CACHED=OFF
      ;;
    "")
      ;;
    SDX_BUILD_CUDA:*)
      echo "(info) Clearing $BUILD (invalid SDX_BUILD_CUDA in CMakeCache: $line)"
      rm -rf "$BUILD"
      ;;
  esac
fi
if [[ -n "$CACHED" && "$CACHED" != "$USE_CUDA" ]]; then
  echo "(info) CMake cache had SDX_BUILD_CUDA=$CACHED but this run uses $USE_CUDA - removing $BUILD"
  rm -rf "$BUILD"
fi

echo "==> CMake configure ($CPP) SDX_BUILD_CUDA=$USE_CUDA"
cmake -S "$CPP" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DSDX_BUILD_CUDA="$USE_CUDA"
echo "==> CMake build"
# Visual Studio generator (typical on Windows): must pass --config Release.
case "$(uname -s 2>/dev/null || true)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    cmake --build "$BUILD" --config Release --parallel
    ;;
  *)
    cmake --build "$BUILD" --parallel
    ;;
esac
if command -v cargo >/dev/null 2>&1; then
  echo "==> cargo build: sdx-jsonl-tools"
  (cd "$ROOT/native/rust/sdx-jsonl-tools" && cargo build --release)
  echo "==> cargo build: sdx-noise-schedule"
  (cd "$ROOT/native/rust/sdx-noise-schedule" && cargo build --release)
else
  echo "(skip) cargo not found"
fi

echo "Done."
