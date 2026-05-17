#!/usr/bin/env bash
# Build SDX native C++ (optional CUDA) and Rust tools when cargo exists.
# Usage: bash scripts/tools/native/build_native.sh
#
# SDX_BUILD_CUDA=0|1 — unset auto-detects nvcc
# SDX_USE_NINJA=0 — do not use Ninja even if ninja is on PATH
# SDX_CUDA_ARCHITECTURES — passed to -DCMAKE_CUDA_ARCHITECTURES
# SDX_C_COMPILER_LAUNCHER / SDX_CXX_COMPILER_LAUNCHER / SDX_CUDA_COMPILER_LAUNCHER —
#   overrides (default: sccache, else ccache, when on PATH)
#
# Uses Ninja when available (faster single-config builds). Clears the build dir
# when SDX_BUILD_CUDA or the generator (Ninja vs default) no longer matches CMakeCache.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CPP="$ROOT/native/cpp"
BUILD="$CPP/build"
OLD_RUST_WRAPPER="${RUSTC_WRAPPER-}"

USE_NINJA=0
if [[ "${SDX_USE_NINJA:-}" != "0" ]] && command -v ninja >/dev/null 2>&1; then
  USE_NINJA=1
fi

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

LAUNCH_C="${SDX_C_COMPILER_LAUNCHER:-}"
LAUNCH_CXX="${SDX_CXX_COMPILER_LAUNCHER:-}"
LAUNCH_CUDA="${SDX_CUDA_COMPILER_LAUNCHER:-}"
if [[ -z "$LAUNCH_C" && -z "$LAUNCH_CXX" ]]; then
  if command -v sccache >/dev/null 2>&1; then
    LAUNCH_C=sccache
    LAUNCH_CXX=sccache
    if [[ "$USE_CUDA" == ON && -z "$LAUNCH_CUDA" ]]; then
      LAUNCH_CUDA=sccache
    fi
  elif command -v ccache >/dev/null 2>&1; then
    LAUNCH_C=ccache
    LAUNCH_CXX=ccache
    if [[ "$USE_CUDA" == ON && -z "$LAUNCH_CUDA" ]]; then
      LAUNCH_CUDA=ccache
    fi
  fi
fi

CACHED=""
GENERATOR_STALE=0
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
  gen_line="$(grep -E '^CMAKE_GENERATOR:INTERNAL=' "$BUILD/CMakeCache.txt" | head -1 || true)"
  if [[ -n "$gen_line" ]]; then
    gen_val="${gen_line#CMAKE_GENERATOR:INTERNAL=}"
    if [[ "$USE_NINJA" -eq 1 ]]; then
      if [[ "$gen_val" != "Ninja" ]]; then
        GENERATOR_STALE=1
      fi
    else
      if [[ "$gen_val" == "Ninja" ]]; then
        GENERATOR_STALE=1
      fi
    fi
  fi
fi

if [[ -n "$CACHED" && "$CACHED" != "$USE_CUDA" ]]; then
  echo "(info) CMake cache had SDX_BUILD_CUDA=$CACHED but this run uses $USE_CUDA - removing $BUILD"
  rm -rf "$BUILD"
elif [[ "$GENERATOR_STALE" -eq 1 ]] && [[ -d "$BUILD" ]]; then
  echo "(info) CMake generator changed (USE_NINJA=$USE_NINJA) - removing $BUILD"
  rm -rf "$BUILD"
fi

CMAKE_ARGS=()
if [[ "$USE_NINJA" -eq 1 ]]; then
  CMAKE_ARGS+=(-G Ninja)
fi
CMAKE_ARGS+=(
  -S "$CPP"
  -B "$BUILD"
  -DCMAKE_BUILD_TYPE=Release
  -DSDX_BUILD_CUDA="$USE_CUDA"
)
if [[ -n "$LAUNCH_C" ]]; then
  CMAKE_ARGS+=(-DCMAKE_C_COMPILER_LAUNCHER="$LAUNCH_C")
fi
if [[ -n "$LAUNCH_CXX" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER_LAUNCHER="$LAUNCH_CXX")
fi
if [[ "$USE_CUDA" == ON && -n "$LAUNCH_CUDA" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER_LAUNCHER="$LAUNCH_CUDA")
fi
if [[ -n "${SDX_CUDA_ARCHITECTURES:-}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$SDX_CUDA_ARCHITECTURES")
fi
case "$(uname -s 2>/dev/null || true)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    if [[ "$LAUNCH_CXX" == "sccache" ]]; then
      CMAKE_ARGS+=(-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded)
    fi
    ;;
esac

if [[ "$USE_NINJA" -eq 1 ]]; then
  echo "==> CMake configure (Ninja) $CPP SDX_BUILD_CUDA=$USE_CUDA"
else
  echo "==> CMake configure (default generator) $CPP SDX_BUILD_CUDA=$USE_CUDA"
fi
if [[ -n "$LAUNCH_CXX" ]]; then
  if [[ "$USE_CUDA" == ON && -n "$LAUNCH_CUDA" ]]; then
    echo "    CXX launcher: $LAUNCH_CXX  CUDA launcher: $LAUNCH_CUDA"
  else
    echo "    CXX launcher: $LAUNCH_CXX"
  fi
fi

cmake "${CMAKE_ARGS[@]}"

echo "==> CMake build"
if [[ "$USE_NINJA" -eq 1 ]]; then
  cmake --build "$BUILD" --parallel
else
  case "$(uname -s 2>/dev/null || true)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
      cmake --build "$BUILD" --config Release --parallel
      ;;
    *)
      cmake --build "$BUILD" --parallel
      ;;
  esac
fi

if [[ -z "${RUSTC_WRAPPER:-}" ]] && command -v sccache >/dev/null 2>&1; then
  export RUSTC_WRAPPER=sccache
  echo "(info) RUSTC_WRAPPER=sccache for cargo"
fi

if command -v cargo >/dev/null 2>&1; then
  echo "==> cargo build: sdx-jsonl-tools"
  (cd "$ROOT/native/rust/sdx-jsonl-tools" && cargo build --release)
  echo "==> cargo build: sdx-noise-schedule"
  (cd "$ROOT/native/rust/sdx-noise-schedule" && cargo build --release)
  echo "==> cargo build: sdx-diffusion-math"
  (cd "$ROOT/native/rust/sdx-diffusion-math" && cargo build --release)
  echo "==> cargo build: sdx-image-metrics"
  (cd "$ROOT/native/rust/sdx-image-metrics" && cargo build --release)
else
  echo "(skip) cargo not found"
fi

if [[ -z "$OLD_RUST_WRAPPER" ]]; then
  unset RUSTC_WRAPPER
else
  export RUSTC_WRAPPER="$OLD_RUST_WRAPPER"
fi

echo "Done."
