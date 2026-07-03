#!/usr/bin/env bash
# Build optional native accelerators (C++/CUDA + Rust CLI tools + PyO3 module).
# Skipped when SDX_SKIP_NATIVE=1 or build tools are missing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ "${SDX_SKIP_NATIVE:-0}" = "1" ]; then
  echo "(skip) SDX_SKIP_NATIVE=1"
  exit 0
fi

if command -v cmake >/dev/null 2>&1; then
  echo "==> Native C++/CUDA (build_native.sh)"
  if [ "$(uname -s 2>/dev/null || echo unknown)" = "Linux" ]; then
    sed -i 's/\r$//' scripts/tools/native/build_native.sh 2>/dev/null || true
  fi
  bash scripts/tools/native/build_native.sh || echo "WARN: native C++/CUDA build failed (non-fatal)."
else
  echo "(skip) cmake not found — native C++ build skipped."
fi

if command -v cargo >/dev/null 2>&1 && python -c "import maturin" 2>/dev/null; then
  if [ -f "$ROOT/native/rust/Cargo.toml" ]; then
    echo "==> maturin develop (sdx-native PyO3)"
    (cd "$ROOT/native/rust" && maturin develop --release) || \
      echo "WARN: maturin develop failed (non-fatal; pure-Python paths still work)."
  fi
else
  echo "(skip) cargo or maturin missing — PyO3 native module not built."
fi

echo "Native build step done."
