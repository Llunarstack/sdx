#!/usr/bin/env bash
# Bootstrap a RunPod (or any Ubuntu CUDA) pod for SDX training.
#
#   cd /workspace/sdx && bash runpod/setup.sh
#
# Installs: system packages (apt), all Python deps, CUDA wheels if needed,
# optional native C++/Rust builds, and wires /workspace paths.
#
# Skip flags:
#   SDX_SKIP_CUDA_WHEELS=1   keep existing torch build
#   SDX_SKIP_NATIVE=1        skip C++/Rust/maturin builds
#   SDX_SKIP_EDITABLE_INSTALL=1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
# shellcheck source=/dev/null
source "$ROOT/runpod/lib.sh"
sdx_load_hf_token || true

# System packages for SDX on Ubuntu/Debian RunPod images.
# Safe to re-run. Skips quietly when apt-get is unavailable (non-root / non-Debian).
install_system_deps() (
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "(skip) apt-get not found — not a Debian/Ubuntu image."
    exit 0
  fi

  export DEBIAN_FRONTEND=noninteractive

  PACKAGES=(
    # Core tooling
    ca-certificates
    curl
    git
    wget
    nano
    # OpenCV (cv2) runtime
    libgl1
    libglib2.0-0
    # OCR (pytesseract)
    tesseract-ocr
    # Video pipelines (ffmpeg / ffprobe)
    ffmpeg
    # Native C++/CUDA build (scripts/tools/native/build_native.sh)
    build-essential
    cmake
    ninja-build
    pkg-config
    python3-dev
    libssl-dev
    # Rust CLI tools (sdx-jsonl-tools, etc.)
    rustc
    cargo
  )

  echo "==> apt-get install (${#PACKAGES[@]} packages)"
  apt-get update -qq
  apt-get install -y -qq "${PACKAGES[@]}" || {
    echo "WARN: apt install failed (non-root?). Install manually:" >&2
    printf '  %s\n' "${PACKAGES[@]}" >&2
    exit 0
  }

  echo "System dependencies OK."
)

# Every Python dependency SDX needs on RunPod.
install_python_deps() (
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
    python -m pip install -e ".[demo]" || \
      echo "WARN: editable install failed (non-fatal; run python train.py from repo root)."
  fi

  echo "Python dependencies OK."
)

# Optional native accelerators (C++/CUDA + Rust; see native/README.md).
# Skipped when SDX_SKIP_NATIVE=1 or build tools are missing.
install_native() (
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

  echo "Native build step done."
)

# Install secrets: bundled runpod/secret.txt -> /workspace/secret.txt (never committed)
sdx_install_scrape_secrets || true
if sdx_scrape_secrets_ok 2>/dev/null; then
  echo "Scrape secrets OK: ${SDX_SECRETS_FILE:-/workspace/secret.txt}"
else
  echo "WARN: danbooru/rule34xxx credentials missing — upload to $ROOT/runpod/secret.txt"
  echo "      then: bash runpod/sdx.sh secrets"
fi

install_system_deps
install_python_deps
install_native

mkdir -p "$HF_HOME" "$SDX_PRETRAINED" "$SDX_DATA" "$SDX_RESULTS"

# train.py resolves models under <repo>/pretrained — symlink to the network volume.
if [ "$(readlink -f "$ROOT/pretrained" 2>/dev/null || true)" != "$(readlink -f "$SDX_PRETRAINED" 2>/dev/null || echo "$SDX_PRETRAINED")" ]; then
  rm -rf "$ROOT/pretrained"
  ln -sfn "$SDX_PRETRAINED" "$ROOT/pretrained"
  echo "Linked $ROOT/pretrained -> $SDX_PRETRAINED"
fi

grep -qxF "source $ROOT/runpod/env.defaults" ~/.bashrc 2>/dev/null || \
  echo "source $ROOT/runpod/env.defaults" >> ~/.bashrc

bash "$ROOT/runpod/verify_env.sh"

echo
echo "Setup complete. Next steps:"
echo "  bash runpod/test.sh --skip-train"
echo "  bash runpod/download.sh"
echo "  bash runpod/train.sh"
