#!/usr/bin/env bash
# System packages for SDX on Ubuntu/Debian RunPod images.
# Safe to re-run. Skips quietly when apt-get is unavailable (non-root / non-Debian).
set -euo pipefail

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
