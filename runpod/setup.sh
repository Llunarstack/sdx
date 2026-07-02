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

if [ -f "$SDX_SECRETS_FILE" ]; then
  echo "Secrets: $SDX_SECRETS_FILE"
else
  echo "WARN: secrets not found at $SDX_SECRETS_FILE — upload secret.txt before downloading datasets."
fi

bash "$ROOT/runpod/lib/install_system_deps.sh"
bash "$ROOT/runpod/lib/install_python_deps.sh"
bash "$ROOT/runpod/lib/install_native.sh"

mkdir -p "$HF_HOME" "$SDX_PRETRAINED" "$SDX_DATA" "$SDX_RESULTS"

# train.py resolves models under <repo>/pretrained — symlink to the network volume.
if [ "$(readlink -f "$ROOT/pretrained" 2>/dev/null || true)" != "$(readlink -f "$SDX_PRETRAINED" 2>/dev/null || echo "$SDX_PRETRAINED")" ]; then
  rm -rf "$ROOT/pretrained"
  ln -sfn "$SDX_PRETRAINED" "$ROOT/pretrained"
  echo "Linked $ROOT/pretrained -> $SDX_PRETRAINED"
fi

grep -qxF "source $ROOT/runpod/env.defaults" ~/.bashrc 2>/dev/null || \
  echo "source $ROOT/runpod/env.defaults" >> ~/.bashrc

bash "$ROOT/runpod/lib/verify_env.sh"

echo
echo "Setup complete. Next steps:"
echo "  bash runpod/test.sh --skip-train"
echo "  bash runpod/download.sh"
echo "  bash runpod/train.sh"
