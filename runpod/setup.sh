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
source "$ROOT/runpod/lib/load_secrets.sh"
sdx_load_hf_token || true

# Install secrets: bundled runpod/secret.txt -> /workspace/secret.txt (never committed)
if [ ! -f "$SDX_SECRETS_FILE" ] && [ -f "$ROOT/runpod/secret.txt" ]; then
  cp "$ROOT/runpod/secret.txt" "$SDX_SECRETS_FILE"
  echo "Installed secrets: $ROOT/runpod/secret.txt -> $SDX_SECRETS_FILE"
elif [ -f "$SDX_SECRETS_FILE" ] && [ -f "$ROOT/runpod/secret.txt" ]; then
  python3 - <<'PY' 2>/dev/null || true
import os, shutil, sys
sys.path.insert(0, os.environ["SDX_ROOT"])
from scripts.scrape.secrets_config import parse_secrets_file
from pathlib import Path
dest = Path(os.environ["SDX_SECRETS_FILE"])
bundled = Path(os.environ["SDX_ROOT"]) / "runpod" / "secret.txt"
need = {"danbooru", "rule34xxx"}
try:
    have = set(parse_secrets_file(dest).keys())
except Exception:
    have = set()
if bundled.is_file() and not need.intersection(have):
    shutil.copy(bundled, dest)
    print(f"Installed scrape credentials (workspace file had no booru sites): {bundled} -> {dest}")
PY
fi

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
