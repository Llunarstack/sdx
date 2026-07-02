#!/usr/bin/env bash
# Post-install sanity checks for RunPod.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> SDX environment health"
python -m toolkit.training.env_health || true

echo
echo "==> Import smoke test"
python - <<'PY'
import importlib
import sys

mods = [
    "torch",
    "torchvision",
    "transformers",
    "diffusers",
    "xformers",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "PIL",
    "numpy",
    "scipy",
    "cv2",
    "safetensors",
    "omegaconf",
    "tqdm",
    "requests",
    "rich",
    "orjson",
    "xxhash",
    "humanize",
    "psutil",
]
optional = {"wandb", "tensorboard", "xformers"}
failed = []
optional_missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except ImportError as e:
        if name in optional:
            optional_missing.append((name, str(e)))
        else:
            failed.append((name, str(e)))

if optional_missing:
    print("Optional imports missing (non-fatal):")
    for name, err in optional_missing:
        print(f"  {name}: {err}")

if failed:
    print("MISSING optional/core imports:", file=sys.stderr)
    for name, err in failed:
        print(f"  {name}: {err}", file=sys.stderr)
    sys.exit(1)

import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  device={torch.cuda.get_device_name(0)}")
else:
    print()
print("All smoke-test imports OK.")
PY

echo
echo "==> System binaries"
for bin in git ffmpeg ffprobe tesseract cmake cargo; do
  if command -v "$bin" >/dev/null 2>&1; then
    echo "  $bin: $(command -v "$bin")"
  else
    echo "  $bin: (not found)"
  fi
done
