#!/usr/bin/env python3
"""
Print environment summary for SDX training / inference debugging.

Usage (repo root):
  python -m toolkit.training.env_health
  python -m toolkit.training.env_health --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from typing import Any

# Optional packages worth knowing about in a bug report (see toolkit/extras/requirements-suggested.txt).
_OPTIONAL_MODULES = (
    "timm",
    "xformers",
    "diffusers",
    "transformers",
    "xxhash",
    "rich",
    "humanize",
    "orjson",
    "psutil",
    "scipy",
    "cv2",
    "wandb",
    "tensorboard",
)


def _version(mod_name: str) -> str | None:
    if importlib.util.find_spec(mod_name) is None:
        return None
    try:
        return str(getattr(__import__(mod_name), "__version__", "installed"))
    except Exception as e:
        return f"import failed: {e}"


def collect_env() -> dict[str, Any]:
    out: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
            out["cuda_version"] = getattr(torch.version, "cuda", None)
        out["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        if "+cpu" in str(out["torch"]) and not out["cuda_available"]:
            out["cuda_wheel_hint"] = (
                "pip install --force-reinstall -r requirements-cuda128.txt  # GPU: replace CPU torch (+cu128)"
            )
    except ImportError:
        out["torch"] = None

    out["triton"] = _version("triton")
    if out["triton"] is None:
        if sys.platform == "win32":
            out["triton_hint"] = "pip install triton-windows  # xformers expects import triton on Windows"
        elif sys.platform == "linux":
            out["triton_hint"] = "pip install triton  # optional; many CUDA stacks bundle it with torch"
    out["optional_libs"] = {m: _version(m) for m in _OPTIONAL_MODULES}

    try:
        from utils.native import native_stack_status

        out["sdx_native"] = native_stack_status()
    except Exception as e:
        out["sdx_native"] = {"error": str(e)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="SDX environment health report")
    p.add_argument("--json", action="store_true", help="Print single JSON object")
    args = p.parse_args()
    data = collect_env()
    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0
    print("=== SDX env_health ===")
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sk, sv in v.items():
                print(f"  {sk}: {sv}")
        else:
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
