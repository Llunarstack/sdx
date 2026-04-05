#!/usr/bin/env python3
"""
Print environment summary for SDX training / inference debugging.

Usage (repo root):
  python -m toolkit.training.env_health
  python -m toolkit.training.env_health --json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict


def _bootstrap_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    for sub in ("native/python",):
        p = root / sub
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def collect_env() -> Dict[str, Any]:
    _bootstrap_paths()
    out: Dict[str, Any] = {
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
        if isinstance(out["torch"], str) and "+cpu" in out["torch"] and not out["cuda_available"]:
            out["cuda_wheel_hint"] = (
                "pip install --force-reinstall -r requirements-cuda128.txt  # GPU: replace CPU torch (+cu128)"
            )
    except ImportError:
        out["torch"] = None

    # Optional stack
    for mod, label in (
        ("timm", "timm"),
        ("xformers", "xformers"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
    ):
        try:
            m = __import__(mod)
            out[label] = getattr(m, "__version__", "installed")
        except ImportError:
            out[label] = None

    try:
        import triton

        out["triton"] = getattr(triton, "__version__", "installed")
    except ImportError:
        out["triton"] = None
        if sys.platform == "win32":
            out["triton_hint"] = "pip install triton-windows  # xformers expects import triton on Windows"
        elif sys.platform == "linux":
            out["triton_hint"] = "pip install triton  # optional; many CUDA stacks bundle it with torch"

    try:
        from toolkit.libs.optional_imports import describe_optional_libs

        out["optional_libs"] = describe_optional_libs()
    except Exception:
        out["optional_libs"] = {}

    try:
        from sdx_native.native_tools import native_stack_status

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
        if k == "sdx_native" and isinstance(v, dict):
            print(f"{k}:")
            for sk, sv in v.items():
                print(f"  {sk}: {sv}")
        else:
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
