#!/usr/bin/env python3
"""Ensure REPA vision encoder (DINOv3-L by default) exists under SDX_PRETRAINED."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DEFAULT_REPA = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _folder_for_model(model: str, dest_root: Path) -> Path:
    lower = model.lower()
    if "dinov3-vitl16" in lower or "dinov3-vit-l" in lower:
        return dest_root / "DINOv3-ViT-L16"
    if "dinov3-vitb16" in lower:
        return dest_root / "DINOv3-ViT-B16"
    if "dinov3-vits16" in lower:
        return dest_root / "DINOv3-ViT-S16"
    if "dinov2-base" in lower:
        return dest_root / "DINOv2-Base"
    if "dinov2-large" in lower:
        return dest_root / "DINOv2-Large"
    return dest_root / "REPA-encoder"


def main() -> int:
    p = argparse.ArgumentParser(description="Download REPA encoder if missing.")
    p.add_argument(
        "--model",
        default=os.environ.get("SDX_REPA_ENCODER", _DEFAULT_REPA),
        help="HF id or local path",
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("SDX_PRETRAINED", "/workspace/pretrained"),
        help="Pretrained root",
    )
    args = p.parse_args()
    model = args.model
    dest_root = Path(args.dest)
    if Path(model).is_dir():
        print(f"REPA encoder already local: {model}")
        return 0
    folder = _folder_for_model(model, dest_root)
    if folder.is_dir() and any(folder.iterdir()):
        print(f"REPA encoder OK: {folder}")
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed", file=sys.stderr)
        return 1
    print(f"Downloading REPA encoder {model} -> {folder}")
    snapshot_download(repo_id=model, local_dir=str(folder))
    print(f"Done: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
