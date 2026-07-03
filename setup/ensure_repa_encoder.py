#!/usr/bin/env python3
"""Ensure REPA vision encoder (DINOv2-base by default) exists under SDX_PRETRAINED."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Download REPA encoder if missing.")
    p.add_argument(
        "--model",
        default=os.environ.get("SDX_REPA_ENCODER", "facebook/dinov2-base"),
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
    folder = dest_root / "DINOv2-Base" if "dinov2-base" in model.lower() else dest_root / "REPA-encoder"
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
