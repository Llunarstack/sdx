#!/usr/bin/env python3
"""
Remove optional/redundant model folders from model/ to free disk space.
Keeps: T5-XXL, sd-vae-ft-mse, sdxl-vae-fp16-fix, SmolLM2-360M-Instruct.
Removes: T5-XL, T5-Large, sd-vae-ft-ema, sdxl-vae, CLIP-ViT-L-14, Qwen2.5-7B-Instruct.

Close any app using these models (IDE, Python, Hugging Face) before running.
"""

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR_DEFAULT = os.path.join(ROOT, "model")

# Folders we can remove (redundant or optional); keep one T5 (XXL), two VAEs (mse + sdxl-fp16), one LLM (SmolLM).
FOLDERS_TO_REMOVE = [
    "T5-XL",
    "T5-Large",
    "sd-vae-ft-ema",
    "sdxl-vae",
    "CLIP-ViT-L-14",
    "Qwen2.5-7B-Instruct",
]


def main():
    parser = argparse.ArgumentParser(description="Remove optional model folders to free space.")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT, help="model/ path")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be removed")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Model dir not found: {args.model_dir}", file=sys.stderr)
        return 1

    removed = 0
    for name in FOLDERS_TO_REMOVE:
        path = os.path.join(args.model_dir, name)
        if not os.path.isdir(path):
            continue
        if args.dry_run:
            print(f"Would remove: {path}")
            removed += 1
            continue
        try:
            shutil.rmtree(path)
            print(f"Removed: {path}")
            removed += 1
        except OSError as e:
            print(f"Could not remove {path}: {e}", file=sys.stderr)

    if removed == 0 and not args.dry_run:
        print("No optional model folders found to remove.")
    elif removed:
        print(f"Freed space by removing {removed} folder(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
