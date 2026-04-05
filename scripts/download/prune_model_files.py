#!/usr/bin/env python3
"""
Remove redundant files inside model/ folders to free space. Does NOT remove entire folders.

- Removes .cache/ in every model subfolder (Hugging Face cache; not needed when loading from local_dir).
- VAE folders: if both .bin and .safetensors exist, keep .safetensors and remove .bin.
- sdxl-vae-fp16-fix: keep diffusion_pytorch_model.safetensors only; remove diffusion_pytorch_model.bin,
  sdxl.vae.safetensors, sdxl_vae.safetensors (duplicate same weights).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR_DEFAULT = os.path.join(ROOT, "model")

# In VAE folders, keep this weight file; remove others of same type
VAE_KEEP_WEIGHTS = "diffusion_pytorch_model.safetensors"
# Weight files to remove when VAE_KEEP_WEIGHTS exists (duplicates)
VAE_REMOVE_WEIGHTS = [
    "diffusion_pytorch_model.bin",
    "sdxl.vae.safetensors",
    "sdxl_vae.safetensors",
]


def prune_folder(path: str, dry_run: bool) -> int:
    freed = 0
    if not os.path.isdir(path):
        return 0

    # Remove .cache
    cache_dir = os.path.join(path, ".cache")
    if os.path.isdir(cache_dir):
        for root, dirs, files in os.walk(cache_dir, topdown=False):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if not dry_run:
                        os.remove(fp)
                    freed += 1
                except OSError:
                    pass
        try:
            if not dry_run:
                os.rmdir(cache_dir)
            # remove empty parents up to .cache
            p = os.path.dirname(cache_dir)
            while p != path and os.path.isdir(p) and not os.listdir(p):
                if not dry_run:
                    os.rmdir(p)
                p = os.path.dirname(p)
        except OSError:
            pass

    # VAE: remove duplicate weight files (keep .safetensors)
    keep_path = os.path.join(path, VAE_KEEP_WEIGHTS)
    if os.path.isfile(keep_path):
        for name in VAE_REMOVE_WEIGHTS:
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                try:
                    if not dry_run:
                        os.remove(fp)
                    freed += 1
                except OSError:
                    pass
    return freed


def main():
    parser = argparse.ArgumentParser(description="Prune redundant files inside model/ (no folder removal).")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Model dir not found: {args.model_dir}", file=sys.stderr)
        return 1

    total = 0
    for name in os.listdir(args.model_dir):
        sub = os.path.join(args.model_dir, name)
        if os.path.isdir(sub):
            n = prune_folder(sub, args.dry_run)
            if n:
                print(f"{name}: would remove {n} items" if args.dry_run else f"{name}: removed {n} items")
                total += n

    if total:
        print(f"Done. {'Would free' if args.dry_run else 'Freed'} space by removing {total} file(s)/dir(s).")
    else:
        print("No redundant files found.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
