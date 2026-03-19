"""
Seed explorer: quickly scout diverse seeds for a prompt.

Usage:
    python -m scripts.tools.seed_explorer --ckpt results/.../best.pt --prompt "..." --preset sdxl --rows 2 --cols 4

This:
- Calls sample.main() programmatically with different seeds.
- Saves N images and a grid, plus seeds.json with used seeds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def _make_grid(images: List[Path], rows: int, cols: int, out_path: Path) -> None:
    pil_images = [Image.open(p).convert("RGB") for p in images]
    if not pil_images:
        return
    w, h = pil_images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(pil_images):
        r, c = divmod(idx, cols)
        if r >= rows:
            break
        grid.paste(img, (c * w, r * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved grid: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore seeds for a given prompt using sample.py.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--negative-prompt", type=str, default="")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--op-mode", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="seed_explorer")
    args, unknown = ap.parse_known_args()

    total = args.rows * args.cols
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.base_seed + i for i in range(total)]
    img_paths: List[Path] = []
    repo_root = Path(__file__).resolve().parents[2]
    sample_py = repo_root / "sample.py"
    for idx, seed in enumerate(seeds):
        out_path = out_dir / f"seed_{seed}.png"
        cli_args = [
            "--ckpt",
            args.ckpt,
            "--prompt",
            args.prompt,
            "--negative-prompt",
            args.negative_prompt,
            "--seed",
            str(seed),
            "--out",
            str(out_path),
            "--num",
            "1",
        ]
        if args.preset:
            cli_args.extend(["--preset", args.preset])
        if args.op_mode:
            cli_args.extend(["--op-mode", args.op_mode])
        cli_args.extend(unknown)
        print(f"[seed_explorer] Running sample.py with seed={seed}")
        subprocess.run([sys.executable, str(sample_py), *cli_args], check=True)
        img_paths.append(out_path)

    # Save grid and seeds.json
    grid_path = out_dir / "grid.png"
    _make_grid(img_paths, args.rows, args.cols, grid_path)
    (out_dir / "seeds.json").write_text(json.dumps({"seeds": seeds}, indent=2), encoding="utf-8")
    print(f"Saved seeds: {out_dir/'seeds.json'}")


if __name__ == "__main__":
    main()

