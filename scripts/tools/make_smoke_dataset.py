#!/usr/bin/env python3
"""
Create a tiny folder of synthetic PNGs + sidecar captions for **smoke-testing** ``train.py``.

No real photos required—just enough files for the dataloader to iterate.

Usage (repo root):
    python scripts/tools/make_smoke_dataset.py --out data/smoke_tiny
    python train.py --data-path data/smoke_tiny --results-dir results/smoke --model DiT-B/2-Text \\
        --image-size 256 --global-batch-size 1 --max-steps 2 --no-compile --num-workers 0 --dry-run

``--data-path`` is the parent folder; images live in ``<out>/train/``.

Use ``--dry-run`` for a single training step, or ``--max-steps 5`` for a few steps without ``--dry-run``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Write synthetic images + .txt captions for train smoke tests.")
    p.add_argument("--out", type=Path, required=True, help="Output directory (created if missing).")
    p.add_argument("--count", type=int, default=4, help="Number of PNG files (default 4).")
    p.add_argument("--size", type=int, default=256, help="Square image side (default 256, match --image-size).")
    args = p.parse_args()

    import numpy as np
    from PIL import Image

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    # Text2ImageDataset folder mode expects one subdir of images (not a flat root).
    train_dir = out / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, int(args.count))
    side = max(64, int(args.size))
    captions = [
        "simple geometric pattern, flat colors",
        "soft gradient background, abstract",
        "noise texture, colorful",
        "minimal shapes, high contrast",
    ]

    rng = np.random.default_rng(0)
    for i in range(n):
        # Deterministic but varied RGB arrays (no external assets).
        base = np.zeros((side, side, 3), dtype=np.uint8)
        c = i % 3
        base[..., c] = np.clip(80 + (i * 37) % 160, 0, 255)
        noise = rng.integers(0, 40, size=(side, side, 3), dtype=np.uint8)
        arr = np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        stem = f"smoke_{i:03d}"
        img.save(train_dir / f"{stem}.png")
        cap = captions[i % len(captions)]
        (train_dir / f"{stem}.txt").write_text(cap + "\n", encoding="utf-8")

    print(f"[make_smoke_dataset] wrote {n} PNG + caption pairs under {train_dir.resolve()}")
    print("Next: see docs/SMOKE_TRAINING.md for a minimal train.py command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
