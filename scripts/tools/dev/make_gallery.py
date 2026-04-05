#!/usr/bin/env python3
"""
Generate a gallery grid from a set of prompts using a trained checkpoint.

Saves individual images and a combined grid to docs/assets/gallery/.
Used to produce the README gallery section.

Usage
-----
    python scripts/tools/dev/make_gallery.py --ckpt results/.../best.pt
    python scripts/tools/dev/make_gallery.py --ckpt results/.../best.pt --preset anime
    python scripts/tools/dev/make_gallery.py --ckpt results/.../best.pt --prompts-file gallery_prompts.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Default gallery prompts — one per line, covers a range of styles/subjects.
DEFAULT_PROMPTS = [
    "cinematic portrait of a young woman, soft window light, shallow depth of field, film grain",
    "hero character full body, dynamic action pose, city rooftop at night, anime style",
    "photoreal landscape, misty mountain valley at dawn, golden hour, ultra detailed",
    "book illustration, cozy cottage interior, warm candlelight, watercolor style",
    "sci-fi concept art, alien megastructure, volumetric fog, dramatic lighting",
    "character sheet, front and side view, clean lineart, white background",
]

DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, oversaturated, deformed"


def _run_sample(
    ckpt: str,
    prompt: str,
    out: Path,
    *,
    preset: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    device: str,
) -> bool:
    """Run sample.py for one prompt. Returns True on success."""
    cmd = [
        sys.executable,
        str(ROOT / "sample.py"),
        "--ckpt", ckpt,
        "--prompt", prompt,
        "--negative-prompt", DEFAULT_NEGATIVE,
        "--holy-grail-preset", preset,
        "--steps", str(steps),
        "--cfg-scale", str(cfg),
        "--width", str(width),
        "--height", str(height),
        "--seed", str(seed),
        "--out", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-300:] if result.stderr else '(no stderr)'}")
        return False
    return True


def _make_grid(images: list[Path], out: Path, cols: int = 3) -> None:
    """Combine images into a grid and save to out."""
    from PIL import Image

    imgs = []
    for p in images:
        if p.exists():
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                pass

    if not imgs:
        print("No images to grid.")
        return

    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))

    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(out))
    print(f"Grid saved: {out}  ({len(imgs)} images, {cols} cols)")


def main():
    parser = argparse.ArgumentParser(description="Generate a gallery grid from a trained SDX checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to SDX checkpoint.")
    parser.add_argument("--prompts-file", default="", help="Text file with one prompt per line.")
    parser.add_argument(
        "--preset",
        default="auto",
        choices=["auto", "balanced", "photoreal", "anime", "illustration", "aggressive"],
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=6.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cols", type=int, default=3, help="Grid columns.")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "docs" / "assets" / "gallery"),
        help="Output directory for individual images and grid.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts.
    if args.prompts_file and Path(args.prompts_file).exists():
        prompts = [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        prompts = DEFAULT_PROMPTS

    print(f"Generating {len(prompts)} images → {out_dir}")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Preset     : {args.preset}  Steps: {args.steps}  CFG: {args.cfg}")
    print()

    generated: list[Path] = []
    for i, prompt in enumerate(prompts):
        slug = prompt[:40].lower().replace(" ", "_").replace(",", "").replace(".", "")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        out = out_dir / f"{i:02d}_{slug}.png"
        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}…")
        ok = _run_sample(
            args.ckpt,
            prompt,
            out,
            preset=args.preset,
            steps=args.steps,
            cfg=args.cfg,
            width=args.width,
            height=args.height,
            seed=args.seed + i,
            device=args.device,
        )
        if ok:
            generated.append(out)
            print(f"  → {out.name}")
        else:
            print("  → skipped (error)")

    if generated:
        grid_path = out_dir / "gallery_grid.png"
        _make_grid(generated, grid_path, cols=args.cols)
        print()
        print(f"Gallery complete: {len(generated)}/{len(prompts)} images")
        print(f"Grid: {grid_path}")
        print()
        print("Add to README.md:")
        print("  ![Gallery](docs/assets/gallery/gallery_grid.png)")
    else:
        print("No images generated — check checkpoint path and sample.py errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
