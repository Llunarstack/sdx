#!/usr/bin/env python3
"""One-shot / few-shot style LoRA recipe (B-LoRA last-block targeting).

Builds a tiny captioned image folder from 1–N reference images and prints the
recommended ``train.py --lora-train --lora-style-preset`` command.

Example:
  python -m scripts.tools train_style_lora_oneshot \\
    --images path/to/style.png --out-dir data/style_tok --trigger styletok
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="One or more style reference images (PNG/JPG/WEBP).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output dataset directory (images + .txt captions).",
    )
    p.add_argument(
        "--trigger",
        type=str,
        default="styletok",
        help="Trigger token used in captions (e.g. 'in the style of styletok').",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="How many copies of each image to write (augments effective epoch length).",
    )
    p.add_argument(
        "--class-prompt",
        type=str,
        default="artwork",
        help="Generic class noun for captions (avoids locking the reference subject).",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=8,
        help="Suggested LoRA rank for the printed train command.",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Do not copy images; only print the train recipe.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = Path(args.out_dir)
    images = [Path(p) for p in args.images]
    for img in images:
        if not img.is_file():
            print(f"Missing image: {img}", file=sys.stderr)
            return 2

    caption = f"in the style of {args.trigger}, {args.class_prompt}"
    if not args.print_only:
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in images:
            for r in range(max(1, int(args.repeats))):
                stem = f"{img.stem}_r{r:03d}"
                dest = out / f"{stem}{img.suffix.lower()}"
                shutil.copy2(img, dest)
                (out / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
                n += 1
        print(f"Wrote {n} image/caption pairs under {out}")

    rank = max(4, int(args.rank))
    cmd = (
        f"python train.py --data-dir {out.as_posix()} --lora-train --lora-style-preset "
        f"--lora-rank {rank} --lora-alpha {rank} --epochs 1 "
        f"# ~200–400 effective steps depending on batch; use trigger '{args.trigger}' at sample time"
    )
    print("\nRecommended train (B-LoRA last-third style blocks):")
    print(cmd)
    print("\nSample with style LoRA on last layers only:")
    print(
        f'python sample.py --prompt "a cat, in the style of {args.trigger}" '
        f"--lora path/to/ckpt_lora.pt:1.0:style --lora-layers last"
    )
    print(
        "\nZero-shot alternative (no train): "
        "python sample.py --reference-image STYLE.png --reference-style-mode instantstyle"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
