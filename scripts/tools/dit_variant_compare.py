#!/usr/bin/env python3
"""
Compare DiT (and optional EnhancedDiT) text variant sizes without training.

Usage (repo root):
    python scripts/tools/dit_variant_compare.py
    python scripts/tools/dit_variant_compare.py --image-size 512
    python scripts/tools/dit_variant_compare.py --models DiT-B/2-Text,DiT-XL/2-Text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DiT variant parameter counts.")
    parser.add_argument("--image-size", type=int, default=256, help="Training image side (latent = /8).")
    parser.add_argument(
        "--text-dim",
        type=int,
        default=4096,
        help="Text embedding dim (default T5-XXL-style 4096).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated DiT_models_text names; empty = all.",
    )
    args = parser.parse_args()

    from utils.dit_architecture import (
        dit_parameter_report,
        instantiate_dit_text,
        list_all_dit_registry_names,
    )

    names = [x.strip() for x in args.models.split(",") if x.strip()]
    if not names:
        names = list_all_dit_registry_names()

    rows = []
    for name in names:
        try:
            m = instantiate_dit_text(name, image_size=args.image_size, text_dim=args.text_dim)
            r = dit_parameter_report(m)
            kind = "EnhancedDiT" if name.startswith("EnhancedDiT") else "DiT-text"
            rows.append((name, kind, r))
        except Exception as e:
            rows.append((name, "?", {"error": str(e)}))

    print(f"image_size={args.image_size}  latent_side={args.image_size // 8}  text_dim={args.text_dim}\n")
    print(f"{'Model':<36} {'Kind':<14} {'Params':>14} {'FP32 GiB':>10} {'BF16 GiB':>10}")
    print("-" * 88)
    for name, kind, r in rows:
        if "error" in r:
            print(f"{name:<36} {kind:<14} ERROR: {r['error']}")
            continue
        print(
            f"{name:<36} {kind:<14} {r['total_parameters']:>14,} "
            f"{r['size_fp32_gib']:>10.2f} {r['size_bf16_gib']:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
