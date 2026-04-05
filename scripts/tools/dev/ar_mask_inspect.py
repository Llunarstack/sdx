#!/usr/bin/env python3
"""Print AR self-attention mask stats (sparsity, shape) for a patch grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect block-causal AR mask.")
    ap.add_argument("--h", type=int, default=16, help="Patch grid height")
    ap.add_argument("--w", type=int, default=16, help="Patch grid width")
    ap.add_argument("--blocks", type=int, default=2, help="Macro-blocks per side (0=full, 2 or 4 typical)")
    ap.add_argument(
        "--order",
        type=str,
        default="raster",
        choices=("raster", "zorder"),
        help="Macro-block visit order",
    )
    ap.add_argument("--compare", action="store_true", help="Also print zorder if order=raster (or vice versa)")
    args = ap.parse_args()

    from models.ar_masks_extended import ar_mask_sparsity_stats, create_block_causal_mask_2d
    from utils.architecture.ar_block_layout import block_visit_order

    if args.blocks <= 0:
        print("num_ar_blocks<=0: full attention (no mask).")
        return 0

    m = create_block_causal_mask_2d(args.h, args.w, args.blocks, block_order=args.order)
    frac, n = ar_mask_sparsity_stats(m)
    print(f"grid={args.h}x{args.w} blocks_per_side={args.blocks} order={args.order!r}")
    print(f"mask shape=({n},{n})  allowed_pair_fraction={frac:.4f}")
    print(f"macro-block visit: {block_visit_order(args.blocks, args.order)}")

    if args.compare:
        import torch

        other = "zorder" if args.order == "raster" else "raster"
        m2 = create_block_causal_mask_2d(args.h, args.w, args.blocks, block_order=other)
        frac2, _ = ar_mask_sparsity_stats(m2)
        same = bool(torch.equal(m, m2))
        print(f"compare order={other!r} allowed_pair_fraction={frac2:.4f} same_tensor={same}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
