#!/usr/bin/env python3
"""
Tag JSONL manifest rows with ``num_ar_blocks`` from an SDX (or compatible) DiT checkpoint.

Use when images were generated with a known AR regime so **ViT** / ranking see the right
4-D one-hot (see ``utils/architecture/ar_dit_vit.py`` and ``docs/AR.md``).

Examples::

  python scripts/tools/data/ar_tag_manifest.py --dit-ckpt results/run/best.pt \\
    --manifest-jsonl data/manifest.jsonl --out data/manifest_ar_tagged.jsonl

  python scripts/tools/data/ar_tag_manifest.py --num-ar-blocks 2 \\
    --manifest-jsonl data/manifest.jsonl --out data/out.jsonl --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from utils.architecture.ar_dit_vit import (
        ar_bridge_summary_lines,
        normalize_num_ar_blocks,
        read_num_ar_blocks_from_checkpoint,
        tag_manifest_row_ar,
    )

    p = argparse.ArgumentParser(
        description="Add num_ar_blocks (+ dit_num_ar_blocks, ar_regime) to JSONL from DiT ckpt or explicit value.",
        epilog=ar_bridge_summary_lines(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dit-ckpt", type=Path, help="SDX train.py checkpoint (.pt) with config.num_ar_blocks")
    g.add_argument("--num-ar-blocks", type=int, help="Explicit 0, 2, or 4 (no checkpoint read)")
    p.add_argument("--manifest-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true", help="Replace existing valid AR fields on rows")
    args = p.parse_args()

    if args.num_ar_blocks is not None:
        n = normalize_num_ar_blocks(args.num_ar_blocks)
        if n == -1:
            print("--num-ar-blocks must be 0, 2, or 4", file=sys.stderr)
            return 2
    else:
        n = read_num_ar_blocks_from_checkpoint(args.dit_ckpt)
        if n == -1:
            print(
                "Could not read num_ar_blocks from checkpoint (missing config?). "
                "Use --num-ar-blocks explicitly.",
                file=sys.stderr,
            )
            return 2

    inp, outp = args.manifest_jsonl, args.out
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 1
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_out = 0
    with inp.open(encoding="utf-8") as rf, outp.open("w", encoding="utf-8") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            row2 = tag_manifest_row_ar(row, n, overwrite=bool(args.overwrite))
            wf.write(json.dumps(row2, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[ar_tag_manifest] num_ar_blocks={n} rows={n_out} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
