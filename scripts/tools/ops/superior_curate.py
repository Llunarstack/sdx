#!/usr/bin/env python3
"""Curate a training JSONL with ``utils/data_quality/pipeline.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Input manifest JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--dedup", choices=["", "phash", "md5"], default="phash")
    p.add_argument("--min-caption-len", type=int, default=8)
    p.add_argument("--max-caption-len", type=int, default=512)
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--min-clip-sim", type=float, default=0.0)
    p.add_argument("--min-aesthetic-proxy", type=float, default=0.0)
    p.add_argument("--bad-words", default="")
    p.add_argument("--image-root", default="")
    args = p.parse_args()

    from utils.data_quality import FilterConfig, filter_jsonl_file

    cfg = FilterConfig(
        dedup=args.dedup,
        min_caption_len=args.min_caption_len,
        max_caption_len=args.max_caption_len,
        min_weight=args.min_weight,
        min_clip_sim=args.min_clip_sim,
        min_aesthetic_proxy=args.min_aesthetic_proxy,
        bad_words=tuple(x.strip().lower() for x in args.bad_words.split(",") if x.strip()),
        image_root=Path(args.image_root) if args.image_root else None,
    )
    _rows, stats = filter_jsonl_file(args.input, config=cfg, output_path=args.out)
    print(
        f"kept={stats.kept} dropped={stats.dropped_total} "
        f"(dup={stats.dropped_dup} caption={stats.dropped_caption} weight={stats.dropped_weight})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
