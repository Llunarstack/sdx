"""
Normalize and boost captions for training.

Usage (JSONL manifest):
    python -m scripts.tools.normalize_captions --in manifest.jsonl --out manifest_norm.jsonl

This script:
- Applies normalize_tag_order (subject/age/height/build/anatomy ordering).
- Applies boost_hard_style_tags, boost_quality_tags, boost_domain_tags.
- Applies add_anti_blending_and_count for multi-person prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from data.caption_utils import (
    add_anti_blending_and_count,
    boost_domain_tags,
    boost_hard_style_tags,
    boost_quality_tags,
    normalize_tag_order,
)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _process_caption_pair(caption: str, negative: str) -> Tuple[str, str]:
    cap = caption or ""
    neg = negative or ""
    cap = normalize_tag_order(cap)
    cap = boost_hard_style_tags(cap, repeat_factor=3)
    cap = boost_quality_tags(cap, repeat_factor=3)
    cap = boost_domain_tags(cap, repeat_factor=2)
    cap, neg = add_anti_blending_and_count(cap, neg)
    return cap.strip(), neg.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize and boost captions for SDX training.")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", dest="out", type=str, required=True, help="Output JSONL manifest")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for rec in _iter_jsonl(in_path):
            caption = str(rec.get("caption", "") or "")
            neg = str(rec.get("negative_caption", rec.get("negative_prompt", "") or ""))
            caption_new, neg_new = _process_caption_pair(caption, neg)
            rec["caption"] = caption_new
            if "negative_caption" in rec:
                rec["negative_caption"] = neg_new
            elif "negative_prompt" in rec:
                rec["negative_prompt"] = neg_new
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"Normalized {count} captions -> {out_path}")


if __name__ == "__main__":
    main()

