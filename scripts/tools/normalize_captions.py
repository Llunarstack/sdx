"""
Normalize and boost captions for training.

Usage (JSONL manifest):
    python -m scripts.tools.normalize_captions --in manifest.jsonl --out manifest_norm.jsonl

This script:
- Applies normalize_tag_order (subject/age/height/build/anatomy ordering).
- Applies boost_hard_style_tags, boost_quality_tags, boost_domain_tags.
- Applies add_anti_blending_and_count for multi-person prompts.
- Optional: --shortcomings-mitigation / --shortcomings-2d (same taxonomy as train.py / sample.py).
- Optional: --art-guidance-mode / --anatomy-guidance (artist-first medium + anatomy packs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from data.caption_utils import (
    add_anti_blending_and_count,
    apply_art_guidance_to_caption_pair,
    apply_shortcomings_to_caption_pair,
    apply_style_guidance_to_caption_pair,
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


def _process_caption_pair(
    caption: str,
    negative: str,
    *,
    shortcomings_mode: str = "none",
    shortcomings_2d: bool = False,
    art_guidance_mode: str = "none",
    art_guidance_photography: bool = True,
    anatomy_guidance: str = "none",
    style_guidance_mode: str = "none",
    style_guidance_artists: bool = True,
) -> Tuple[str, str]:
    cap = caption or ""
    neg = negative or ""
    cap = normalize_tag_order(cap)
    cap = boost_hard_style_tags(cap, repeat_factor=3)
    cap = boost_quality_tags(cap, repeat_factor=3)
    cap = boost_domain_tags(cap, repeat_factor=2)
    cap, neg = add_anti_blending_and_count(cap, neg)
    cap, neg = apply_shortcomings_to_caption_pair(
        cap,
        neg,
        mode=shortcomings_mode,
        include_2d=shortcomings_2d,
    )
    cap, neg = apply_art_guidance_to_caption_pair(
        cap,
        neg,
        mode=art_guidance_mode,
        include_photography=art_guidance_photography,
        anatomy_mode=anatomy_guidance,
    )
    cap, neg = apply_style_guidance_to_caption_pair(
        cap,
        neg,
        mode=style_guidance_mode,
        include_artist_refs=style_guidance_artists,
    )
    return cap.strip(), neg.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize and boost captions for SDX training.")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", dest="out", type=str, required=True, help="Output JSONL manifest")
    ap.add_argument(
        "--shortcomings-mitigation",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Append failure-mode hints (same as train.py --train-shortcomings-mitigation)",
    )
    ap.add_argument(
        "--shortcomings-2d",
        action="store_true",
        help="With auto|all: include stylized 2D packs",
    )
    ap.add_argument(
        "--art-guidance-mode",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Artist-first medium guidance packs (traditional/digital/photography)",
    )
    ap.set_defaults(art_guidance_photography=True)
    ap.add_argument(
        "--no-art-guidance-photography",
        action="store_false",
        dest="art_guidance_photography",
        help="Disable photography packs for --art-guidance-mode auto|all",
    )
    ap.add_argument(
        "--anatomy-guidance",
        type=str,
        default="none",
        choices=["none", "lite", "strong"],
        help="Add anatomy/proportion guidance tags",
    )
    ap.add_argument(
        "--style-guidance-mode",
        type=str,
        default="none",
        choices=["none", "auto", "all"],
        help="Style-domain guidance packs (anime/comic/concept/game/photo language)",
    )
    ap.set_defaults(style_guidance_artists=True)
    ap.add_argument(
        "--no-style-guidance-artists",
        action="store_false",
        dest="style_guidance_artists",
        help="Disable artist/game-name stabilization cues in style guidance",
    )
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for rec in _iter_jsonl(in_path):
            caption = str(rec.get("caption", "") or "")
            neg = str(rec.get("negative_caption", rec.get("negative_prompt", "") or ""))
            caption_new, neg_new = _process_caption_pair(
                caption,
                neg,
                shortcomings_mode=str(getattr(args, "shortcomings_mitigation", "none") or "none"),
                shortcomings_2d=bool(getattr(args, "shortcomings_2d", False)),
                art_guidance_mode=str(getattr(args, "art_guidance_mode", "none") or "none"),
                art_guidance_photography=bool(getattr(args, "art_guidance_photography", True)),
                anatomy_guidance=str(getattr(args, "anatomy_guidance", "none") or "none"),
                style_guidance_mode=str(getattr(args, "style_guidance_mode", "none") or "none"),
                style_guidance_artists=bool(getattr(args, "style_guidance_artists", True)),
            )
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
