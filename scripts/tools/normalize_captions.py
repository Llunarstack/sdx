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
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

try:
    from sdx_native.jsonl_caption_hygiene import normalize_jsonl_caption_fields as _native_hygiene_row
except Exception:
    _native_hygiene_row = None  # type: ignore[assignment]

_CHUNK_ROWS = 2048


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

def _process_jsonl_lines_worker(payload: Dict[str, Any]) -> List[str]:
    """
    Worker for multiprocessing: takes a dict with keys:
    - lines: List[str]
    - cfg: Dict[str, Any]
    Returns a list of JSON strings (one per output row).
    """
    import json as _json

    # Import inside worker to avoid pickling issues.
    from data.caption_utils import (  # noqa: WPS433
        add_anti_blending_and_count,
        apply_art_guidance_to_caption_pair,
        apply_shortcomings_to_caption_pair,
        apply_style_guidance_to_caption_pair,
        boost_domain_tags,
        boost_hard_style_tags,
        boost_quality_tags,
        normalize_tag_order,
    )

    try:
        from sdx_native.jsonl_caption_hygiene import (  # noqa: WPS433
            normalize_jsonl_caption_fields as _hyg_row,
        )
    except Exception:
        _hyg_row = None

    cfg = dict(payload.get("cfg") or {})
    unicode_normalize = bool(cfg.get("unicode_normalize", False))

    def process_pair(caption: str, negative: str) -> Tuple[str, str]:
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
            mode=str(cfg.get("shortcomings_mitigation", "none") or "none"),
            include_2d=bool(cfg.get("shortcomings_2d", False)),
        )
        cap, neg = apply_art_guidance_to_caption_pair(
            cap,
            neg,
            mode=str(cfg.get("art_guidance_mode", "none") or "none"),
            include_photography=bool(cfg.get("art_guidance_photography", True)),
            anatomy_mode=str(cfg.get("anatomy_guidance", "none") or "none"),
        )
        cap, neg = apply_style_guidance_to_caption_pair(
            cap,
            neg,
            mode=str(cfg.get("style_guidance_mode", "none") or "none"),
            include_artist_refs=bool(cfg.get("style_guidance_artists", True)),
        )
        return cap.strip(), neg.strip()

    out_lines: List[str] = []
    for line in payload.get("lines") or []:
        line = (line or "").strip()
        if not line:
            continue
        try:
            rec = _json.loads(line)
        except Exception:
            continue

        if unicode_normalize and _hyg_row is not None:
            _hyg_row(rec)

        caption = str(rec.get("caption", "") or "")
        neg = str(rec.get("negative_caption", rec.get("negative_prompt", "") or ""))
        caption_new, neg_new = process_pair(caption, neg)
        rec["caption"] = caption_new
        if "negative_caption" in rec:
            rec["negative_caption"] = neg_new
        elif "negative_prompt" in rec:
            rec["negative_prompt"] = neg_new
        out_lines.append(_json.dumps(rec, ensure_ascii=False))
    return out_lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize and boost captions for SDX training.")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", dest="out", type=str, required=True, help="Output JSONL manifest")
    ap.add_argument(
        "--unicode-normalize",
        action="store_true",
        help="Apply NFKC + zero-width strip per comma segment before other processing (via sdx_native).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="If >0, parallelize JSONL caption normalization using ProcessPool (order not guaranteed).",
    )
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

    cfg = dict(
        unicode_normalize=bool(getattr(args, "unicode_normalize", False)),
        shortcomings_mitigation=str(getattr(args, "shortcomings_mitigation", "none") or "none"),
        shortcomings_2d=bool(getattr(args, "shortcomings_2d", False)),
        art_guidance_mode=str(getattr(args, "art_guidance_mode", "none") or "none"),
        art_guidance_photography=bool(getattr(args, "art_guidance_photography", True)),
        anatomy_guidance=str(getattr(args, "anatomy_guidance", "none") or "none"),
        style_guidance_mode=str(getattr(args, "style_guidance_mode", "none") or "none"),
        style_guidance_artists=bool(getattr(args, "style_guidance_artists", True)),
    )

    workers = int(getattr(args, "workers", 0) or 0)
    count = 0
    if workers > 0:
        max_workers = min(workers, max(1, (os.cpu_count() or 4)))
        with out_path.open("w", encoding="utf-8", newline="\n") as fw, in_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            pending: List[Any] = []
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                chunk: List[str] = []
                for line in f:
                    chunk.append(line)
                    if len(chunk) >= _CHUNK_ROWS:
                        pending.append(ex.submit(_process_jsonl_lines_worker, {"lines": chunk, "cfg": cfg}))
                        chunk = []
                if chunk:
                    pending.append(ex.submit(_process_jsonl_lines_worker, {"lines": chunk, "cfg": cfg}))

                for fut in pending:
                    for out_line in fut.result():
                        fw.write(out_line + "\n")
                        count += 1
    else:
        with out_path.open("w", encoding="utf-8", newline="\n") as fw:
            for rec in _iter_jsonl(in_path):
                if bool(getattr(args, "unicode_normalize", False)) and _native_hygiene_row is not None:
                    _native_hygiene_row(rec)
                caption = str(rec.get("caption", "") or "")
                neg = str(rec.get("negative_caption", rec.get("negative_prompt", "") or ""))
                caption_new, neg_new = _process_caption_pair(
                    caption,
                    neg,
                    shortcomings_mode=cfg["shortcomings_mitigation"],
                    shortcomings_2d=bool(cfg["shortcomings_2d"]),
                    art_guidance_mode=cfg["art_guidance_mode"],
                    art_guidance_photography=bool(cfg["art_guidance_photography"]),
                    anatomy_guidance=cfg["anatomy_guidance"],
                    style_guidance_mode=cfg["style_guidance_mode"],
                    style_guidance_artists=bool(cfg["style_guidance_artists"]),
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
