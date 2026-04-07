"""
Fast, dependency-light caption hygiene for JSONL manifests.

This is intentionally placed under ``sdx_native`` so training and tools can
optionally use it without pulling heavyweight dataset modules.

It focuses on the *lowest-level* cleanup that is safe to apply everywhere:
- Unicode normalization (default NFKC)
- Zero-width stripping
- Optional C0 control stripping

See: ``sdx_native.text_hygiene.normalize_caption_for_training``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .text_hygiene import normalize_caption_for_training


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def normalize_jsonl_caption_fields(
    rec: Dict[str, Any],
    *,
    caption_key: str = "caption",
    negative_caption_key: str = "negative_caption",
    negative_prompt_fallback_key: str = "negative_prompt",
    unicode_form: str = "NFKC",
    strip_controls: bool = True,
) -> Tuple[str, str]:
    """
    Normalize caption-like fields in a JSONL row in-place.

    Returns ``(caption, negative)`` after normalization (empty if missing).
    """
    caption = str(rec.get(caption_key, "") or "")
    negative = str(rec.get(negative_caption_key, rec.get(negative_prompt_fallback_key, "") or ""))
    if caption.strip():
        caption = normalize_caption_for_training(caption, unicode_form=unicode_form, strip_controls=strip_controls)
        rec[caption_key] = caption
    if negative.strip():
        negative = normalize_caption_for_training(negative, unicode_form=unicode_form, strip_controls=strip_controls)
        if negative_caption_key in rec:
            rec[negative_caption_key] = negative
        elif negative_prompt_fallback_key in rec:
            rec[negative_prompt_fallback_key] = negative
    return caption, negative


def normalize_manifest_jsonl(
    *,
    inp: Path,
    out: Path,
    unicode_form: str = "NFKC",
    strip_controls: bool = True,
) -> int:
    """Stream-normalize a manifest JSONL file. Returns number of rows written."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8", newline="\n") as fw:
        for rec in _iter_jsonl(inp):
            normalize_jsonl_caption_fields(rec, unicode_form=unicode_form, strip_controls=strip_controls)
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Unicode + zero-width hygiene for JSONL caption fields.")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", dest="out", type=str, required=True, help="Output JSONL manifest")
    ap.add_argument("--unicode-form", type=str, default="NFKC", help="Unicode normalization form (NFC/NFD/NFKC/NFKD)")
    ap.add_argument("--no-strip-controls", action="store_true", help="Do not strip C0 controls")
    args = ap.parse_args()
    n = normalize_manifest_jsonl(
        inp=Path(args.inp),
        out=Path(args.out),
        unicode_form=str(args.unicode_form or "NFKC"),
        strip_controls=not bool(args.no_strip_controls),
    )
    print(f"Hygiene-normalized {n} rows -> {Path(args.out)}")


if __name__ == "__main__":
    main()

