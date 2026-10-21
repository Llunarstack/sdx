"""Caption hygiene for JSONL manifests (``sdx_native.text_hygiene.normalize_caption_for_training`` on the
caption fields): Unicode normalization, zero-width and optional C0-control stripping. Lives under
``sdx_native`` so tools can use it without importing the dataset modules.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .text_hygiene import normalize_caption_for_training


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """JSON-object rows of ``path``; blank, malformed and non-object lines are skipped."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def normalize_jsonl_caption_fields(
    rec: dict[str, Any],
    *,
    caption_key: str = "caption",
    negative_caption_key: str = "negative_caption",
    negative_prompt_fallback_key: str = "negative_prompt",
    unicode_form: str = "NFKC",
    strip_controls: bool = True,
) -> tuple[str, str]:
    """Normalize a row's caption and negative caption in place; returns ``(caption, negative)`` ("" if missing).

    The negative comes from ``negative_caption_key`` when present, else ``negative_prompt_fallback_key``.
    """
    caption = str(rec.get(caption_key) or "")
    neg_key = negative_caption_key if negative_caption_key in rec else negative_prompt_fallback_key
    negative = str(rec.get(neg_key) or "")
    if caption.strip():
        caption = normalize_caption_for_training(caption, unicode_form=unicode_form, strip_controls=strip_controls)
        rec[caption_key] = caption
    if negative.strip():
        negative = normalize_caption_for_training(negative, unicode_form=unicode_form, strip_controls=strip_controls)
        rec[neg_key] = negative
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
