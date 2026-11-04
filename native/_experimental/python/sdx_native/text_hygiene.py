"""
Unicode and structure hygiene for training/inference captions and JSONL manifests (stdlib + optional xxhash):
NFKC normalization, zero-width / C0-control stripping, dedupe fingerprints, pos/neg overlap.

Kept under ``sdx_native`` so it does not pull ``data.__init__`` (importing ``data`` here would create cycles).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_ZWSP_TABLE = dict.fromkeys(map(ord, "​‌‍﻿"))
_C0_REMOVE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_zwsp(text: str) -> str:
    """Remove zero-width spaces / joiners / BOM from *text*."""
    return text.translate(_ZWSP_TABLE) if text else text


def strip_c0_controls(text: str) -> str:
    """Remove C0 control characters except tab/newline/carriage return."""
    return _C0_REMOVE.sub("", text)


def normalize_caption_for_training(
    caption: str,
    *,
    unicode_form: str = "NFKC",
    strip_controls: bool = True,
) -> str:
    """Unicode-normalize (default NFKC), strip zero-width characters and optionally C0 controls, then trim
    each comma segment, drop empty ones and rejoin with ``", "``.

    Normalization runs before the comma split so a full-width comma (U+FF0C) becomes a separator too, which
    keeps the function idempotent.
    """
    if not (caption or "").strip():
        return (caption or "").strip()
    form = (unicode_form or "NFKC").upper()
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        form = "NFKC"
    if strip_controls:
        caption = strip_c0_controls(caption)
    caption = strip_zwsp(unicodedata.normalize(form, caption))
    return ", ".join(s for s in (seg.strip() for seg in caption.split(",")) if s)


def caption_fingerprint(
    caption: str,
    *,
    algorithm: str = "auto",
    normalize_first: bool = True,
) -> str:
    """Near-dedup fingerprint: ``auto`` = xxhash 64-bit hex if installed, else SHA256 hex."""
    raw = normalize_caption_for_training(caption) if normalize_first else (caption or "")
    data = raw.encode("utf-8")
    algo = (algorithm or "auto").lower()
    if algo in ("auto", "xxhash"):
        try:
            import xxhash  # type: ignore[import-untyped]

            return xxhash.xxh64(data).hexdigest()
        except ImportError:
            if algo == "xxhash":
                raise ImportError("algorithm='xxhash' requires: pip install xxhash") from None
    return hashlib.sha256(data).hexdigest()


def _token_set(text: str) -> set[str]:
    return {t.strip().lower() for t in (text or "").split(",") if t.strip()}


def pos_neg_token_overlap(positive: str, negative: str) -> tuple[int, int, float]:
    """Comma tokens shared by ``positive`` and ``negative`` (case-insensitive): ``(overlap, union, jaccard)``."""
    p, n = _token_set(positive), _token_set(negative)
    inter, union = p & n, p | n
    return len(inter), len(union), (len(inter) / len(union) if union else 0.0)


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
