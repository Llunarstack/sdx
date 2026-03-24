"""
Unicode and structure hygiene for training/inference captions (stdlib + optional xxhash).

Implements the Python side of ``docs/NATIVE_AND_SYSTEM_LIBS.md`` (NFKC normalization,
zero-width stripping, dedupe fingerprints). Kept under ``sdx_native`` so it does not
pull ``data.__init__`` (which would create import cycles if this module imported ``data``).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import FrozenSet, List, Set, Tuple

_ZWSP_CHARS: FrozenSet[str] = frozenset(("\u200b", "\u200c", "\u200d", "\ufeff"))
_C0_REMOVE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_zwsp(text: str) -> str:
    """Remove zero-width spaces / joiners / BOM from *text*."""
    if not text:
        return text
    return "".join(c for c in text if c not in _ZWSP_CHARS)


def strip_c0_controls(text: str) -> str:
    """Remove C0 control characters except tab/newline/carriage return."""
    return _C0_REMOVE.sub("", text)


def normalize_caption_for_training(
    caption: str,
    *,
    unicode_form: str = "NFKC",
    strip_controls: bool = True,
) -> str:
    """
    Normalize a comma-separated caption for training or lint:

    - ``unicodedata.normalize`` (default **NFKC**)
    - Strip zero-width characters
    - Optionally strip C0 controls
    - Trim each comma segment; drop empties; rejoin with ``, ``
    """
    if not (caption or "").strip():
        return (caption or "").strip()
    form = (unicode_form or "NFKC").upper()
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        form = "NFKC"

    parts: List[str] = []
    for segment in caption.split(","):
        s = segment
        if strip_controls:
            s = strip_c0_controls(s)
        s = strip_zwsp(unicodedata.normalize(form, s)).strip()
        if s:
            parts.append(s)
    return ", ".join(parts)


def caption_fingerprint(
    caption: str,
    *,
    algorithm: str = "auto",
    normalize_first: bool = True,
) -> str:
    """
    Fingerprint a caption for near-dedup.

    - ``auto``: **xxhash** 64-bit hex if installed, else **SHA256** hex.
    """
    raw = normalize_caption_for_training(caption) if normalize_first else (caption or "")
    data = raw.encode("utf-8")
    algo = (algorithm or "auto").lower()
    if algo in ("auto", "xxhash"):
        try:
            import xxhash  # type: ignore[import-untyped]

            return xxhash.xxh64(data).hexdigest()
        except ImportError:
            if algo == "xxhash":
                raise ImportError("algorithm='xxhash' requires: pip install xxhash")
    return hashlib.sha256(data).hexdigest()


def _token_set(text: str) -> Set[str]:
    return {t.strip().lower() for t in (text or "").split(",") if t.strip()}


def pos_neg_token_overlap(positive: str, negative: str) -> Tuple[int, int, float]:
    """
    Tokens in both positive and negative (case-insensitive).
    Returns ``(overlap_count, union_count, jaccard)``.
    """
    p = _token_set(positive)
    n = _token_set(negative)
    if not p and not n:
        return 0, 0, 0.0
    inter = p & n
    union = p | n
    j = len(inter) / len(union) if union else 0.0
    return len(inter), len(union), j


def jsonl_row_caption_keys() -> Tuple[str, ...]:
    """Field names for caption text in a manifest dict."""
    return ("caption", "text", "prompt")


__all__ = [
    "caption_fingerprint",
    "jsonl_row_caption_keys",
    "normalize_caption_for_training",
    "pos_neg_token_overlap",
    "strip_c0_controls",
    "strip_zwsp",
]
