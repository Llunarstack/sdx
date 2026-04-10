"""Fast comma-separated caption normalization (training / dedupe hot paths)."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def split_caption_parts(text: str) -> List[str]:
    """Split on commas; strip whitespace; drop empties."""
    if not text or not text.strip():
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def dedupe_caption_parts_preserve_order(parts: Sequence[str]) -> List[str]:
    """Case-insensitive dedupe while preserving first-seen casing."""
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def normalize_caption_csv(text: str) -> str:
    """Strip, split, dedupe (case-insensitive), rejoin with ``, ``."""
    parts = split_caption_parts(text)
    return ", ".join(dedupe_caption_parts_preserve_order(parts))


def merge_caption_csv(*chunks: str) -> str:
    """Merge multiple caption fragments with global dedupe."""
    acc: List[str] = []
    for c in chunks:
        acc.extend(split_caption_parts(c or ""))
    return ", ".join(dedupe_caption_parts_preserve_order(acc))


def token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard-like overlap on lowercased comma tokens."""
    sa = {x.lower() for x in split_caption_parts(a)}
    sb = {x.lower() for x in split_caption_parts(b)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)


def batch_normalize_captions(captions: Iterable[str]) -> List[str]:
    return [normalize_caption_csv(c) for c in captions]
