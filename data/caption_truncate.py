"""Caption length limits without splitting mid-tag."""

from __future__ import annotations

from typing import List


def truncate_caption_at_comma_boundary(caption: str, max_len: int) -> str:
    """
    Truncate a comma-separated caption to *max_len* chars without cutting inside a tag.

    Keeps whole comma-separated segments from the start; drops trailing segments that
    would exceed the limit.
    """
    if max_len <= 0 or not caption:
        return (caption or "").strip()
    text = caption.strip()
    if len(text) <= max_len:
        return text
    parts: List[str] = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return text[:max_len]
    kept: List[str] = []
    length = 0
    for i, part in enumerate(parts):
        add = len(part) if not kept else len(part) + 2
        if length + add > max_len:
            break
        kept.append(part)
        length += add
    if kept:
        return ", ".join(kept)
    return parts[0][:max_len]
