"""Comma-tag utilities for the prompt stack (delegates to ``utils.prompt.fast_paths``)."""

from __future__ import annotations

from typing import Set

from utils.prompt.fast_paths import (
    append_unique,
    join_tags,
    merge_fragments,
    split_tags,
)

__all__ = [
    "append_csv",
    "append_unique",
    "join_tags",
    "merge_fragments",
    "split_tags",
    "token_set",
]


def token_set(text: str) -> Set[str]:
    """Lowercased tokens from comma- and space-separated prompt text."""
    out: Set[str] = set()
    for part in (text or "").split(","):
        for word in part.split():
            w = word.strip().lower()
            if w:
                out.add(w)
    return out


def append_csv(base: str, fragment: str) -> str:
    """Append a fragment string to base (legacy ``*, {frag}`` pattern)."""
    frag = (fragment or "").strip().strip(",")
    if not frag:
        return (base or "").strip()
    return merge_fragments(base or "", frag)
