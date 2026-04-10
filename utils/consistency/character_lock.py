"""**Character / subject consistency** helpers (LANDSCAPE cross-cutting trends).

Production stacks often advertise “character lock” without a separate LoRA per scene.
For tr/inference in SDX you can:

- Store a stable ``character_id`` (or short **anchor** description) in JSONL metadata.
- Prepend a **fixed identity block** to captions so T5 sees the same tokens across samples.

This does not replace LoRA fine-tunes — it standardizes prompt structure for datasets and tools.
"""

from __future__ import annotations

import re
from typing import Optional


def format_character_lock_block(
    character_id: str,
    description: str,
    *,
    tag: str = "character_ref",
) -> str:
    """
    Return a single paragraph to **prefix** captions for consistent conditioning.

    ``character_id`` should be stable across scenes (e.g. ``"hero_01"`` or a short hash).
    ``description`` is the visible traits (hair, outfit, face shape) you want held fixed.
    """
    cid = (character_id or "").strip()
    desc = (description or "").strip()
    if not cid and not desc:
        return ""
    parts = []
    if cid:
        parts.append(f"{tag}={cid}")
    if desc:
        parts.append(desc)
    return "[" + " | ".join(parts) + "]"


def merge_character_into_caption(
    caption: str,
    character_id: str,
    description: str,
    *,
    tag: str = "character_ref",
    mode: str = "prefix",
) -> str:
    """
    Insert a character lock block into ``caption``.

    ``mode``: ``prefix`` (default) | ``append``.
    """
    block = format_character_lock_block(character_id, description, tag=tag)
    if not block:
        return (caption or "").strip()
    cap = (caption or "").strip()
    if mode == "append":
        return f"{cap}, {block}" if cap else block
    return f"{block} {cap}".strip()


def extract_character_tag(caption: str, tag: str = "character_ref") -> Optional[str]:
    """Return the id after ``character_ref=`` in a ``[...]`` block if present (before ``|`` if any)."""
    m = re.search(rf"\[{re.escape(tag)}=([^\]]+)\]", caption or "")
    if not m:
        return None
    inner = m.group(1).strip()
    return inner.split("|")[0].strip()
