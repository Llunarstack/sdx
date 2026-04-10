"""
Lightweight **prompt intelligence** for long sequential-art runs: length estimates,
cast mention checks, and optional warning strings (T5 / CLIP context is finite).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set


def approximate_token_estimate(text: str, *, chars_per_token: float = 3.2) -> int:
    """Rough upper-ish token count (English-ish prose; OK for budgeting, not exact)."""
    s = (text or "").strip()
    if not s:
        return 0
    return max(1, int(len(s) / max(0.5, float(chars_per_token))))


def composed_prompt_length_report(
    *,
    user_prompt: str,
    narration_prefix: str = "",
    consistency_block: str = "",
    panel_hint: str = "",
    rolling_context: str = "",
    visual_memory_fragment: str = "",
    oc_block: str = "",
) -> Dict[str, int]:
    """Return per-block character counts and total for a composed book page prompt."""
    parts = {
        "narration_prefix": len((narration_prefix or "").strip()),
        "consistency_block": len((consistency_block or "").strip()),
        "panel_hint": len((panel_hint or "").strip()),
        "rolling_context": len((rolling_context or "").strip()),
        "visual_memory": len((visual_memory_fragment or "").strip()),
        "oc_block": len((oc_block or "").strip()),
        "user_prompt": len((user_prompt or "").strip()),
    }
    parts["total_chars"] = sum(parts[k] for k in parts if k != "total_chars")
    parts["approx_tokens"] = approximate_token_estimate(
        ", ".join(
            (narration_prefix, consistency_block, panel_hint, rolling_context, visual_memory_fragment, oc_block, user_prompt)
        )
    )
    return parts


@dataclass
class CastMentionResult:
    """Which cast labels were found in a page prompt (case-insensitive substring)."""

    found: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    def soft_reminder_fragment(self) -> str:
        """Short positive fragment nudging unnamed cast into the page line (empty if none missing)."""
        if not self.missing:
            return ""
        names = ", ".join(self.missing[:6])
        tail = " (and others)" if len(self.missing) > 6 else ""
        return f"this page should visibly include: {names}{tail}"


_WORD_BOUNDARY_RE = re.compile(r"[\s,.;:!?|(){}\[\]\"']+")


def find_cast_mentions(page_prompt: str, cast_names: Sequence[str]) -> CastMentionResult:
    """
    Heuristic: cast name is "mentioned" if it appears as a substring (case-insensitive),
    or as a whole token for single-word names.
    """
    text = (page_prompt or "").lower()
    found: List[str] = []
    missing: List[str] = []
    for raw in cast_names:
        name = str(raw).strip()
        if not name:
            continue
        low = name.lower()
        if low in text:
            found.append(name)
            continue
        if " " not in name.strip():
            tokens = set(_WORD_BOUNDARY_RE.split(text))
            if low in tokens:
                found.append(name)
                continue
        missing.append(name)
    return CastMentionResult(found=sorted(set(found)), missing=sorted(set(missing)))


def strip_duplicate_prompt_phrases(text: str) -> str:
    """
    Remove repeated comma-separated segments (case-insensitive), keeping first occurrence.
    Helps when merging many blocks that repeat the same quality tags.
    """
    s = (text or "").strip()
    if not s:
        return ""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seen: Set[str] = set()
    out: List[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return ", ".join(out)


def panel_layout_hint(
    *,
    panels: int = 1,
    layout: str = "",
    reading_order: str = "left-to-right",
) -> str:
    """Positive tokens for multi-panel pages (comic / manga grids)."""
    bits: List[str] = []
    n = max(1, int(panels))
    if n == 1:
        bits.append("single clear focal illustration")
    else:
        bits.append(f"multi-panel page with {n} distinct panels")
    lv = (layout or "").strip().lower()
    if lv in ("grid", "strip", "splash", "inset", "tier", "webtoon_column"):
        bits.append(f"layout style: {lv.replace('_', ' ')}")
    ro = (reading_order or "").strip().lower()
    if ro and ro != "none":
        bits.append(f"panel reading order {ro}")
    return ", ".join(bits)
