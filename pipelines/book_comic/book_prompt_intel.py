"""
Lightweight **prompt intelligence** for long sequential-art runs: length estimates,
cast mention checks, and optional warning strings (T5 / CLIP context is finite).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


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
) -> dict[str, int]:
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
    parts["total_chars"] = sum(parts.values())
    blocks = (
        narration_prefix,
        consistency_block,
        panel_hint,
        rolling_context,
        visual_memory_fragment,
        oc_block,
        user_prompt,
    )
    # Join only non-empty blocks so empty slots do not inflate the estimate with ", " separators.
    parts["approx_tokens"] = approximate_token_estimate(", ".join(b.strip() for b in blocks if b and b.strip()))
    return parts


@dataclass(slots=True)
class CastMentionResult:
    """Which cast labels were found in a page prompt (case-insensitive substring)."""

    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def soft_reminder_fragment(self) -> str:
        """Short positive fragment nudging unnamed cast into the page line (empty if none missing)."""
        if not self.missing:
            return ""
        names = ", ".join(self.missing[:6])
        tail = " (and others)" if len(self.missing) > 6 else ""
        return f"this page should visibly include: {names}{tail}"


def find_cast_mentions(page_prompt: str, cast_names: Sequence[str]) -> CastMentionResult:
    """Heuristic: a cast name is "mentioned" if it appears as a case-insensitive substring."""
    text = (page_prompt or "").lower()
    found: set[str] = set()
    missing: set[str] = set()
    for raw in cast_names:
        name = str(raw).strip()
        if name:
            (found if name.lower() in text else missing).add(name)
    return CastMentionResult(found=sorted(found), missing=sorted(missing))
