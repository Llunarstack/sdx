"""
Lightweight composition / text-fidelity briefing for diffusion prompts.

Mirrors operational goals of commercial models that emphasize legible typography,
consistent layout, and preserving identity/lighting across edits — without relying
on a closed competitor's weights.

Use ``mode="auto"`` to enable only when the prompt looks UI-, poster-, or text-heavy.
"""

from __future__ import annotations

import re
from typing import Literal

CompositionBriefMode = Literal["off", "auto", "on"]

# Triggers for auto mode (UI, posters, quoted strings, multilingual text mentions).
_NEEDS_BRIEF_PATTERNS: tuple[str, ...] = (
    r"\b(ui|ux|gui|user interface|app screen|mobile app|web page|mockup|wireframe|"
    r"dashboard|navbar|menu bar|icon pack|component library)\b",
    r"\b(poster|billboard|flyer|brochure|magazine cover|book cover|album cover|"
    r"packaging|label|signage|neon sign|title card|opening credits|credits scene)\b",
    r"\b(logo|logotype|wordmark|subtitle|caption|speech bubble|typography)\b",
    r"\b(japanese text|chinese text|korean text|arabic text|mixed language)\b",
    r"\b(render (the |any )?text|exact (text|string)|readable text)\b",
    r"\blayout\b|\b(grid|baseline|margins)\b",
)

_SUFFIX = (
    "coherent perspective and unified lighting across the frame; "
    "if letters or signage appear, crisp edges and correct spelling for the stated language (no gibberish glyphs); "
    "preserve subject identity, facial likeness, material colors, and key shadows when adapting from references"
)


def composition_brief_warranted(prompt: str) -> bool:
    """Heuristic: prompt likely benefits from text/layout fidelity guidance."""
    p = (prompt or "").strip().lower()
    if not p:
        return False
    if re.search(r'"[^"]{2,}"', prompt):
        return True
    return any(re.search(pat, p, re.IGNORECASE) for pat in _NEEDS_BRIEF_PATTERNS)


def apply_composition_brief(prompt: str, mode: CompositionBriefMode) -> str:
    """Append a concise composition-and-text fidelity suffix when appropriate."""
    prompt = (prompt or "").strip()
    if mode == "off" or not prompt:
        return prompt
    if mode == "auto" and not composition_brief_warranted(prompt):
        return prompt
    low = prompt.lower()
    # Avoid stacking the same guidance if a prior pass or the user already included it.
    if "coherent perspective and unified lighting" in low:
        return prompt
    fragment = _SUFFIX
    if fragment in low:
        return prompt
    return f"{prompt}, {fragment}".strip().strip(",")


__all__ = [
    "CompositionBriefMode",
    "apply_composition_brief",
    "composition_brief_warranted",
]
