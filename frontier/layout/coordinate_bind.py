"""
ConsistCompose-style coordinate binding in language prompts (LELG).

Embeds normalized bbox tokens so coordinate-aware CFG can parse layout from text.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_LOC_TOKEN = re.compile(
    r"<loc_([0-9.]+)_([0-9.]+)_([0-9.]+)_([0-9.]+)>",
    flags=re.IGNORECASE,
)


def bind_coordinates_to_prompt(
    prompt: str,
    box: Tuple[float, float, float, float],
    *,
    prefix: bool = True,
) -> str:
    """Insert ``<loc_x1_y1_x2_y2>`` token before prompt text."""
    x1, y1, x2, y2 = (round(float(v), 3) for v in box)
    token = f"<loc_{x1}_{y1}_{x2}_{y2}>"
    p = (prompt or "").strip()
    if token.lower() in p.lower():
        return p
    return f"{token} {p}" if prefix else f"{p} {token}"


def parse_loc_tokens(prompt: str) -> List[Tuple[float, float, float, float]]:
    """Extract all location tokens from a prompt string."""
    out: List[Tuple[float, float, float, float]] = []
    for m in _LOC_TOKEN.finditer(prompt or ""):
        out.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return out
