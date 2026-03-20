from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


_SPLIT_RE = re.compile(r"[,\n;|]+")
_WS_RE = re.compile(r"\s+")

# "Negative inside positive" defaults (Nannano-Bannas style directive pack)
DEFAULT_AVOID = [
    "lowres",
    "blurry",
    "jpeg artifacts",
    "bad anatomy",
    "extra fingers",
    "deformed hands",
    "text watermark",
    "logo",
]


@dataclass(frozen=True)
class PromptBreakdown:
    add: List[str]
    avoid: List[str]
    neutral: List[str]


def _norm_tag(s: str) -> str:
    return _WS_RE.sub(" ", s.strip().lower())


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def breakdown_prompt(prompt: str) -> PromptBreakdown:
    """
    Split a free-form prompt into:
      - add: things to explicitly encourage
      - avoid: negatives extracted from neg-like syntax
      - neutral: unknown directives left untouched
    Supported negative markers:
      - "no X", "without X", "avoid X"
      - "--neg: X"
      - "[NEG] X"
    """
    raw_parts = [p.strip() for p in _SPLIT_RE.split(prompt or "") if p.strip()]
    add: List[str] = []
    avoid: List[str] = []
    neutral: List[str] = []

    for p in raw_parts:
        low = _norm_tag(p)
        if low.startswith("--neg:"):
            t = _norm_tag(p.split(":", 1)[1])
            if t:
                avoid.append(t)
            continue
        if low.startswith("[neg]"):
            t = _norm_tag(p[5:])
            if t:
                avoid.append(t)
            continue
        if low.startswith("no "):
            t = _norm_tag(p[3:])
            if t:
                avoid.append(t)
            continue
        if low.startswith("without "):
            t = _norm_tag(p[8:])
            if t:
                avoid.append(t)
            continue
        if low.startswith("avoid "):
            t = _norm_tag(p[6:])
            if t:
                avoid.append(t)
            continue

        tag = _norm_tag(p)
        if tag:
            add.append(tag)
        else:
            neutral.append(p)

    return PromptBreakdown(
        add=_unique_keep_order(add),
        avoid=_unique_keep_order(avoid),
        neutral=_unique_keep_order(neutral),
    )


def compose_positive_with_embedded_negative(
    add: List[str],
    avoid: List[str],
    *,
    inject_default_avoid: bool = True,
) -> str:
    """
    Build a single positive prompt that includes anti-features as constraints.
    Format:
      "<add tags...>, clean composition, detailed rendering, avoid: <avoid tags...>"
    """
    add_clean = _unique_keep_order([_norm_tag(x) for x in add if _norm_tag(x)])
    avoid_clean = _unique_keep_order([_norm_tag(x) for x in avoid if _norm_tag(x)])
    if inject_default_avoid:
        avoid_clean = _unique_keep_order(avoid_clean + DEFAULT_AVOID)

    if not add_clean:
        add_clean = ["high quality", "coherent composition"]

    add_block = ", ".join(add_clean + ["clean composition", "detailed rendering"])
    avoid_block = ", ".join(avoid_clean) if avoid_clean else "none"
    return f"{add_block}, avoid: {avoid_block}"


def build_prompt_plan(prompt: str, *, inject_default_avoid: bool = True) -> Dict[str, object]:
    b = breakdown_prompt(prompt)
    composed = compose_positive_with_embedded_negative(
        b.add,
        b.avoid,
        inject_default_avoid=inject_default_avoid,
    )
    return {
        "input_prompt": prompt,
        "add": b.add,
        "avoid": b.avoid,
        "neutral": b.neutral,
        "composed_prompt": composed,
    }

