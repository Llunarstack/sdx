"""
Automatic **prompt breakdown** for clearer conditioning.

Splits comma-separated (and semicolon-separated) clauses into buckets aligned with
``utils/prompt/prompt_layout`` sections, then **reorders** them for encoder-friendly
priority (subject-first by default). Optional **labeled** multi-line string gives T5
explicit section boundaries without requiring a hand-written JSON layout.

This is heuristic keyword routing, not NLP parsing — good enough to lift adherence on
long tag-style prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Tuple

from utils.prompt.prompt_layout import PRESET_SECTION_ORDER, T5_SECTION_LABELS

BreakdownMode = Literal["off", "auto", "on"]
BreakdownFormat = Literal["ordered", "labeled"]
BreakdownOrder = Literal["subject_first", "quality_first", "scene_first"]

# Lowercase substrings → layout section (first match wins by highest score per segment).
_KEYWORD_BUCKETS: Dict[str, Tuple[str, ...]] = {
    "quality": (
        "masterpiece",
        "best quality",
        "high quality",
        "highly detailed",
        "absurdres",
        "8k",
        "4k",
        "uhd",
        "hdr",
        "sharp focus",
        "detailed",
    ),
    "style": (
        "anime",
        "manga",
        "oil painting",
        "watercolor",
        "photorealistic",
        "realistic",
        "cinematic",
        "digital art",
        "concept art",
        "illustration",
        "3d render",
        "pixel art",
        "impressionist",
        "baroque",
        "ukiyo",
        "sketch",
        "pastel drawing",
    ),
    "camera": (
        "wide angle",
        "telephoto",
        "macro",
        "depth of field",
        "bokeh",
        "lens flare",
        "35mm",
        "50mm",
        "85mm",
        "f/1.4",
        "f/2.8",
        "close-up",
        "closeup",
        "full body",
        "bust portrait",
        "headshot",
        "establishing shot",
        "dutch angle",
        "low angle",
        "high angle",
        "bird's eye",
        "worm's eye",
    ),
    "lighting": (
        "golden hour",
        "blue hour",
        "rim light",
        "backlight",
        "softbox",
        "studio lighting",
        "neon light",
        "candlelight",
        "volumetric",
        "god rays",
        "chiaroscuro",
        "ambient",
        "hard light",
        "soft light",
        "three-point lighting",
        "moonlight",
        "sunset lighting",
    ),
    "composition": (
        "rule of thirds",
        "symmetrical",
        "leading lines",
        "negative space",
        "centered composition",
        "dynamic composition",
        "frame within",
        "foreground",
        "midground",
        "background",
    ),
    "environment": (
        "indoor",
        "outdoor",
        "forest",
        "jungle",
        "city",
        "street",
        "alley",
        "beach",
        "ocean",
        "desert",
        "mountain",
        "sky",
        "clouds",
        "underwater",
        "office",
        "bedroom",
        "kitchen",
        "castle",
        "temple",
        "ruins",
        "space station",
        "cyberpunk city",
        "meadow",
        "snow",
        "rain",
        "fog",
        "night sky",
    ),
    "interaction": (
        "fighting",
        "dancing",
        "hugging",
        "kissing",
        "talking",
        "running",
        "walking together",
        "looking at",
        "holding hands",
        "high five",
        "shaking hands",
        "chasing",
        "embracing",
        "arguing",
        "dueling",
        "battling",
    ),
    "props": (
        "holding ",
        "wearing ",
        "carrying ",
        "sitting on ",
        "standing on ",
        "leaning on ",
        "sword",
        "shield",
        "staff",
        "gun",
        "bow",
        "book",
        "cup",
        "phone",
        "backpack",
        "helmet",
        "crown",
        "wings",
        "tail",
        "hat",
        "glasses",
    ),
    "color_script": (
        "warm colors",
        "cool colors",
        "cold colors",
        "monochrome",
        "sepia",
        "teal and orange",
        "pastel palette",
        "muted colors",
        "vibrant colors",
        "desaturated",
        "high contrast",
        "low contrast",
        "color grading",
    ),
}


def warrant_prompt_breakdown(prompt: str) -> bool:
    """Enable auto breakdown for long or clause-heavy prompts."""
    p = (prompt or "").strip()
    if not p:
        return False
    if len(p) >= 120:
        return True
    chunks = [c.strip() for c in re.split(r"[,;]", p) if c.strip()]
    return len(chunks) >= 5


def _split_segments(prompt: str) -> List[str]:
    parts: List[str] = []
    for chunk in re.split(r"[,;]", prompt):
        c = chunk.strip()
        if c:
            parts.append(c)
    return parts


def _classify_segment(seg: str) -> str:
    low = seg.lower()
    best = "subjects"
    best_n = 0
    for cat, kws in _KEYWORD_BUCKETS.items():
        n = sum(1 for k in kws if k in low)
        if n > best_n:
            best_n = n
            best = cat
    if best_n == 0:
        return "subjects"
    return best


def _dedupe_preserve(seq: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in seq:
        k = s.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s.strip())
    return out


@dataclass(slots=True)
class PromptBreakdownResult:
    """Structured buckets + rendered strings."""

    sections: Dict[str, List[str]] = field(default_factory=dict)
    ordered_flat: str = ""
    labeled_t5: str = ""
    section_order: Tuple[str, ...] = ()


def breakdown_prompt_to_sections(prompt: str) -> Dict[str, List[str]]:
    """Map section name → list of clause strings."""
    buckets: Dict[str, List[str]] = {k: [] for k in _KEYWORD_BUCKETS}
    buckets["subjects"] = []
    for seg in _split_segments(prompt):
        cat = _classify_segment(seg)
        buckets.setdefault(cat, []).append(seg)
    return buckets


def build_breakdown(
    prompt: str,
    *,
    order: BreakdownOrder = "subject_first",
) -> PromptBreakdownResult:
    """
    Classify clauses, reorder by ``PRESET_SECTION_ORDER[order]``, emit flat + labeled strings.

    ``subjects`` holds everything not matching a specialized bucket strongly enough.
    Automatic breakdown does not emit ``intent`` (reserved for hand-written layouts).
    """
    prompt = (prompt or "").strip()
    raw = breakdown_prompt_to_sections(prompt)
    merged: Dict[str, List[str]] = {
        "subjects": _dedupe_preserve(raw.get("subjects", [])),
        **{k: _dedupe_preserve(raw.get(k, [])) for k in _KEYWORD_BUCKETS},
    }
    order_names = PRESET_SECTION_ORDER.get(order, PRESET_SECTION_ORDER["subject_first"])
    ordered_parts: List[str] = []
    section_blocks: List[Tuple[str, str]] = []

    for name in order_names:
        if name == "intent":
            continue
        body_list = merged.get(name, [])
        if not body_list:
            continue
        joined = ", ".join(body_list)
        ordered_parts.append(joined)
        section_blocks.append((name, joined))

    flat = ", ".join(ordered_parts).strip().strip(",")
    if not flat:
        flat = prompt

    lines = [
        "Image generation: the labeled sections below describe one coherent image; use all of them together.",
    ]
    for name, body in section_blocks:
        b = (body or "").strip()
        if not b:
            continue
        label = T5_SECTION_LABELS.get(name, name.replace("_", " ").upper())
        lines.append(f"{label}: {b}.")
    labeled = "\n".join(lines)

    return PromptBreakdownResult(
        sections={k: list(v) for k, v in merged.items() if v},
        ordered_flat=flat,
        labeled_t5=labeled,
        section_order=order_names,
    )


def apply_prompt_breakdown(
    prompt: str,
    *,
    order: BreakdownOrder = "subject_first",
    output_format: BreakdownFormat = "ordered",
) -> Tuple[str, str]:
    """
    Returns ``(string_for_cache_and_clip, string_for_t5)``.

    When ``output_format`` is ``ordered``, both strings are identical (reordered flat prompt).
    When ``labeled``, first is flat reordered, second is multi-line labeled block for T5.
    """
    res = build_breakdown(prompt, order=order)
    if output_format == "labeled":
        return res.ordered_flat, res.labeled_t5
    return res.ordered_flat, res.ordered_flat


__all__ = [
    "BreakdownFormat",
    "BreakdownMode",
    "BreakdownOrder",
    "PromptBreakdownResult",
    "apply_prompt_breakdown",
    "breakdown_prompt_to_sections",
    "build_breakdown",
    "warrant_prompt_breakdown",
]
