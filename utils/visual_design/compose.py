from __future__ import annotations

import re
from typing import Final, Literal

from .registry import DOMAIN_NEGATIVES, DOMAIN_POSITIVES

VisualDesignDomain = Literal[
    "ui_ux",
    "architecture",
    "stem",
    "textbook",
    "brand",
    "infographic",
    "packaging",
    "wayfinding",
    "general_product",
    "editorial_layout",
    "presentation_slide",
    "technical_blueprint",
    "fashion_flat",
]

Intensity = Literal["lite", "standard", "strong"]

# Module cache: immutable, single tuple build at import (faster repeated CLI --help).
DESIGN_PACK_IDS: Final[tuple[str, ...]] = tuple(DOMAIN_POSITIVES.keys())
_DOMAIN_KEYS: Final[frozenset[str]] = frozenset(DOMAIN_POSITIVES.keys())

_TIER_LOOKUP: Final[dict[str, str]] = {"lite": "lite", "standard": "standard", "strong": "strong"}

_VISUAL_CLI_CHOICES: Final[tuple[str, ...]] = ("none", "auto") + DESIGN_PACK_IDS

# Pre-compiled IGNORECASE avoids per-call `.lower()` on multi-kB prompts.
_RAW_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("ui_ux", r"\b(ui|ux|gui|mockup|wireframe|figma|design system|component)\b"),
    ("brand", r"\b(logo|wordmark|brand identity|style guide|mascot|monogram)\b"),
    ("architecture", r"\b(architecture|archviz|facade|floor plan|brutalist|bauhaus)\b"),
    ("stem", r"\b(graph|equation|diagram|formula|physics|chemistry|vector field|chart)\b"),
    ("textbook", r"\b(textbook|didactic|workbook|primer|syllabus|student edition)\b"),
    ("infographic", r"\b(infographic|timeline|dashboard explainer|data viz)\b"),
    ("packaging", r"\b(packaging|label design|product box|sleeve|carton)\b"),
    ("wayfinding", r"\b(wayfinding|signage|pictogram|exit sign|map legend)\b"),
    ("editorial_layout", r"\b(editorial spread|magazine spread|annual report spread|masthead)\b"),
    ("presentation_slide", r"\b(keynote|powerpoint slide|pitch deck slide|speaker deck)\b"),
    (
        "technical_blueprint",
        r"\b(blueprint|patent drawing|engineering drawing|section view|exploded diagram|cad drawing)\b",
    ),
    ("fashion_flat", r"\b(fashion flat|technical flat|tech pack flat|garment flat)\b"),
)


def _compiled_domain_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple((d, re.compile(p, re.IGNORECASE)) for d, p in _RAW_PATTERNS)


_COMPILED_DOMAIN_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = _compiled_domain_patterns()


def design_pack_ids() -> tuple[str, ...]:
    """Known domain keys (cached tuple, same object as ``DESIGN_PACK_IDS``)."""
    return DESIGN_PACK_IDS


def visual_design_cli_domain_choices() -> tuple[str, ...]:
    """Argparse ``choices`` (includes none/auto)."""
    return _VISUAL_CLI_CHOICES


def merge_visual_fragments(*parts: str) -> str:
    """Join non-empty prompt fragments with comma spacing (single pass)."""
    out: list[str] = []
    ap = out.append
    for p in parts:
        s = (p or "").strip().strip(",")
        if s:
            ap(s)
    return ", ".join(out)


def _tier_key(intensity: str) -> str:
    return _TIER_LOOKUP.get(intensity, "standard")


def build_visual_design_prompt_pair(
    prompt: str,
    domain: str,
    *,
    intensity: Intensity = "standard",
    dedupe_positive: bool = True,
) -> tuple[str, str]:
    """Return ``(positive, negative_addon)``."""
    base = (prompt or "").strip()
    d = (domain or "").strip().lower()
    if d not in _DOMAIN_KEYS:
        raise ValueError(f"Unknown visual design domain {domain!r}; use one of: {DESIGN_PACK_IDS}")

    tier = _tier_key(intensity)
    pos_map = DOMAIN_POSITIVES[d]
    fragment = pos_map.get(tier) or pos_map["standard"]

    base_l = base.lower()
    frag_l = fragment.lower()
    if dedupe_positive and frag_l in base_l:
        positive = base
    else:
        positive = merge_visual_fragments(base, fragment)

    neg = ""
    neg_map = DOMAIN_NEGATIVES.get(d)
    if neg_map:
        neg = (neg_map.get(tier) or neg_map.get("standard", "") or "").strip()

    return positive, neg


def apply_visual_design_pack(
    prompt: str,
    domain: str,
    *,
    intensity: Intensity = "standard",
    dedupe: bool = True,
) -> str:
    pos, _ = build_visual_design_prompt_pair(prompt, domain, intensity=intensity, dedupe_positive=dedupe)
    return pos


def prompt_suggests_domain(prompt: str) -> VisualDesignDomain | None:
    p = (prompt or "").strip()
    if not p:
        return None
    for dom, rx in _COMPILED_DOMAIN_PATTERNS:
        if rx.search(p):
            return dom  # type: ignore[return-value]
    return None


__all__ = [
    "DESIGN_PACK_IDS",
    "VisualDesignDomain",
    "Intensity",
    "build_visual_design_prompt_pair",
    "apply_visual_design_pack",
    "design_pack_ids",
    "merge_visual_fragments",
    "prompt_suggests_domain",
    "visual_design_cli_domain_choices",
]
