"""Prompt intelligence: analyze free text and suggest stack enrichments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .tokens import split_tags, token_set


@dataclass
class PromptAnalysis:
    """Structured read of a user prompt before generation."""

    token_count: int = 0
    char_count: int = 0
    comma_rich: bool = False
    domains: List[str] = field(default_factory=list)
    complexity: str = "simple"  # simple | moderate | complex | extreme
    missing_quality: bool = False
    photographic: bool = False
    has_text_intent: bool = False
    solo_subject: bool = False
    multi_subject: bool = False
    suggested_controls: Dict[str, str] = field(default_factory=dict)
    suggested_positive: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


_QUALITY_MARKERS = frozenset(
    {
        "masterpiece",
        "best quality",
        "high quality",
        "highres",
        "absurdres",
        "8k",
        "ultra detailed",
        "detailed",
    }
)

_PHOTO_MARKERS = (
    "photoreal",
    "raw photo",
    "dslr",
    "35mm",
    "film photograph",
    "iphone photo",
    "shot on",
    "f/1.",
    "bokeh",
)

_TEXT_MARKERS = (
    "sign that says",
    "text that says",
    "lettering",
    "speech bubble",
    "caption reads",
    'reads "',
    'says "',
)


def _detect_domains(p_lower: str, tags: set[str]) -> List[str]:
    domains: List[str] = []
    if any(x in p_lower for x in ("anime", "manga", "cel shading", "1girl", "1boy")) or tags & {
        "1girl",
        "1boy",
        "anime",
    }:
        domains.append("anime")
    if any(x in p_lower for x in _PHOTO_MARKERS):
        domains.append("photographic")
    if any(x in p_lower for x in ("3d render", "octane", "blender", "unreal engine", "cgi")):
        domains.append("3d")
    if any(x in p_lower for x in ("interior", "architecture", "facade", "building")):
        domains.append("architecture")
    if any(x in p_lower for x in ("vehicle", "car", "motorcycle", "aircraft")):
        domains.append("vehicles")
    if any(x in p_lower for x in ("horror", "eldritch", "cosmic", "nightmare")):
        domains.append("horror")
    if any(x in p_lower for x in ("dragon", "furry", "anthro", "robot", "mecha")):
        domains.append("creature")
    return domains


def analyze_prompt(prompt: str) -> PromptAnalysis:
    raw = (prompt or "").strip()
    p_lower = raw.lower()
    tags = token_set(raw)
    tag_list = split_tags(raw)
    n_tags = len(tag_list)
    n_char = len(raw)

    analysis = PromptAnalysis(
        token_count=n_tags,
        char_count=n_char,
        comma_rich="," in raw,
        domains=_detect_domains(p_lower, tags),
    )

    if n_tags >= 22 or n_char >= 520:
        analysis.complexity = "extreme"
    elif n_tags >= 12 or n_char >= 280:
        analysis.complexity = "complex"
    elif n_tags >= 6 or n_char >= 120:
        analysis.complexity = "moderate"

    analysis.missing_quality = not any(m in p_lower for m in _QUALITY_MARKERS)
    analysis.photographic = "photographic" in analysis.domains
    analysis.has_text_intent = any(m in p_lower for m in _TEXT_MARKERS)
    analysis.solo_subject = bool(tags & {"solo", "1girl", "1boy", "1other"}) or "single " in p_lower
    analysis.multi_subject = bool(tags & {"2girls", "2boys", "3girls"}) or any(
        x in p_lower for x in ("crowd", "group", "2girls", "couple", "duo")
    )

    suggested: Dict[str, str] = {}
    if analysis.complexity in ("complex", "extreme"):
        suggested["adherence_pack"] = "standard"
    if analysis.complexity == "extreme":
        suggested["adherence_pack"] = "strict"
    if analysis.photographic:
        suggested["human_media_mode"] = "photographic"
    if analysis.solo_subject and not analysis.multi_subject:
        suggested["composition_mode"] = "single_subject"
        suggested["people_layout"] = "solo"
    if analysis.missing_quality and analysis.complexity in ("moderate", "complex", "extreme"):
        analysis.suggested_positive.append("detailed")
        analysis.notes.append("Consider --quality-pack top or masterpiece tags for short prompts.")

    analysis.suggested_controls = suggested
    return analysis


def apply_intelligence(
    positive: str,
    analysis: PromptAnalysis,
    *,
    auto_quality: bool = True,
    auto_controls: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """Optionally prepend light quality tags and return control hints."""
    out = positive
    controls: Dict[str, str] = dict(analysis.suggested_controls) if auto_controls else {}
    if auto_quality and analysis.missing_quality and analysis.complexity in ("simple", "moderate"):
        if "detailed" in analysis.suggested_positive:
            from .tokens import append_unique

            out = append_unique(out, analysis.suggested_positive)
    return out, controls
