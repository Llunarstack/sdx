"""Prompt priors the invention stack appends: anatomy, anti-plastic texture, composition and camera cues.

Each prior is gated by what the prompt actually shows: people priors need a person, camera priors
need a photographic (or unstated) medium, depth-layer language needs depth words.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from config.defaults.art_mediums import detect_medium

__all__ = ["AnatomonPlan", "apply_anatomon_prompts", "apply_friction_to_prompts", "plan_anatomon", "scene_priors"]

HUMANISH = re.compile(
    r"\b(1girl|1boy|girl|boy|woman|man|person|hands?|fingers?|pose|standing|sitting|"
    r"holding|arm|legs?|body|portrait|selfie)\b",
    re.IGNORECASE,
)
_HANDS = re.compile(r"\b(hand|hands|fingers|holding|grasp|typing|peace sign)\b", re.IGNORECASE)
_DEPTH = re.compile(r"\b(behind|in front|foreground|background|depth)\b", re.IGNORECASE)

_ANATOMY_POS = (
    "anatomically correct limbs",
    "five fingers per hand",
    "correct joint articulation",
    "plausible bone lengths",
    "shoulders connected to arms",
    "no extra limbs",
)
_ANATOMY_NEG = (
    "extra fingers",
    "fused fingers",
    "broken wrists",
    "inverted elbows",
    "disconnected limbs",
    "melted hands",
    "extra arms",
    "bad anatomy",
)

_FRICTION_POS = (
    "natural skin pores",
    "subtle film grain",
    "asymmetric detail",
    "fabric weave visible",
    "imperfect real-world texture",
    "lived-in surfaces",
)
_FRICTION_NEG = (
    "plastic skin",
    "waxy skin",
    "oversmooth",
    "airbrushed",
    "porcelain doll skin",
    "CGI sheen",
    "beauty filter blur",
)

_PEOPLE_PRIORS: tuple[tuple[str, str], ...] = (
    (
        "subtle natural eye asymmetry, lived-in gaze, uneven catchlights",
        "perfectly symmetric doll eyes, identical mirrored pupils",
    ),
    (
        "natural teeth, correct gum line, coherent mouth interior",
        "broken teeth mesh, shark mouth, melted lips, extra teeth",
    ),
    ("feet planted with soft contact shadows on ground", "floating feet, missing contact shadow, hovering subject"),
    ("unfiltered skin, visible pores and peach fuzz", "Instagram beauty filter, Facetune smooth, poreless plastic"),
)
_HAND_PRIOR = (
    "detailed correct hands, five fingers, natural knuckles",
    "extra fingers, fused fingers, melted hands, poorly drawn hands",
)
_COMPOSITION_PRIOR = (
    "off-center subject, rule of thirds composition, asymmetric framing",
    "perfectly centered subject, bullseye composition, dead-center framing",
)
_DEPTH_PRIOR = (
    "explicit z-order, foreground subject, midground secondary, background environment, clear depth layers",
    "flat cardboard depth, everything same distance, z-fighting",
)
_CAMERA_PRIORS: tuple[tuple[str, str], ...] = (
    (
        "filmic highlight rolloff, soft clipped whites, no RGB burn",
        "harsh digital highlight clipping, neon burned whites",
    ),
    (
        "50mm lens look, f/2.8, optical vignetting subtle, Kodak Portra 400 color science, soft skin rolloff, "
        "EXIF-plausible photography",
        "",
    ),
)


@dataclass
class AnatomonPlan:
    active: bool = False
    positive_addon: str = ""
    negative_addon: str = ""
    fix_regions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_anatomon(prompt: str) -> AnatomonPlan:
    """Joint-chain anatomy language for prompts that show a person."""
    if not HUMANISH.search(prompt or ""):
        return AnatomonPlan()
    return AnatomonPlan(
        active=True,
        positive_addon=", ".join(_ANATOMY_POS),
        negative_addon=", ".join(_ANATOMY_NEG),
        fix_regions=["hands", "face"],
    )


def _append(positive: str, negative: str, pos_addon: str, neg_addon: str) -> tuple[str, str]:
    pos, neg = str(positive or "").rstrip(", "), str(negative or "").rstrip(", ")
    if pos_addon and pos_addon.lower() not in pos.lower():
        pos = f"{pos}, {pos_addon}".strip(", ")
    if neg_addon:
        neg = f"{neg}, {neg_addon}".strip(", ")
    return pos, neg


def apply_anatomon_prompts(positive: str, negative: str = "") -> tuple[str, str, AnatomonPlan]:
    plan = plan_anatomon(positive)
    if not plan.active:
        return positive, negative, plan
    pos, neg = _append(positive, negative, plan.positive_addon, plan.negative_addon)
    return pos, neg, plan


def apply_friction_to_prompts(positive: str, negative: str = "", *, strength: float = 0.55) -> tuple[str, str]:
    """Anti-plastic microtexture language; ``strength`` in 0..1 picks how many cues are used."""
    if strength <= 0:
        return positive, negative
    k = max(1, round(len(_FRICTION_POS) * float(strength)))
    return _append(positive, negative, ", ".join(_FRICTION_POS[:k]), ", ".join(_FRICTION_NEG[: max(2, k)]))


def scene_priors(prompt: str) -> list[tuple[str, str]]:
    """``(positive, negative)`` priors that fit ``prompt``; hands are skipped when ANATOMON already covers them."""
    p = str(prompt or "")
    out = [_COMPOSITION_PRIOR]
    if _DEPTH.search(p):
        out.append(_DEPTH_PRIOR)
    if HUMANISH.search(p):
        out.extend(_PEOPLE_PRIORS)
    if _HANDS.search(p) and "five fingers per hand" not in p.lower():
        out.append(_HAND_PRIOR)
    if detect_medium(p).family in ("photo", "generic"):
        out.extend(_CAMERA_PRIORS)
    return out
