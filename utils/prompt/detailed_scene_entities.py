"""
Helpers for **busy scenes**: several characters or objects, each noun readable and consistent,
with **pose / weight / contact** and **physics** cues layered from existing SDX packs.

This does not add a scene graph or simulator; it prepends/appends **caption-shaped priors**
that diffusion responds to: separation, anti-merge negatives, anatomy, materials, fluids/stacking.
"""

from __future__ import annotations

import re
from typing import List, Literal, Tuple

from config.defaults.art_mediums import (
    ANATOMY_NEG_LITE,
    ANATOMY_NEG_STRONG,
    ANATOMY_POS_LITE,
    ANATOMY_POS_STRONG,
)

DetailedSceneMode = Literal["off", "auto", "on"]
StrengthMode = Literal["lite", "strong"]

# Crowd / multi-count / ensemble cues.
_GROUP_HINTS = (
    "group",
    "crowd",
    "team",
    "family",
    "soldiers",
    "knights",
    "ensemble",
    "cast of",
    "several ",
    "multiple ",
    "many ",
    "each ",
    "gathered",
    "standing together",
)

# ``five cats``, ``3 people``, etc.
_COUNT_PHRASE = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|pair|pairs)\b|\b\d{1,2}\s+(people|person|men|women|characters|children|kids|figures|objects|items|books|posters|soldiers|horses|cats|dogs|robots)\b",
    re.IGNORECASE,
)

_PEOPLE_HINT = re.compile(
    r"\b(person|people|man|woman|men|women|boy|girl|child|children|kid|kids|character|characters|portrait|crowd|knight|warrior|soldier|dancer|couple|family)\b",
    re.IGNORECASE,
)


def _merge_unique_csv(*parts: str) -> str:
    seen: set[str] = set()
    out: List[str] = []
    for block in parts:
        if not block:
            continue
        for chunk in block.split(","):
            c = chunk.strip()
            if not c:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
    return ", ".join(out)


def detailed_scene_warrants_boost(prompt: str) -> bool:
    """Conservative: busy or explicitly multi-entity prompts."""
    p = (prompt or "").strip()
    if not p:
        return False
    low = p.lower()
    if any(h in low for h in _GROUP_HINTS):
        return True
    if _COUNT_PHRASE.search(p):
        return True
    # Long tag-style prompts with several distinct clauses (avoid short boilerplate tags).
    segments = [s.strip() for s in p.split(",") if len(s.strip()) >= 16]
    if len(segments) >= 6:
        return True
    if " and " in low and len(p) > 100:
        return True
    return False


def extract_key_segments(prompt: str, *, max_segments: int = 12) -> List[str]:
    """Rough noun-phrases from comma-separated prompt chunks (for logging / UI)."""
    raw = [s.strip() for s in (prompt or "").replace("\n", ",").split(",") if s.strip()]
    # Drop very short noise tokens
    out = [s for s in raw if len(s) >= 4 and s.lower() not in {"best quality", "masterpiece", "detailed", "8k"}]
    return out[:max_segments]


def apply_detailed_scene_boost(
    prompt: str,
    mode: DetailedSceneMode,
    *,
    strength: StrengthMode = "lite",
) -> Tuple[str, str]:
    """
    Return ``(augmented_prompt, negative_fragment)``.

    Pulls in:
      - entity separation / anti-identity-swap language
      - pose, weight, contact (ground plane)
      - anatomy packs when people are likely
      - physics / creature heuristics via existing research helpers
    """
    prompt = (prompt or "").strip()
    if mode == "off" or not prompt:
        return prompt, ""

    if mode == "auto" and not detailed_scene_warrants_boost(prompt):
        return prompt, ""

    sep_pos = (
        "spatially separated readable subjects, clear negative space or depth between figures and props, "
        "consistent scale relationships, foreground midground background readable"
    )
    sep_neg = (
        "merged subjects, melted silhouettes, conjoined torsos, props fused between characters, "
        "ambiguous which limb belongs to whom, scale doubling errors"
    )

    noun_pos = (
        "each principal noun in the description keeps its own palette, material, and silhouette; "
        "no attribute bleed between adjacent entities"
    )
    noun_neg = (
        "wrong outfit on wrong character, swapped hair or faces between people, duplicated same person "
        "with different labels, object A drawn with B's texture"
    )

    pose_lite_pos = (
        "plausible stance, weight supported by feet or seat, coherent center of mass, "
        "joints flex in natural ranges, feet meet ground plane or visible support"
    )
    pose_lite_neg = (
        "broken spine twist, floating feet, impossible lean without support, dislocated hips, "
        "limbs emerging from wrong torso, gravity-defying group pose"
    )

    pose_strong_pos = (
        pose_lite_pos + ", contact shadows under shoes and props, consistent cast shadow direction for the whole scene"
    )
    pose_strong_neg = pose_lite_neg + ", contradictory shadow directions, hovering furniture"

    if strength == "strong":
        pose_pos, pose_neg = pose_strong_pos, pose_strong_neg
    else:
        pose_pos, pose_neg = pose_lite_pos, pose_lite_neg

    pos_parts: List[str] = [sep_pos, noun_pos, pose_pos]
    neg_parts: List[str] = [sep_neg, noun_neg, pose_neg]

    if _PEOPLE_HINT.search(prompt):
        if strength == "strong":
            pos_parts.append(ANATOMY_POS_STRONG)
            neg_parts.append(ANATOMY_NEG_STRONG)
        else:
            pos_parts.append(ANATOMY_POS_LITE)
            neg_parts.append(ANATOMY_NEG_LITE)

    try:
        from research.creature_character_guidance import suggest_creature_prompt_addons

        cpos, cneg = suggest_creature_prompt_addons(prompt, rating="auto")
        if cpos:
            pos_parts.append(cpos)
        if cneg:
            neg_parts.append(cneg)
    except Exception:
        pass

    try:
        from research.physics_visual_guidance import suggest_physics_prompt_addons

        ppos, pneg = suggest_physics_prompt_addons(prompt)
        if ppos:
            pos_parts.append(ppos)
        if pneg:
            neg_parts.append(pneg)
    except Exception:
        pass

    pos_merged = _merge_unique_csv(*pos_parts)
    neg_merged = _merge_unique_csv(*neg_parts)

    # Dedupe against prompt body for positives
    low = prompt.lower()
    pos_tokens = [t.strip() for t in pos_merged.split(",") if t.strip() and t.strip().lower() not in low]
    pos_final = ", ".join(pos_tokens[:24])  # cap length
    if pos_final:
        out_prompt = f"{prompt}, {pos_final}".strip().strip(",")
    else:
        out_prompt = prompt

    return out_prompt, neg_merged


__all__ = [
    "DetailedSceneMode",
    "StrengthMode",
    "apply_detailed_scene_boost",
    "detailed_scene_warrants_boost",
    "extract_key_segments",
]
