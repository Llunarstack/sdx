"""
Heuristic **prompt add-ons** for physics-heavy captions (fluids, glass, contact, volume).

Uses ``config.defaults.physics_material_prompts`` — no physics engine. Intended for tooling,
batch caption prep, or optional prepend/append before encoding text.
"""

from __future__ import annotations

from typing import List, Tuple

from config.defaults.physics_material_prompts import (
    PHYSICS_COMMON_NEGATIVE_ADDON,
    PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN,
)


def suggest_physics_prompt_addons(prompt: str) -> Tuple[str, str]:
    """
    Return ``(positive_snippet, negative_snippet)`` from light keyword matching.

    Empty strings if no physics-related keywords match (caller can skip).
    """
    lower = prompt.lower()
    pos: List[str] = []

    fluid_kw = (
        "water",
        "liquid",
        "fluid",
        "splash",
        "pour",
        "pouring",
        "wave",
        "rain",
        "ocean",
        "underwater",
        "droplet",
        "meniscus",
        "steam",
        "mist",
        "fog",
    )
    if any(k in lower for k in fluid_kw):
        pos.append(PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN["fluids"][0])

    trans_kw = (
        "glass",
        "transparent",
        "translucent",
        "refraction",
        "frosted",
        "acrylic",
        "caustic",
        "see-through",
        "crystalline",
    )
    if any(k in lower for k in trans_kw):
        pos.append(PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN["transparency"][0])

    soft_kw = ("fabric", "cloth", "drap", "silk", "velvet", "hair", "ponytail", "cape", "dress")
    if any(k in lower for k in soft_kw):
        pos.append(PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN["soft_bodies"][0])

    rigid_kw = ("stack", "balance", "leaning", "resting on", "sitting on", "standing on", "contact shadow")
    if any(k in lower for k in rigid_kw):
        pos.append(PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN["physics_materials"][2])

    vol_kw = ("smoke", "fire", "flame", "explosion", "volumetric", "god ray", "god rays")
    if any(k in lower for k in vol_kw):
        pos.append(PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN["physics_materials"][0])

    if not pos:
        return "", ""
    return ", ".join(pos), PHYSICS_COMMON_NEGATIVE_ADDON
