"""Keyword-matched prompt add-ons for creatures/characters and physics-heavy materials.

Both return ``(positive_snippet, negative_snippet)``, or ``("", "")`` when nothing matches.
"""

from __future__ import annotations

from typing import Literal

from config.defaults.creature_character_prompts import (
    CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_COMMON_NEGATIVE_ADDON,
    CREATURE_NSFW_CONTEXT_POSITIVE,
    CREATURE_NSFW_NEGATIVE_ADDON,
    CREATURE_SFW_CONTEXT_POSITIVE,
    CREATURE_SFW_NEGATIVE_ADDON,
    HUMANOID_MONSTER_PROMPT_KEYWORDS,
)
from config.defaults.physics_material_prompts import (
    PHYSICS_COMMON_NEGATIVE_ADDON,
    PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN,
)

from utils.safety.minor_guard import is_sexual

CreatureRating = Literal["sfw", "nsfw", "auto"]

_ANTHRO_KW = ("anthro", "anthropomorphic", "furry", "fursona", "kemono", "muzzle", "digitigrade", "plantigrade", "paw")
_ROBOT_KW = ("robot", "android", "mech", "mecha", "cyborg", "gundam", "automaton", "synthetic humanoid")
_BEAST_KW = tuple(
    k
    for k in (
        "dragon",
        "griffin",
        "gryphon",
        "wyvern",
        "kaiju",
        "beast",
        "creature",
        "alien",
        "xenomorph",
        "mermaid",
        "merfolk",
        "golem",
    )
    if k not in frozenset(HUMANOID_MONSTER_PROMPT_KEYWORDS)
)

_PHYSICS_RULES: tuple[tuple[tuple[str, ...], str, int], ...] = (
    (
        (
            "water",
            "liquid",
            "fluid",
            "splash",
            "pour",
            "wave",
            "rain",
            "ocean",
            "underwater",
            "droplet",
            "meniscus",
            "steam",
            "mist",
            "fog",
        ),
        "fluids",
        0,
    ),
    (
        (
            "glass",
            "transparent",
            "translucent",
            "refraction",
            "frosted",
            "acrylic",
            "caustic",
            "see-through",
            "crystalline",
        ),
        "transparency",
        0,
    ),
    (("fabric", "cloth", "drap", "silk", "velvet", "hair", "ponytail", "cape", "dress"), "soft_bodies", 0),
    (
        ("stack", "balance", "leaning", "resting on", "sitting on", "standing on", "contact shadow"),
        "physics_materials",
        2,
    ),
    (("smoke", "fire", "flame", "explosion", "volumetric", "god ray"), "physics_materials", 0),
)


def suggest_creature_prompt_addons(prompt: str, *, rating: CreatureRating = "auto") -> tuple[str, str]:
    """Add-ons for anthro, robot/mech, humanoid-monster and beast prompts.

    ``rating="auto"`` switches to the mature packs only for sexual prompts (``minor_guard.is_sexual``).
    """
    lower = prompt.lower()
    mature = rating == "nsfw" or (rating == "auto" and is_sexual(lower))
    if rating == "sfw":
        domain_map = CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN
        ctx_pos, extra_neg = CREATURE_SFW_CONTEXT_POSITIVE, CREATURE_SFW_NEGATIVE_ADDON
    elif mature:
        domain_map = CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN
        ctx_pos, extra_neg = CREATURE_NSFW_CONTEXT_POSITIVE, CREATURE_NSFW_NEGATIVE_ADDON
    else:
        domain_map = CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN
        ctx_pos, extra_neg = "", ""

    pos = [
        domain_map[domain][0]
        for keywords, domain in (
            (_ANTHRO_KW, "anthro_furry"),
            (_ROBOT_KW, "robots_mechs"),
            (HUMANOID_MONSTER_PROMPT_KEYWORDS, "humanoid_monsters"),
            (_BEAST_KW, "creatures"),
        )
        if any(k in lower for k in keywords)
    ]
    if not pos:
        return "", ""
    neg = ", ".join(x for x in (CREATURE_COMMON_NEGATIVE_ADDON, extra_neg) if x)
    return ", ".join(([ctx_pos] if ctx_pos else []) + pos), neg


def suggest_physics_prompt_addons(prompt: str) -> tuple[str, str]:
    """Add-ons for fluids, transparency, soft bodies, rigid contact and volumetrics."""
    lower = prompt.lower()
    pos = [
        PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN[domain][idx]
        for keywords, domain, idx in _PHYSICS_RULES
        if any(k in lower for k in keywords)
    ]
    if not pos:
        return "", ""
    return ", ".join(pos), PHYSICS_COMMON_NEGATIVE_ADDON


__all__ = ["CreatureRating", "suggest_creature_prompt_addons", "suggest_physics_prompt_addons"]
