"""
Heuristic **prompt add-ons** for creatures, anthro characters, robots/mechs, and humanoid monsters.

Uses ``config.defaults.creature_character_prompts`` for prepend/append snippets in tooling or batch prep.

``rating`` selects **SFW** vs **mature/explicit** packs (positives + negatives). ``auto`` infers from
``CREATURE_NSFW_KEYWORDS`` (substring match, conservative).
"""

from __future__ import annotations

from typing import List, Literal, Tuple

from config.defaults.creature_character_prompts import (
    CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN,
    CREATURE_COMMON_NEGATIVE_ADDON,
    CREATURE_NSFW_CONTEXT_POSITIVE,
    CREATURE_NSFW_KEYWORDS,
    CREATURE_NSFW_NEGATIVE_ADDON,
    CREATURE_SFW_CONTEXT_POSITIVE,
    CREATURE_SFW_NEGATIVE_ADDON,
    HUMANOID_MONSTER_PROMPT_KEYWORDS,
)

CreatureRating = Literal["sfw", "nsfw", "auto"]


def _detect_nsfw(prompt_lower: str) -> bool:
    return any(k in prompt_lower for k in CREATURE_NSFW_KEYWORDS)


def _domain_prompt_maps(rating: CreatureRating, *, prompt_lower: str) -> dict:
    if rating == "sfw":
        return CREATURE_CHARACTER_SFW_RECOMMENDED_PROMPTS_BY_DOMAIN
    if rating == "nsfw":
        return CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN
    # auto
    return (
        CREATURE_CHARACTER_NSFW_RECOMMENDED_PROMPTS_BY_DOMAIN
        if _detect_nsfw(prompt_lower)
        else CREATURE_CHARACTER_RECOMMENDED_PROMPTS_BY_DOMAIN
    )


def _context_and_extra_negative(rating: CreatureRating, *, prompt_lower: str) -> Tuple[str, str]:
    if rating == "sfw":
        return CREATURE_SFW_CONTEXT_POSITIVE, CREATURE_SFW_NEGATIVE_ADDON
    if rating == "nsfw":
        return CREATURE_NSFW_CONTEXT_POSITIVE, CREATURE_NSFW_NEGATIVE_ADDON
    if _detect_nsfw(prompt_lower):
        return CREATURE_NSFW_CONTEXT_POSITIVE, CREATURE_NSFW_NEGATIVE_ADDON
    return "", ""


def suggest_creature_prompt_addons(
    prompt: str,
    *,
    rating: CreatureRating = "auto",
) -> Tuple[str, str]:
    """
    Return ``(positive_snippet, negative_snippet)`` from keyword matching.

    When ``rating`` is ``"sfw"`` or ``"nsfw"``, uses the corresponding per-domain prompt lines and
    appends rating-specific negatives. ``"auto"`` uses NSFW packs only if ``CREATURE_NSFW_KEYWORDS``
    matches (otherwise neutral domain lines + common negative only).
    """
    lower = prompt.lower()
    domain_map = _domain_prompt_maps(rating, prompt_lower=lower)
    ctx_pos, extra_neg = _context_and_extra_negative(rating, prompt_lower=lower)

    pos: List[str] = []
    if ctx_pos:
        pos.append(ctx_pos)

    anthro_kw = (
        "anthro",
        "anthropomorphic",
        "furry",
        "fursona",
        "kemono",
        "muzzle",
        "digitigrade",
        "plantigrade",
        "paw",
        "paws",
    )
    if any(k in lower for k in anthro_kw):
        pos.append(domain_map["anthro_furry"][0])

    robot_kw = ("robot", "android", "mech", "mecha", "cyborg", "gundam", "automaton", "synthetic humanoid")
    if any(k in lower for k in robot_kw):
        pos.append(domain_map["robots_mechs"][0])

    monster_kw = HUMANOID_MONSTER_PROMPT_KEYWORDS
    if any(k in lower for k in monster_kw):
        pos.append(domain_map["humanoid_monsters"][0])

    # Non-humanoid beasts / aliens (avoid overlap with ``HUMANOID_MONSTER_PROMPT_KEYWORDS``).
    _humanoid_set = frozenset(HUMANOID_MONSTER_PROMPT_KEYWORDS)
    creature_kw = tuple(
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
        if k not in _humanoid_set
    )
    if any(k in lower for k in creature_kw):
        pos.append(domain_map["creatures"][0])

    neg_parts = [CREATURE_COMMON_NEGATIVE_ADDON]
    if extra_neg:
        neg_parts.append(extra_neg)

    # Drop context if nothing domain-specific matched (avoid prepending only sfw/nsfw banner).
    if len(pos) <= 1 and not any(
        k in lower
        for k in anthro_kw + robot_kw + monster_kw + creature_kw
    ):
        return "", ""

    out_pos = ", ".join(pos)
    out_neg = ", ".join(neg_parts)
    return out_pos, out_neg
