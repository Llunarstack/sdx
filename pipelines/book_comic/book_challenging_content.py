"""
Prompt fragments for **hard** sequential-art cases: mature-rated narrative fidelity (when NSFW
stack is enabled), surreal or non-human designs, crowds, reflections, and extreme perspective.

These strings are craft / rendering guidance only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

# Keyed by visual_memory ``challenge_tags`` / consistency JSON tag names.
CHALLENGE_TAG_FRAGMENTS: dict[str, str] = {
    "mature_narrative_fidelity": (
        "mature-rated scenes follow the writer brief without arbitrary modesty drift, "
        "wardrobe or blocking changes that contradict the stated page intent"
    ),
    "surreal_weird": (
        "surreal or dream-logic staging reads intentional: coherent internal rules, "
        "stable focal hierarchy, readable silhouettes despite odd proportions"
    ),
    "non_human_morph": (
        "non-human or hybrid anatomy stays internally consistent across panels: "
        "same limb count, markings, material reads, and silhouette logic"
    ),
    "extreme_perspective": (
        "aggressive foreshortening and wide-angle framing with correct overlap and scale cues, "
        "ground plane reads clearly, figures stay anchored in space"
    ),
    "crowd_scale": (
        "dense crowd: many distinct readable figures, depth layering, no face cloning soup, "
        "clear staging of foreground vs background actors"
    ),
    "hands_heavy": (
        "hands-heavy composition: deliberate finger poses, correct thumb side, "
        "clean overlaps, props contacted with believable grip contact"
    ),
    "reflections_glass": (
        "mirrors, glass, and water reflections: consistent viewpoint, "
        "plausible flipped detail, no impossible duplicate geometry"
    ),
    "horror_mood": (
        "horror-adjacent illustration mood: readable dread through lighting and staging, "
        "no gratuitous gore emphasis unless the page prompt explicitly requests it"
    ),
    "technical_materials": (
        "difficult materials rendered coherently: chrome, wet skin, thin fabric folds, "
        "subsurface hints where appropriate, stable contact shadows"
    ),
}


def _tags(*keys: str) -> str:
    return merge_prompt_fragments(*(CHALLENGE_TAG_FRAGMENTS[k] for k in keys))


_PACK_POSITIVE: dict[str, str] = {
    "mature_coherence": _tags("mature_narrative_fidelity"),
    "surreal_weird": _tags("surreal_weird", "non_human_morph"),
    "technical_hard": _tags("extreme_perspective", "reflections_glass", "technical_materials"),
    "horror_mood": _tags("horror_mood"),
    "crowd_hands": _tags("crowd_scale", "hands_heavy"),
    "max": _tags(
        "mature_narrative_fidelity",
        "surreal_weird",
        "non_human_morph",
        "extreme_perspective",
        "crowd_scale",
        "hands_heavy",
        "reflections_glass",
        "technical_materials",
        "horror_mood",
    ),
}

CHALLENGE_PACK_CHOICES = frozenset({"none", *_PACK_POSITIVE})

CHALLENGE_NEGATIVE_ADDON = (
    "random unexplained censor bars, lens censorship blur, arbitrary black voids over anatomy, "
    "sanitized rewrite of the written scene, sudden modesty costume insertion"
)


def _pack_name(pack: str | None) -> str:
    return str(pack or "none").strip().lower()


def _tag_list(raw: Any) -> list[str]:
    """Accept a comma-separated string or a list of tag names."""
    if isinstance(raw, str):
        return raw.split(",")
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return []


def merge_challenge_tags(tags: Sequence[str]) -> str:
    """Map challenge tag names to lexicon fragments; unknown tags are skipped."""
    out: list[str] = []
    for raw in tags or []:
        key = str(raw or "").strip().lower().replace("-", "_")
        frag = CHALLENGE_TAG_FRAGMENTS.get(key)
        if frag:
            out.append(frag)
    return merge_prompt_fragments(*out)


def challenge_pack_positive(pack: str) -> str:
    """Resolve a named challenge pack to a positive prompt block ("" for none/unknown)."""
    return _PACK_POSITIVE.get(_pack_name(pack), "")


def challenge_pack_negative(pack: str) -> str:
    """Negative fragment for any real pack; "" for none/unknown."""
    return CHALLENGE_NEGATIVE_ADDON if _pack_name(pack) in _PACK_POSITIVE else ""


def challenging_content_from_mapping(block: Mapping[str, Any]) -> str:
    """
    JSON block shape::

        {"pack": "surreal_weird", "tags": ["hands_heavy"], "extra": "freeform"}

    *pack* uses the same names as ``--book-challenge-pack``. *tags* add CHALLENGE_TAG_FRAGMENTS keys.
    """
    if not block:
        return ""
    extra = str(block.get("extra", "") or "").strip()
    tags = _tag_list(block.get("tags") or block.get("challenge_tags"))
    return merge_prompt_fragments(challenge_pack_positive(block.get("pack")), merge_challenge_tags(tags), extra)


def visual_memory_challenge_clause(root: Mapping[str, Any]) -> str:
    """
    Read optional visual-memory keys:

    - ``book_challenge_pack``: same as CLI pack name
    - ``challenge_tags`` / ``challenge_tag``: list or comma string of tag keys
    - ``challenging_content``: object passed to :func:`challenging_content_from_mapping`
    - ``weird_character_notes`` / ``unusual_character_notes``: freeform string

    The mature-fidelity fragment is always included (once).
    """
    bits: list[str] = []
    ch = root.get("challenging_content")
    if isinstance(ch, dict):
        bits.append(challenging_content_from_mapping(ch))
    bits.append(challenge_pack_positive(root.get("book_challenge_pack")))
    bits.append(merge_challenge_tags(_tag_list(root.get("challenge_tags") or root.get("challenge_tag"))))

    mature = CHALLENGE_TAG_FRAGMENTS["mature_narrative_fidelity"]
    if mature.lower() not in merge_prompt_fragments(*bits).lower():
        bits.append(mature)

    for key in ("weird_character_notes", "unusual_character_notes"):
        w = str(root.get(key, "") or "").strip()
        if w:
            bits.append(
                f"unusual recurring character design locked across pages: {w}, "
                "same distinctive traits in every appearance unless the page prompt overrides"
            )
            break

    return merge_prompt_fragments(*bits)
