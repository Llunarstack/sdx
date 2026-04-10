"""
Prompt fragments for **hard** sequential-art cases: mature-rated narrative fidelity (when NSFW
stack is enabled), surreal or non-human designs, crowds, reflections, and extreme perspective.

These strings are **craft / rendering** guidance. They do not substitute for ``--safety-mode``,
``--nsfw-pack`` in ``sample.py``—pair this module with those flags when
you need uncensored adult pipelines.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

# ---------------------------------------------------------------------------
# Per-tag fragments (visual_memory ``challenge_tags`` or consistency JSON)
# ---------------------------------------------------------------------------

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

CHALLENGE_PACK_CHOICES = frozenset(
    {
        "none",
        "mature_coherence",
        "surreal_weird",
        "technical_hard",
        "horror_mood",
        "crowd_hands",
        "max",
    }
)

# Positive lines merged per pack (excluding mature line unless allowed).
_PACK_POSITIVE_CORE: dict[str, str] = {
    "mature_coherence": CHALLENGE_TAG_FRAGMENTS["mature_narrative_fidelity"],
    "surreal_weird": merge_prompt_fragments(
        CHALLENGE_TAG_FRAGMENTS["surreal_weird"],
        CHALLENGE_TAG_FRAGMENTS["non_human_morph"],
    ),
    "technical_hard": merge_prompt_fragments(
        CHALLENGE_TAG_FRAGMENTS["extreme_perspective"],
        CHALLENGE_TAG_FRAGMENTS["reflections_glass"],
        CHALLENGE_TAG_FRAGMENTS["technical_materials"],
    ),
    "horror_mood": CHALLENGE_TAG_FRAGMENTS["horror_mood"],
    "crowd_hands": merge_prompt_fragments(
        CHALLENGE_TAG_FRAGMENTS["crowd_scale"],
        CHALLENGE_TAG_FRAGMENTS["hands_heavy"],
    ),
    "max": merge_prompt_fragments(
        CHALLENGE_TAG_FRAGMENTS["surreal_weird"],
        CHALLENGE_TAG_FRAGMENTS["non_human_morph"],
        CHALLENGE_TAG_FRAGMENTS["extreme_perspective"],
        CHALLENGE_TAG_FRAGMENTS["crowd_scale"],
        CHALLENGE_TAG_FRAGMENTS["hands_heavy"],
        CHALLENGE_TAG_FRAGMENTS["reflections_glass"],
        CHALLENGE_TAG_FRAGMENTS["technical_materials"],
        CHALLENGE_TAG_FRAGMENTS["horror_mood"],
    ),
}

CHALLENGE_NEGATIVE_ADDON = (
    "random unexplained censor bars, lens censorship blur, arbitrary black voids over anatomy, "
    "sanitized rewrite of the written scene, sudden modesty costume insertion"
)


def _safety_is_nsfw(safety_mode: str) -> bool:
    return str(safety_mode or "").strip().lower() == "nsfw"


def _content_rating_allows_mature(root: Mapping[str, Any]) -> bool:
    r = str(root.get("content_rating", "") or "").strip().lower()
    return r in ("mature", "unrestricted", "adult", "nsfw")


def merge_challenge_tags(tags: Sequence[str]) -> str:
    """Map challenge tag names to lexicon fragments; unknown tags are skipped."""
    out: List[str] = []
    for raw in tags or []:
        key = str(raw or "").strip().lower().replace("-", "_")
        frag = CHALLENGE_TAG_FRAGMENTS.get(key)
        if frag:
            out.append(frag)
    return merge_prompt_fragments(*out)


def challenge_pack_positive(pack: str, *, safety_mode: str = "") -> str:
    """
    Resolve a named challenge pack to a positive prompt block.

    ``mature_coherence`` and the ``mature_narrative_fidelity`` portion of ``max`` only apply when
    *safety_mode* is ``nsfw`` (avoids pushing mature-fidelity language into SFW runs).
    """
    name = str(pack or "none").strip().lower()
    if name == "none" or name not in CHALLENGE_PACK_CHOICES:
        return ""
    allow_mature = _safety_is_nsfw(safety_mode)
    if name == "mature_coherence":
        return _PACK_POSITIVE_CORE["mature_coherence"] if allow_mature else ""
    if name == "max":
        parts: List[str] = []
        if allow_mature:
            parts.append(_PACK_POSITIVE_CORE["mature_coherence"])
        parts.append(
            merge_prompt_fragments(
                CHALLENGE_TAG_FRAGMENTS["surreal_weird"],
                CHALLENGE_TAG_FRAGMENTS["non_human_morph"],
                CHALLENGE_TAG_FRAGMENTS["extreme_perspective"],
                CHALLENGE_TAG_FRAGMENTS["crowd_scale"],
                CHALLENGE_TAG_FRAGMENTS["hands_heavy"],
                CHALLENGE_TAG_FRAGMENTS["reflections_glass"],
                CHALLENGE_TAG_FRAGMENTS["technical_materials"],
                CHALLENGE_TAG_FRAGMENTS["horror_mood"],
            )
        )
        return merge_prompt_fragments(*parts)
    return _PACK_POSITIVE_CORE.get(name, "")


def challenge_pack_negative(pack: str) -> str:
    """Optional negative fragment; only non-none packs."""
    name = str(pack or "none").strip().lower()
    if name == "none" or name not in CHALLENGE_PACK_CHOICES:
        return ""
    return CHALLENGE_NEGATIVE_ADDON


def challenging_content_from_mapping(block: Mapping[str, Any], *, safety_mode: str = "") -> str:
    """
    JSON block shape::

        {"pack": "surreal_weird", "tags": ["hands_heavy"], "extra": "freeform"}

    *pack* uses the same names as ``--book-challenge-pack``. *tags* add CHALLENGE_TAG_FRAGMENTS keys.
    """
    if not block:
        return ""
    pack = str(block.get("pack", "none") or "none").strip().lower()
    extra = str(block.get("extra", "") or "").strip()
    tags_raw = block.get("tags") or block.get("challenge_tags") or []
    tags: List[str] = []
    if isinstance(tags_raw, str) and tags_raw.strip():
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    elif isinstance(tags_raw, list):
        tags = [str(t).strip() for t in tags_raw if str(t).strip()]
    tag_frag = merge_challenge_tags(tags)
    pack_frag = challenge_pack_positive(pack, safety_mode=safety_mode)
    return merge_prompt_fragments(pack_frag, tag_frag, extra)


def visual_memory_challenge_clause(root: Mapping[str, Any], *, safety_mode: str = "") -> str:
    """
    Read optional visual-memory keys:

    - ``book_challenge_pack``: same as CLI pack name
    - ``challenge_tags`` / ``challenge_tag``: list or comma string of tag keys
    - ``challenging_content``: object passed to :func:`challenging_content_from_mapping`
    - ``weird_character_notes`` / ``unusual_character_notes``: freeform string
    """
    bits: List[str] = []
    ch = root.get("challenging_content")
    if isinstance(ch, dict):
        bits.append(challenging_content_from_mapping(ch, safety_mode=safety_mode))
    bp = str(root.get("book_challenge_pack", "") or "").strip().lower()
    if bp and bp != "none":
        bits.append(challenge_pack_positive(bp, safety_mode=safety_mode))
    raw_tags = root.get("challenge_tags") or root.get("challenge_tag")
    if isinstance(raw_tags, str) and raw_tags.strip():
        tag_list = [t.strip() for t in raw_tags.split(",") if t.strip()]
        bits.append(merge_challenge_tags(tag_list))
    elif isinstance(raw_tags, list):
        bits.append(merge_challenge_tags([str(t) for t in raw_tags]))

    if _safety_is_nsfw(safety_mode) or _content_rating_allows_mature(root):
        mature_only = merge_challenge_tags(["mature_narrative_fidelity"])
        existing = merge_prompt_fragments(*bits)
        if mature_only and mature_only.lower() not in existing.lower():
            bits.append(mature_only)

    for key in ("weird_character_notes", "unusual_character_notes"):
        w = str(root.get(key, "") or "").strip()
        if w:
            bits.append(
                f"unusual recurring character design locked across pages: {w}, "
                "same distinctive traits in every appearance unless the page prompt overrides"
            )
            break

    return merge_prompt_fragments(*bits)
