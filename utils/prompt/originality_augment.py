"""Inject **originality / novelty** phrases into prompts (``sample.py --originality``).

Used at **inference** and optionally during **training** (``--train-originality-prob``) so the model
sees compositional variety tokens, not only at sample time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from data.caption_utils import (
    AGE_TAGS,
    ANATOMY_FRAMING_TAGS,
    BODY_PART_TAGS,
    BUILD_BODY_TAGS,
    HEIGHT_TAGS,
    SUBJECT_PREFIXES,
)

if TYPE_CHECKING:
    import numpy as np

__all__ = ["ORIGINALITY_POSITIVE_TOKENS", "inject_originality_tokens"]

ORIGINALITY_POSITIVE_TOKENS: tuple[str, ...] = (
    "unique composition",
    "original concept",
    "fresh perspective",
    "unexpected details",
    "distinctive lighting",
    "novel scene",
    "inventive composition",
    "creative accidents",
    "new idea",
    "rare details",
    "unusual angle",
    "dynamic framing",
    "asymmetric balance",
    "bold color story",
    "muted palette twist",
    "dramatic rim light",
    "soft volumetric light",
    "cinematic depth",
    "environmental storytelling",
    "foreground interest",
    "layered depth",
    "tactile materials",
    "hand-crafted look",
    "non-stock pose",
    "candid moment",
    "slice of life",
    "dreamlike twist",
    "subtle surreal touch",
    "editorial fashion vibe",
    "fine art photography",
    "concept art energy",
    "indie aesthetic",
    "one-of-a-kind mood",
    "surprising negative space",
    "rhythm in shapes",
)


def inject_originality_tokens(
    prompt: str,
    strength: float,
    rng: np.random.Generator,
    *,
    tokens: Sequence[str] | None = None,
) -> str:
    """
    Insert *k* random originality tokens after leading person/subject tags (comma-separated).

    *strength* in ``[0, 1]`` maps to ``k = 1 + round(3 * strength)`` clipped to token count.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if not (prompt and prompt.strip()) or strength <= 0:
        return prompt

    pool = list(tokens) if tokens is not None else list(ORIGINALITY_POSITIVE_TOKENS)
    if not pool:
        return prompt

    k = 1 + int(round(strength * 3))
    k = max(1, min(k, len(pool)))
    chosen = list(rng.choice(pool, size=k, replace=False))

    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    person_terms = (
        list(SUBJECT_PREFIXES)
        + list(AGE_TAGS)
        + list(HEIGHT_TAGS)
        + list(BUILD_BODY_TAGS)
        + list(ANATOMY_FRAMING_TAGS)
        + list(BODY_PART_TAGS)
    )

    def _norm(x: str) -> str:
        return x.lower().strip().replace("_", " ")

    person_norm = [_norm(t) for t in person_terms]

    def _is_person_term(tag: str) -> bool:
        t = _norm(tag)
        return any(t == pt or t.startswith(pt + " ") for pt in person_norm)

    insert_at = 0
    while insert_at < len(parts) and _is_person_term(parts[insert_at]):
        insert_at += 1

    parts[insert_at:insert_at] = chosen
    return ", ".join(parts)
