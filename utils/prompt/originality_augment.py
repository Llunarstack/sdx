"""
Inject **originality / novelty** phrases into prompts (same rules as ``sample.py --originality``).

Used at **inference** and optionally during **training** (``--train-originality-prob``) so the model
sees compositional variety tokens, not only at sample time. Token list:
``config.defaults.prompt_domains.ORIGINALITY_POSITIVE_TOKENS``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

__all__ = ["inject_originality_tokens", "default_originality_tokens"]


def default_originality_tokens() -> List[str]:
    try:
        from config.defaults.prompt_domains import ORIGINALITY_POSITIVE_TOKENS

        return list(ORIGINALITY_POSITIVE_TOKENS)
    except ImportError:
        try:
            from config.defaults.prompt_domains import ORIGINALITY_POSITIVE_TOKENS

            return list(ORIGINALITY_POSITIVE_TOKENS)
        except ImportError:
            return []


def inject_originality_tokens(
    prompt: str,
    strength: float,
    rng: np.random.Generator,
    *,
    tokens: Optional[Sequence[str]] = None,
) -> str:
    """
    Insert *k* random originality tokens after leading person/subject tags (comma-separated).

    *strength* in ``[0, 1]`` maps to ``k = 1 + round(3 * strength)`` clipped to token count.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if not (prompt and prompt.strip()) or strength <= 0:
        return prompt

    pool = list(tokens) if tokens is not None else default_originality_tokens()
    if not pool:
        return prompt

    try:
        from data.caption_utils import (
            AGE_TAGS,
            ANATOMY_FRAMING_TAGS,
            BODY_PART_TAGS,
            BUILD_BODY_TAGS,
            HEIGHT_TAGS,
            SUBJECT_PREFIXES,
        )
    except ImportError:
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
