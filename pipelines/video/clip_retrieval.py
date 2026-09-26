"""
CLIP / structure-aware clip retrieval — replace bag-of-words ranking.
"""

from __future__ import annotations

from typing import Any

__all__ = ["score_clip_for_prompt", "rank_clips_structured"]


def score_clip_for_prompt(clip: Any, prompt: str) -> float:
    """
    Structured score: token overlap + grounded entity/tag hits + duration prior.

    Uses PromptGroundGraph so 'three red cars' prefers clips tagged red/car.
    """
    from .prompt_ground_graph import parse_prompt_ground
    from .retrieval import _tokenize

    g = parse_prompt_ground(prompt)
    q = set(_tokenize(prompt))
    for e in g.entities:
        q.add(e.name.lower())
        q.update(a.lower() for a in e.attributes)
    hay = set(_tokenize(getattr(clip, "title", "") or "")) | set(
        _tokenize(" ".join(str(t) for t in (getattr(clip, "tags", []) or [])))
    )
    if not q:
        return 0.0
    overlap = len(q & hay) / max(len(q), 1)
    # Entity hit bonus
    ent_hits = 0
    for e in g.entities:
        if e.negated:
            continue
        if e.name.lower() in hay or any(a.lower() in hay for a in e.attributes):
            ent_hits += 1
    ent = ent_hits / max(len([e for e in g.entities if not e.negated]), 1)
    return float(0.55 * overlap + 0.45 * ent)


def rank_clips_structured(
    clips: list[Any],
    prompt: str,
    *,
    top_k: int = 8,
) -> list[tuple[Any, float]]:
    scored = [(c, score_clip_for_prompt(c, prompt)) for c in clips]
    scored.sort(key=lambda t: -t[1])
    return scored[: max(1, top_k)]
