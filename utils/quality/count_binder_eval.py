"""Count / attribute-binding eval helpers for prompts and pick-best.

Uses ``PromptParser`` structure (counts, subjects) without a detector when possible;
pairs with OpenCV people/object estimates already in ``test_time_pick``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CountBindReport:
    expected_subjects: int
    expected_counts: dict[str, int]
    negation_count: int
    attribute_bindings: int
    score_hint: float  # 0–1 prior before looking at the image


def analyze_prompt_counts(prompt: str) -> CountBindReport:
    """Parse prompt for count/negation/binding complexity."""
    from models.prompt_adherence import PromptParser

    parsed = PromptParser().parse(prompt or "")
    counts = dict(getattr(parsed, "count_constraints", {}) or {})
    triples = list(getattr(parsed, "triples", []) or [])
    negs = list(getattr(parsed, "negation_indices", []) or [])
    n_subj = max(len(triples), len(counts), 1 if (prompt or "").strip() else 0)
    # Harder prompts get a slightly lower prior (need image verification more)
    complexity = min(1.0, 0.15 * len(counts) + 0.1 * len(negs) + 0.05 * len(triples))
    hint = float(max(0.35, 1.0 - 0.4 * complexity))
    return CountBindReport(
        expected_subjects=int(n_subj),
        expected_counts={str(k): int(v) for k, v in counts.items()},
        negation_count=len(negs),
        attribute_bindings=sum(len(getattr(t, "attributes", []) or []) for t in triples),
        score_hint=hint,
    )


def score_count_bind_match(
    rgb_uint8: np.ndarray,
    prompt: str,
    *,
    estimated_people: int | None = None,
) -> float:
    """
    Combine prompt-expected people count with an estimate.

    If ``estimated_people`` is None, tries ``test_time_pick.estimate_people_count``.
    """
    report = analyze_prompt_counts(prompt)
    target = 0
    for v in report.expected_counts.values():
        target = max(target, int(v))
    if target <= 0:
        # Fallback: people words
        pl = (prompt or "").lower()
        for word, n in (("three", 3), ("four", 4), ("five", 5), ("two", 2), ("a pair", 2)):
            if word in pl:
                target = n
                break
    if target <= 0:
        return float(report.score_hint)

    est = estimated_people
    if est is None:
        try:
            from utils.quality.test_time_pick import estimate_people_count

            est = int(estimate_people_count(np.asarray(rgb_uint8)))
        except Exception:
            est = 0
    if est <= 0:
        return float(report.score_hint * 0.7)
    # Soft match
    err = abs(int(est) - int(target))
    return float(max(0.0, 1.0 - 0.35 * err))


__all__ = ["CountBindReport", "analyze_prompt_counts", "score_count_bind_match"]
