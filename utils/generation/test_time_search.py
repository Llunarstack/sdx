"""
Test-time **compute search** — successive-halving over seeds with verifier
pruning (the o1-style scaling lever, applied to image generation).

Naive best-of-N spends the full step budget on every candidate. This
orchestrator auditions a wide seed pool at few steps, prunes with a verifier,
and promotes only survivors to progressively more steps — so the same NFE
budget explores several times more of the seed space, which is where most of
the quality variance between generations actually lives.

Rung 0 additionally runs the ``variation_guard`` collapse audit: near-duplicate
candidates are re-rolled with fresh seeds so the pool covers distinct modes
instead of paying to score the same image twice.

Model-agnostic by design: callers supply ``render(seed, steps, scale)`` and
``score(images)`` callables, so this composes with any sampler and any scorer
(``test_time_pick`` combos, ``OnlineRewardModel``, a human). The editing-phase
loop makes a natural final rung.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CandidateResult",
    "SearchConfig",
    "SearchReport",
    "SearchRung",
    "default_rungs",
    "run_search",
]


@dataclass(slots=True)
class SearchRung:
    """One tournament round: render survivors at ``steps``, keep the best ``keep``."""

    steps: int
    keep: int
    resolution_scale: float = 1.0


@dataclass(slots=True)
class SearchConfig:
    pool_size: int = 8
    rungs: list[SearchRung] = field(default_factory=lambda: default_rungs())
    reroll_duplicates: bool = True
    near_duplicate_threshold: float = 0.05
    max_rerolls: int = 4


@dataclass(slots=True)
class CandidateResult:
    seed: int
    scores: list[float] = field(default_factory=list)
    """One verifier score per rung the candidate survived into."""
    eliminated_at_rung: int = -1
    """-1 while alive; otherwise the rung index where it was pruned."""


@dataclass(slots=True)
class SearchReport:
    candidates: list[CandidateResult]
    winner_seed: int
    winner_image: np.ndarray | None
    total_nfe: int
    """Total sampler steps spent (network function evaluations)."""
    naive_nfe: int
    """What best-of-``pool_size`` at final-rung steps would have cost."""
    rerolled_seeds: list[int] = field(default_factory=list)

    @property
    def nfe_savings(self) -> float:
        return 1.0 - self.total_nfe / self.naive_nfe if self.naive_nfe else 0.0


def default_rungs(pool_size: int = 8, final_steps: int = 50) -> list[SearchRung]:
    """
    Geometric audition→final schedule: everyone at ~⅛ of the final budget,
    top half at ~⅖, one winner at full steps.
    """
    audition = max(4, final_steps // 8)
    mid = max(audition + 2, int(final_steps * 0.4))
    return [
        SearchRung(steps=audition, keep=max(2, pool_size // 3)),
        SearchRung(steps=mid, keep=1),
        SearchRung(steps=final_steps, keep=1),
    ]


def _reroll_pool(
    images: list[np.ndarray],
    seeds: list[int],
    render: Callable[[int, int, float], np.ndarray],
    rung: SearchRung,
    config: SearchConfig,
    next_seed: int,
) -> tuple[list[np.ndarray], list[int], list[int], int, int]:
    """Replace near-duplicate candidates with fresh seeds (bounded retries)."""
    from utils.quality.variation_guard import audit_variation, reseed_indices

    rerolled: list[int] = []
    spent = 0
    for _ in range(config.max_rerolls):
        report = audit_variation(images, near_duplicate_threshold=config.near_duplicate_threshold)
        flagged = reseed_indices(report, max_collapse=1.01)  # only discrete duplicates here
        if not flagged:
            break
        for idx in flagged:
            seeds[idx] = next_seed
            images[idx] = render(next_seed, rung.steps, rung.resolution_scale)
            rerolled.append(next_seed)
            spent += rung.steps
            next_seed += 1
    return images, seeds, rerolled, spent, next_seed


def run_search(
    render: Callable[[int, int, float], np.ndarray],
    score: Callable[[Sequence[np.ndarray]], Sequence[float]],
    config: SearchConfig | None = None,
    *,
    base_seed: int = 0,
) -> SearchReport:
    """
    Run the tournament. Seeds are ``base_seed .. base_seed+pool_size-1`` plus
    any re-rolls. Returns the full bracket so callers can inspect runner-ups.
    """
    cfg = config or SearchConfig()
    if not cfg.rungs:
        raise ValueError("SearchConfig.rungs must not be empty")
    seeds = [base_seed + i for i in range(cfg.pool_size)]
    next_seed = base_seed + cfg.pool_size
    candidates = {s: CandidateResult(seed=s) for s in seeds}
    alive = list(seeds)
    total_nfe = 0
    rerolled_all: list[int] = []
    best_image: np.ndarray | None = None

    for rung_i, rung in enumerate(cfg.rungs):
        images = [render(s, rung.steps, rung.resolution_scale) for s in alive]
        total_nfe += rung.steps * len(alive)
        if rung_i == 0 and cfg.reroll_duplicates and len(alive) > 1:
            before = set(alive)
            images, alive, rerolled, spent, next_seed = _reroll_pool(images, alive, render, rung, cfg, next_seed)
            total_nfe += spent
            rerolled_all.extend(rerolled)
            # Drop replaced seeds so reports don't keep multiple alive "winners".
            for old in before - set(alive):
                candidates.pop(old, None)
            for s in set(alive) - before:
                candidates[s] = CandidateResult(seed=s)
        scores = [float(v) for v in score(images)]
        order = sorted(range(len(alive)), key=lambda i: scores[i], reverse=True)
        for i, s in enumerate(alive):
            candidates[s].scores.append(scores[i])
        survivors = [alive[i] for i in order[: max(1, rung.keep)]]
        for i in order[max(1, rung.keep) :]:
            candidates[alive[i]].eliminated_at_rung = rung_i
        best_image = images[order[0]]
        alive = survivors

    winner = alive[0]
    naive = cfg.rungs[-1].steps * cfg.pool_size
    return SearchReport(
        candidates=list(candidates.values()),
        winner_seed=winner,
        winner_image=best_image,
        total_nfe=total_nfe,
        naive_nfe=naive,
        rerolled_seeds=rerolled_all,
    )
