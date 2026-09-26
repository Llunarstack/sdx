import numpy as np
from utils.generation.test_time_search import (
    SearchConfig,
    SearchRung,
    default_rungs,
    run_search,
)

# Hidden per-seed quality: seed 5 is the best, but only more steps reveal it.
_QUALITY = {0: 0.2, 1: 0.5, 2: 0.3, 3: 0.6, 4: 0.1, 5: 0.9, 6: 0.4, 7: 0.55}


def _render(seed: int, steps: int, scale: float) -> np.ndarray:
    # Image brightness encodes quality, sharpening with more steps; each seed
    # gets a distinct spatial pattern so the variation guard sees diversity.
    q = _QUALITY.get(seed, 0.35)
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 60, size=(32, 32, 3))
    signal = q * min(1.0, steps / 20.0) * 195.0
    return np.clip(base + signal, 0, 255).astype(np.uint8)


def _score(images) -> list[float]:
    return [float(np.asarray(im).mean()) for im in images]


def test_search_finds_best_seed_under_reduced_budget():
    cfg = SearchConfig(pool_size=8, rungs=default_rungs(8, final_steps=48), reroll_duplicates=False)
    report = run_search(_render, _score, cfg, base_seed=0)
    assert report.winner_seed == 5
    assert report.total_nfe < report.naive_nfe  # cheaper than best-of-8 at 48 steps
    assert report.nfe_savings > 0.3
    assert report.winner_image is not None


def test_bracket_bookkeeping_is_consistent():
    cfg = SearchConfig(pool_size=8, rungs=default_rungs(8, final_steps=48), reroll_duplicates=False)
    report = run_search(_render, _score, cfg, base_seed=0)
    eliminated = [c for c in report.candidates if c.eliminated_at_rung >= 0]
    alive = [c for c in report.candidates if c.eliminated_at_rung == -1]
    assert len(alive) == 1 and alive[0].seed == 5
    assert len(eliminated) == 7
    # Survivors accumulated one score per rung they reached.
    assert len(alive[0].scores) == 3
    assert all(len(c.scores) == c.eliminated_at_rung + 1 for c in eliminated)


def test_duplicate_seeds_get_rerolled():
    def render_collapsed(seed: int, steps: int, scale: float) -> np.ndarray:
        # Seeds 0-3 all collapse to the same image; re-rolls (seed >= 4) diverge.
        if seed < 4:
            return np.full((32, 32, 3), 100, dtype=np.uint8)
        return _render(seed, steps, scale)

    cfg = SearchConfig(pool_size=4, rungs=[SearchRung(steps=6, keep=2), SearchRung(steps=24, keep=1)])
    report = run_search(render_collapsed, _score, cfg, base_seed=0)
    assert report.rerolled_seeds  # collapse was detected and fresh seeds drawn
    assert all(s >= 4 for s in report.rerolled_seeds)
    alive = [c for c in report.candidates if c.eliminated_at_rung == -1]
    assert len(alive) == 1
    assert alive[0].seed == report.winner_seed


def test_single_rung_degenerates_to_best_of_n():
    cfg = SearchConfig(pool_size=4, rungs=[SearchRung(steps=10, keep=1)], reroll_duplicates=False)
    report = run_search(_render, _score, cfg, base_seed=2)
    assert report.winner_seed == 5  # best quality among seeds 2..5
    assert report.total_nfe == 40
