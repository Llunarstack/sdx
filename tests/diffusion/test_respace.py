"""``diffusion.respace.space_timesteps`` (ddim string vs integer count)."""

import numpy as np

from diffusion.respace import space_timesteps


def _naive_ddim_steps(t: int, n: int) -> np.ndarray:
    for stride in range(1, t):
        steps = np.arange(0, t, stride)
        if len(steps) == n:
            return steps
    return np.linspace(0, t - 1, n, dtype=np.int64)


def test_ddim_matches_naive_scan():
    for t in (16, 100, 256, 1000):
        for n in (1, 2, 5, 10, 25, 50, 100):
            key = f"ddim{n}"
            got = space_timesteps(t, key)
            ref = _naive_ddim_steps(t, n)
            assert np.array_equal(got, ref), (t, n, got.tolist(), ref.tolist())


def test_ddim_arange_length_formula():
    t = 1000
    for stride in range(1, t):
        steps = np.arange(0, t, stride)
        assert len(steps) == (t - 1) // stride + 1
