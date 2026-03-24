"""Parity: C++ ``sdx_it_finalize_path`` vs NumPy ``_resample_length_numpy`` (skip if lib missing)."""

import numpy as np
import pytest

from diffusion.inference_timesteps import _resample_length_numpy
from sdx_native.native_tools import inference_timesteps_shared_library_path


@pytest.fixture(scope="module")
def _native_lib():
    if not inference_timesteps_shared_library_path():
        pytest.skip("sdx_inference_timesteps not built (cmake in native/cpp)")
    import sdx_native.inference_timesteps_native as nit

    nit._DLL = None  # type: ignore[attr-defined]
    yield
    nit._DLL = None  # type: ignore[attr-defined]


def test_native_parity_random_paths(_native_lib):
    from sdx_native.inference_timesteps_native import finalize_inference_timesteps_native

    rng = np.random.default_rng(0)
    for T in (16, 100, 1000):
        for target in (1, 2, 7, 25, 50):
            for _ in range(12):
                raw = rng.integers(0, T, size=rng.integers(0, 40), dtype=np.int64)
                py = _resample_length_numpy(raw, target, T)
                nat = finalize_inference_timesteps_native(raw, target, T)
                assert nat is not None
                assert np.array_equal(py, nat), (T, target, raw.tolist(), py.tolist(), nat.tolist())


def test_native_parity_handcrafted(_native_lib):
    from sdx_native.inference_timesteps_native import finalize_inference_timesteps_native

    T = 100
    cases = [
        (np.array([], dtype=np.int64), 5),
        (np.array([99, 99, 50, 50, 0], dtype=np.int64), 10),
        (np.linspace(0, T - 1, 30, dtype=np.int64)[::-1], 25),
    ]
    for raw, target in cases:
        py = _resample_length_numpy(raw, target, T)
        nat = finalize_inference_timesteps_native(raw, target, T)
        assert nat is not None and np.array_equal(py, nat)
