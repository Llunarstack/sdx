"""Parity: C++ squared-cosine betas vs NumPy (skip if ``sdx_beta_schedules`` not built)."""

import numpy as np
import pytest

from diffusion.schedules import _squared_cosine_beta_schedule_v2_numpy
from sdx_native.native_tools import beta_schedules_shared_library_path


@pytest.fixture(scope="module")
def _native_lib():
    if not beta_schedules_shared_library_path():
        pytest.skip("sdx_beta_schedules not built (cmake in native/cpp)")
    import sdx_native.beta_schedules_native as nbs

    nbs._DLL = None  # type: ignore[attr-defined]
    yield
    nbs._DLL = None  # type: ignore[attr-defined]


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
@pytest.mark.parametrize("max_beta", [0.5, 0.999])
def test_squared_cosine_native_matches_numpy(_native_lib, n: int, max_beta: float) -> None:
    from sdx_native.beta_schedules_native import squared_cosine_betas_v2_native

    py = _squared_cosine_beta_schedule_v2_numpy(n, max_beta)
    nat = squared_cosine_betas_v2_native(n, max_beta)
    assert nat is not None
    assert np.array_equal(py, nat), (np.max(np.abs(py - nat)), py[:5], nat[:5])
