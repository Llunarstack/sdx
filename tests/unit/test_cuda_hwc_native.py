"""Optional ``sdx_cuda_hwc_to_chw`` DLL vs NumPy reference."""

from __future__ import annotations

import numpy as np
import pytest

from sdx_native.cuda_hwc_to_chw import maybe_u8_hwc_to_chw_f32_cuda, u8_hwc_to_chw_f32_numpy
from sdx_native.native_tools import cuda_hwc_to_chw_shared_library_path


def test_cuda_hwc_matches_numpy_small():
    if cuda_hwc_to_chw_shared_library_path() is None:
        pytest.skip("sdx_cuda_hwc_to_chw not built (cmake -DSDX_BUILD_CUDA=ON)")
    hwc = np.random.default_rng(0).integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    ref = u8_hwc_to_chw_f32_numpy(hwc)
    got = maybe_u8_hwc_to_chw_f32_cuda(hwc)
    assert got is not None
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)
