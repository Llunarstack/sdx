"""NumPy reference path for optional CUDA HWC→CHW (no GPU build required)."""

from __future__ import annotations

import numpy as np

from sdx_native.cuda_hwc_to_chw import u8_hwc_to_chw_f32_numpy


def test_u8_hwc_to_chw_numpy_layout():
    hwc = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)  # 1x2x3
    chw = u8_hwc_to_chw_f32_numpy(hwc)
    assert chw.shape == (3, 1, 2)
    assert chw[0, 0, 0] == 1.0 and chw[1, 0, 1] == 1.0
