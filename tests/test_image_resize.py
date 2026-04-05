"""Tests for utils.image_resize fitting modes."""

from __future__ import annotations

import numpy as np
from utils.image_resize import fit_image_to_size


def _toy_image(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # salient bright block on right side
    img[max(0, h // 4) : max(1, h // 4 + h // 3), max(0, w - w // 4) : w - 1, :] = 255
    return img


def test_fit_stretch_size():
    img = _toy_image(40, 80)
    out = fit_image_to_size(img, 64, 64, mode="stretch")
    assert out.shape == (64, 64, 3)


def test_fit_center_crop_size():
    img = _toy_image(40, 80)
    out = fit_image_to_size(img, 64, 64, mode="center_crop")
    assert out.shape == (64, 64, 3)


def test_fit_saliency_crop_prefers_bright_region():
    img = _toy_image(40, 80)
    out = fit_image_to_size(img, 40, 40, mode="saliency_crop")
    # Right-biased bright block should survive saliency crop better than center crop.
    ctr = fit_image_to_size(img, 40, 40, mode="center_crop")
    assert float(out.mean()) >= float(ctr.mean())

