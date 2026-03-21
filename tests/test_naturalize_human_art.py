from __future__ import annotations

import numpy as np
from utils.quality import naturalize


def test_naturalize_human_art_modifies_image() -> None:
    # Simple gradient image: shape (H, W, C) in uint8.
    h, w = 64, 64
    x = np.linspace(0, 255, w, dtype=np.float32)
    grad = np.tile(x[None, :], (h, 1))
    img = np.stack([grad, grad * 0.95, grad * 0.8], axis=2).clip(0, 255).astype(np.uint8)

    out = naturalize(img, grain_amount=0.015, micro_contrast=1.02, seed=123)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Should change at least some pixels.
    assert np.any(out != img)
    # Keep values within valid range.
    assert out.min() >= 0 and out.max() <= 255
