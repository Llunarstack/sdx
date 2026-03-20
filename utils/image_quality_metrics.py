from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image


def _to_grayscale_f32(image: Image.Image) -> np.ndarray:
    # Convert to luminance in a cheap way.
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)
    return arr


def laplacian_sharpness(image: Image.Image) -> float:
    """
    Sharpness score = variance of a Laplacian response.
    Higher = sharper. Pure numpy/PIL (no scipy/cv2 required).
    """
    g = _to_grayscale_f32(image)
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    # reflect pad then apply 4-neighborhood Laplacian:
    # lap = 4*center - left - right - up - down
    p = np.pad(g, pad_width=1, mode="reflect")
    center = p[1:-1, 1:-1]
    lap = 4.0 * center - p[1:-1, 0:-2] - p[1:-1, 2:] - p[0:-2, 1:-1] - p[2:, 1:-1]
    return float(np.var(lap))


def grayscale_contrast(image: Image.Image) -> float:
    """Contrast score = stddev of grayscale pixels (higher = more contrast)."""
    g = _to_grayscale_f32(image)
    if g.size == 0:
        return 0.0
    return float(np.std(g))


def analyze_image_quality(image: Image.Image) -> Dict[str, float]:
    """Return a small dict of image quality metrics."""
    return {
        "sharpness": laplacian_sharpness(image),
        "contrast": grayscale_contrast(image),
    }

