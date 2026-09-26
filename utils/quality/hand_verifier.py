"""Hand / grip quality heuristics for test-time pick (no detector required).

Uses edge density + asymmetry in lower-third / side bands as a cheap proxy for
melted-hand mush vs articulated structure. Not MediaPipe — good enough to bias BoN.
"""

from __future__ import annotations

import numpy as np


def score_hand_structure(rgb_uint8: np.ndarray) -> float:
    """
    Return score in ~[0, 1]; higher = more local edge structure in hand-ish bands.

    Heuristic: lower 40% of the frame + left/right margins (typical hand locations
    for portraits / full-body). Rewards mid edge density; penalizes both blank mush
    and crunchy noise soup.
    """
    arr = np.asarray(rgb_uint8, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return 0.5
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    h, w = gray.shape
    # Sobel-ish
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    edge = np.sqrt(gx * gx + gy * gy)

    y0 = int(h * 0.55)
    band = edge[y0:, :]
    left = edge[:, : max(1, w // 4)]
    right = edge[:, -max(1, w // 4) :]
    regions = [band, left, right]

    scores: list[float] = []
    for reg in regions:
        m = float(reg.mean())
        # Ideal mid structure ~ 8–40 on 0–255 scale
        if m < 2.0:
            scores.append(0.15)  # mush / blank
        elif m > 80.0:
            scores.append(0.35)  # crunch / noise
        else:
            # Peak around 18
            scores.append(float(np.exp(-((m - 18.0) ** 2) / (2 * 14.0**2))))
    # Asymmetry bonus (perfect mirror faces look fake; hands rarely mirror)
    asym = float(np.abs(left.mean() - right.mean()) / (left.mean() + right.mean() + 1e-3))
    asym_score = float(np.clip(asym * 2.0, 0.0, 1.0))
    base = float(np.mean(scores)) if scores else 0.5
    return float(np.clip(0.75 * base + 0.25 * asym_score, 0.0, 1.0))


__all__ = ["score_hand_structure"]
