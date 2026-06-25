"""Extended transition effects between segments."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .types import TransitionType

__all__ = ["apply_transition", "transition_overlap_frames_fx"]


def transition_overlap_frames_fx(transition: TransitionType, fps: float) -> int:
    t = transition.value if hasattr(transition, "value") else str(transition)
    if t == TransitionType.DISSOLVE.value:
        return max(2, int(round(fps * 0.35)))
    if t == TransitionType.MATCH_ACTION.value:
        return max(1, int(round(fps * 0.12)))
    if t == TransitionType.WHIP.value:
        return max(2, int(round(fps * 0.18)))
    if t == TransitionType.FLASH.value:
        return max(1, int(round(fps * 0.08)))
    if t == TransitionType.DIP.value:
        return max(2, int(round(fps * 0.25)))
    return 0


def _crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.clip((1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32), 0, 255).astype(np.uint8)


def _whip_blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    try:
        import cv2

        shift = int((t - 0.5) * a.shape[1] * 0.35)
        m = np.float32([[1, 0, shift], [0, 1, 0]])
        wa = cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.warpAffine(b, m, (b.shape[1], b.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        return _crossfade(wa, wb, t)
    except ImportError:
        return _crossfade(a, b, t)


def _flash_blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    peak = max(0.0, 1.0 - abs(t - 0.5) * 4.0)
    base = _crossfade(a, b, t)
    flash = np.full_like(base, 255)
    return np.clip(base.astype(np.float32) * (1.0 - peak * 0.55) + flash * (peak * 0.55), 0, 255).astype(np.uint8)


def _dip_blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    if t < 0.5:
        fade = t * 2.0
        return np.clip(a.astype(np.float32) * (1.0 - fade), 0, 255).astype(np.uint8)
    fade = (t - 0.5) * 2.0
    return np.clip(b.astype(np.float32) * fade, 0, 255).astype(np.uint8)


def apply_transition(
    frames_a: Sequence[np.ndarray],
    frames_b: Sequence[np.ndarray],
    transition: TransitionType,
    overlap: int,
) -> List[np.ndarray]:
    if overlap <= 0 or not frames_a or not frames_b:
        return list(frames_a) + list(frames_b)
    overlap = min(overlap, len(frames_a), len(frames_b))
    head = list(frames_a[:-overlap])
    tail = list(frames_b[overlap:])
    blend: List[np.ndarray] = []
    tname = transition.value if hasattr(transition, "value") else str(transition)
    for i in range(overlap):
        t = (i + 1) / (overlap + 1)
        a = frames_a[-overlap + i]
        b = frames_b[i]
        if tname == TransitionType.WHIP.value:
            blend.append(_whip_blend(a, b, t))
        elif tname == TransitionType.FLASH.value:
            blend.append(_flash_blend(a, b, t))
        elif tname == TransitionType.DIP.value:
            blend.append(_dip_blend(a, b, t))
        else:
            blend.append(_crossfade(a, b, t))
    return head + blend + tail
