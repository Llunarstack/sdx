"""Exposure / luminance deflicker across frame sequences."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["apply_deflicker", "match_luminance"]


def _frame_luma(rgb: np.ndarray) -> float:
    return float(rgb[..., :3].astype(np.float32).mean())


def match_luminance(rgb: np.ndarray, target_luma: float, *, strength: float = 0.85) -> np.ndarray:
    cur = _frame_luma(rgb)
    if cur < 1e-3:
        return rgb
    gain = target_luma / cur
    gain = 1.0 + (gain - 1.0) * float(max(0.0, min(1.0, strength)))
    out = rgb.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_deflicker(frame_paths: List[Path], *, window: int = 5, strength: float = 0.75) -> List[Path]:
    """Rolling median luminance target reduces flicker from per-frame edits."""
    if not frame_paths:
        return []
    lumas = [_frame_luma(read_frame_rgb(p)) for p in frame_paths]
    w = max(1, int(window))
    targets: List[float] = []
    for i in range(len(lumas)):
        lo = max(0, i - w // 2)
        hi = min(len(lumas), i + w // 2 + 1)
        targets.append(float(np.median(lumas[lo:hi])))
    for p, tgt in zip(frame_paths, targets):
        save_frame_rgb(p, match_luminance(read_frame_rgb(p), tgt, strength=strength))
    return frame_paths
