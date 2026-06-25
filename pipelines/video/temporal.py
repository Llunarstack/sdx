"""Temporal consistency passes on frame sequences."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = [
    "TemporalHarmonizer",
    "apply_temporal_smoothing",
    "harmonize_frame_sequence",
]


class TemporalHarmonizer:
    """
    Pixel-level temporal harmonizer (latent stub lives in inference_research_hooks).

    Blends each frame toward previous with optional flow-aware edge preservation.
    """

    def __init__(self, alpha_prev: float = 0.12) -> None:
        self.alpha_prev = float(max(0.0, min(0.95, alpha_prev)))

    def harmonize_pair(self, prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> np.ndarray:
        a = self.alpha_prev
        return np.clip((1.0 - a) * curr_rgb.astype(np.float32) + a * prev_rgb.astype(np.float32), 0, 255).astype(
            np.uint8
        )

    def harmonize_sequence(self, frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        out = [frames[0]]
        for i in range(1, len(frames)):
            out.append(self.harmonize_pair(out[-1], frames[i]))
        return out


def apply_temporal_smoothing(
    frame_paths: List[Path],
    *,
    alpha: float = 0.15,
    window: int = 3,
) -> List[Path]:
    """Moving average in RGB space (cheap flicker reduction)."""
    if len(frame_paths) <= 1:
        return frame_paths
    window = max(1, int(window))
    rgbs = [read_frame_rgb(p).astype(np.float32) for p in frame_paths]
    smoothed: List[np.ndarray] = []
    for i in range(len(rgbs)):
        lo = max(0, i - window)
        hi = min(len(rgbs), i + window + 1)
        stack = np.stack(rgbs[lo:hi], axis=0)
        mean = stack.mean(axis=0)
        if alpha > 0 and i > 0:
            mean = (1.0 - alpha) * rgbs[i] + alpha * smoothed[-1].astype(np.float32)
        smoothed.append(np.clip(mean, 0, 255).astype(np.uint8))
    for p, rgb in zip(frame_paths, smoothed):
        save_frame_rgb(p, rgb)
    return frame_paths


def harmonize_frame_sequence(
    frame_paths: List[Path],
    *,
    alpha_prev: float = 0.12,
    smooth_window: int = 0,
) -> List[Path]:
    th = TemporalHarmonizer(alpha_prev=alpha_prev)
    rgbs = th.harmonize_sequence([read_frame_rgb(p) for p in frame_paths])
    for p, rgb in zip(frame_paths, rgbs):
        save_frame_rgb(p, rgb)
    if smooth_window > 0:
        apply_temporal_smoothing(frame_paths, alpha=0.08, window=smooth_window)
    return frame_paths
