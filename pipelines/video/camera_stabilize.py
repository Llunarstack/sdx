"""Stabilize shaky reference footage before keyframe extraction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .motion import compute_flow_magnitude
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["stabilize_frame_paths", "stabilize_rgb_sequence"]


def _shift_rgb(rgb: np.ndarray, dx: float, dy: float) -> np.ndarray:
    try:
        import cv2

        h, w = rgb.shape[:2]
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(rgb, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    except ImportError:
        return rgb


def stabilize_rgb_sequence(frames: Sequence[np.ndarray], *, strength: float = 0.65) -> List[np.ndarray]:
    """Compensate cumulative camera pan using mean optical flow."""
    if len(frames) < 2:
        return list(frames)
    s = float(max(0.0, min(1.0, strength)))
    out: List[np.ndarray] = [frames[0]]
    cum_x, cum_y = 0.0, 0.0
    prev = frames[0]
    for curr in frames[1:]:
        try:
            flow, _ = compute_flow_magnitude(prev, curr)
            dx = float(np.mean(flow[..., 0]))
            dy = float(np.mean(flow[..., 1]))
            cum_x += dx * s
            cum_y += dy * s
            stabilized = _shift_rgb(curr, -cum_x, -cum_y)
        except Exception:
            stabilized = curr
        out.append(stabilized)
        prev = curr
    return out


def stabilize_frame_paths(frame_paths: List[Path], *, strength: float = 0.65) -> List[Path]:
    if len(frame_paths) < 2:
        return frame_paths
    rgbs = [read_frame_rgb(p) for p in frame_paths]
    stable = stabilize_rgb_sequence(rgbs, strength=strength)
    for p, rgb in zip(frame_paths, stable):
        save_frame_rgb(p, rgb)
    return frame_paths
