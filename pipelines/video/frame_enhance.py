"""Lightweight sharpening / clarity pass on frame sequences."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["enhance_frame", "enhance_sequence"]


def enhance_frame(rgb: np.ndarray, *, amount: float = 0.35) -> np.ndarray:
    """Unsharp mask."""
    a = float(max(0.0, min(1.0, amount)))
    try:
        import cv2

        blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(rgb, 1.0 + a, blur, -a, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)
    except ImportError:
        return rgb


def enhance_sequence(frame_paths: List[Path], *, amount: float = 0.35) -> List[Path]:
    for p in frame_paths:
        save_frame_rgb(p, enhance_frame(read_frame_rgb(p), amount=amount))
    return frame_paths
