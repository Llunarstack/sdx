"""Optional post color grade on frame sequences."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["apply_grade_to_sequence", "grade_frame"]

_PRESETS = {
    "cinematic": {"contrast": 1.08, "saturation": 0.92, "warmth": 8},
    "teal_orange": {"contrast": 1.12, "saturation": 1.05, "warmth": 12, "shadow_teal": 0.06},
    "muted": {"contrast": 0.95, "saturation": 0.78, "warmth": 0},
    "vibrant": {"contrast": 1.05, "saturation": 1.18, "warmth": 4},
}


def grade_frame(rgb: np.ndarray, preset: str = "cinematic") -> np.ndarray:
    p = _PRESETS.get(preset.lower(), _PRESETS["cinematic"])
    x = rgb.astype(np.float32)
    c = float(p.get("contrast", 1.0))
    x = (x - 127.5) * c + 127.5
    sat = float(p.get("saturation", 1.0))
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * sat
    warm = float(p.get("warmth", 0))
    x[..., 0] = x[..., 0] + warm
    x[..., 2] = x[..., 2] - warm * 0.35
    te = float(p.get("shadow_teal", 0))
    if te > 0:
        shadow = (1.0 - x / 255.0).mean(axis=2, keepdims=True)
        x[..., 1] = x[..., 1] + shadow[..., 0] * te * 40
        x[..., 2] = x[..., 2] + shadow[..., 0] * te * 55
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_grade_to_sequence(frame_paths: List[Path], preset: str) -> List[Path]:
    if not preset or preset.lower() in ("none", "off"):
        return frame_paths
    for fp in frame_paths:
        save_frame_rgb(fp, grade_frame(read_frame_rgb(fp), preset))
    return frame_paths
