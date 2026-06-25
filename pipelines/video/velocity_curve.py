"""Ease-in-out retiming for frame sequences."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["ease_indices", "apply_velocity_ease"]


def _smoothstep(t: float) -> float:
    t = float(max(0.0, min(1.0, t)))
    return t * t * (3.0 - 2.0 * t)


def ease_indices(source_count: int, target_count: int, *, ease: str = "smooth") -> List[int]:
    if source_count <= 0 or target_count <= 0:
        return []
    if source_count == target_count:
        return list(range(source_count))
    out: List[int] = []
    for t in range(target_count):
        u = t / max(1, target_count - 1)
        if ease == "smooth":
            u = _smoothstep(u)
        elif ease == "ease_in":
            u = u * u
        elif ease == "ease_out":
            u = 1.0 - (1.0 - u) ** 2
        src = int(round(u * (source_count - 1)))
        out.append(min(source_count - 1, max(0, src)))
    return out


def apply_velocity_ease(frame_paths: List[Path], *, ease: str = "smooth") -> List[Path]:
    """Retime in-place with smoothstep easing (cinematic acceleration)."""
    if len(frame_paths) < 3:
        return frame_paths
    rgbs = [read_frame_rgb(p) for p in frame_paths]
    idx = ease_indices(len(rgbs), len(rgbs), ease=ease)
    for p, j in zip(frame_paths, idx):
        save_frame_rgb(p, rgbs[j])
    return frame_paths
