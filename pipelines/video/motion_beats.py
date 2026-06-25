"""Place keyframes on motion peaks in reference footage."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .motion import compute_flow_magnitude
from .video_io import read_frame_rgb

__all__ = ["detect_motion_beats", "merge_beat_indices"]


def detect_motion_beats(frame_paths: Sequence[Path], *, min_gap: int = 4) -> List[int]:
    """Return frame indices where motion magnitude is a local peak."""
    n = len(frame_paths)
    if n < 3:
        return [0] if n else []
    mags: List[float] = [0.0]
    for i in range(1, n):
        try:
            _, m = compute_flow_magnitude(read_frame_rgb(frame_paths[i - 1]), read_frame_rgb(frame_paths[i]))
            mags.append(m)
        except Exception:
            mags.append(mags[-1] if mags else 0.0)
    arr = np.array(mags, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()
    peaks: List[int] = [0, n - 1]
    gap = max(1, int(min_gap))
    for i in range(1, n - 1):
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1] and arr[i] > 0.35:
            if not peaks or i - peaks[-1] >= gap:
                peaks.append(i)
    return sorted(set(peaks))


def merge_beat_indices(base: Sequence[int], beats: Sequence[int], total: int) -> List[int]:
    """Union keyframe cadence with motion beat anchors."""
    idx = set(int(i) for i in base if 0 <= int(i) < total)
    idx.update(int(b) for b in beats if 0 <= int(b) < total)
    idx.add(0)
    if total > 0:
        idx.add(total - 1)
    return sorted(idx)
