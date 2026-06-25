"""Keyframe selection cadence for segment editing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .timeline import frame_count_for_duration

__all__ = ["KeyframeSchedule", "pick_keyframe_indices", "schedule_keyframes_for_segment"]


@dataclass(slots=True)
class KeyframeSchedule:
    segment_index: int
    frame_indices: List[int]
    frame_paths: List[Path]
    interval: int


def pick_keyframe_indices(total_frames: int, *, interval: int = 6, always_include: Sequence[int] = ()) -> List[int]:
    """Every ``interval`` frames plus first/last and optional anchors."""
    if total_frames <= 0:
        return []
    interval = max(1, int(interval))
    indices = {0, total_frames - 1}
    for i in range(0, total_frames, interval):
        indices.add(i)
    for a in always_include:
        if 0 <= int(a) < total_frames:
            indices.add(int(a))
    return sorted(indices)


def schedule_keyframes_for_segment(
    frame_paths: List[Path],
    *,
    segment_index: int,
    duration_sec: float,
    fps: float,
    interval: int = 6,
    beat_indices: Optional[Sequence[int]] = None,
) -> KeyframeSchedule:
    target = frame_count_for_duration(duration_sec, fps)
    if len(frame_paths) > target:
        # subsample path list to target count first
        step = max(1, len(frame_paths) // max(1, target))
        subsampled = [frame_paths[i] for i in range(0, len(frame_paths), step)][:target]
    else:
        subsampled = frame_paths
    idxs = pick_keyframe_indices(len(subsampled), interval=interval)
    if beat_indices:
        from .motion_beats import merge_beat_indices

        idxs = merge_beat_indices(idxs, beat_indices, len(subsampled))
    return KeyframeSchedule(
        segment_index=segment_index,
        frame_indices=idxs,
        frame_paths=[subsampled[i] for i in idxs],
        interval=interval,
    )
