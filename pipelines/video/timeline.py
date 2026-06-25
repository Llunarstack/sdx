"""Master timeline: normalize fps, retime segments, transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .types import MasterTimeline, SegmentAssignment, TransitionType

__all__ = [
    "frame_count_for_duration",
    "normalize_segment_window",
    "retime_frame_indices",
    "transition_overlap_frames",
]


def frame_count_for_duration(duration_sec: float, fps: float) -> int:
    return max(1, int(round(float(duration_sec) * float(fps))))


def normalize_segment_window(
    clip_duration_sec: float,
    target_duration_sec: float,
    *,
    in_sec: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Pick in/out window inside clip and speed factor to hit target duration.

    Returns (in_sec, out_sec, speed_factor).
    """
    avail = max(0.01, clip_duration_sec - in_sec)
    need = max(0.01, target_duration_sec)
    if avail >= need:
        return in_sec, in_sec + need, 1.0
    speed = avail / need
    return in_sec, in_sec + avail, speed


@dataclass(slots=True)
class TimelineSlot:
    segment_index: int
    start_frame: int
    end_frame: int
    transition: TransitionType


def build_master_slots(assignments: List[SegmentAssignment], timeline: MasterTimeline) -> List[TimelineSlot]:
    slots: List[TimelineSlot] = []
    cursor = 0
    for i, seg in enumerate(assignments):
        n = frame_count_for_duration(seg.shot.duration_sec, timeline.fps)
        start = cursor
        end = cursor + n
        slots.append(
            TimelineSlot(
                segment_index=i,
                start_frame=start,
                end_frame=end,
                transition=seg.transition if i > 0 else TransitionType.CUT,
            )
        )
        cursor = end
    return slots


def transition_overlap_frames(transition: TransitionType, fps: float) -> int:
    if transition == TransitionType.DISSOLVE:
        return max(2, int(round(fps * 0.35)))
    if transition == TransitionType.MATCH_ACTION:
        return max(1, int(round(fps * 0.12)))
    return 0


def retime_frame_indices(source_count: int, target_count: int) -> List[int]:
    """Map target frame indices to nearest source indices."""
    if source_count <= 0 or target_count <= 0:
        return []
    if source_count == target_count:
        return list(range(source_count))
    out: List[int] = []
    for t in range(target_count):
        src = int(round(t * (source_count - 1) / max(1, target_count - 1)))
        out.append(min(source_count - 1, max(0, src)))
    return out
