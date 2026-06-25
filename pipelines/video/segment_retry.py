"""Retry failed segments with adjusted parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .segment_processor import process_segment
from .types import SegmentAssignment, VideoPlan

__all__ = ["RetryPolicy", "process_segment_with_retry"]


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 2
    min_temporal: float = 0.35
    min_sharpness: float = 0.08
    strength_decay: float = 0.08
    temporal_boost: float = 0.04


def process_segment_with_retry(
    assignment: SegmentAssignment,
    plan: VideoPlan,
    work_dir: str | Path,
    *,
    ckpt: str = "",
    dry_run: bool = False,
    sample_extra_args: Optional[Sequence[str]] = None,
    control_plan: Any = None,
    policy: Optional[RetryPolicy] = None,
    provenance: Any = None,
    process_options: Optional[Any] = None,
) -> tuple[List[Path], Any]:
    pol = policy or RetryPolicy()
    best_frames: List[Path] = []
    best_q: Any = None
    wd = Path(work_dir)

    for attempt in range(max(1, pol.max_attempts)):
        attempt_dir = wd / f"attempt_{attempt:02d}"
        asg = assignment
        if attempt > 0:
            asg.edit_strength = max(0.35, float(asg.edit_strength) - pol.strength_decay * attempt)
        opts_adj = process_options
        if attempt > 0 and process_options is not None:
            opts_adj = replace(
                process_options,
                temporal_alpha=min(0.28, process_options.temporal_alpha + pol.temporal_boost * attempt),
                temporal_smooth=min(5, process_options.temporal_smooth + 1),
            )
        frames, q = process_segment(
            asg,
            plan,
            attempt_dir,
            ckpt=ckpt,
            dry_run=dry_run,
            provenance=provenance,
            sample_extra_args=sample_extra_args,
            control_plan=control_plan,
            process_options=opts_adj,
        )
        if best_q is None or q.temporal_score > best_q.temporal_score:
            best_frames, best_q = frames, q
        if q.passed or q.temporal_score >= pol.min_temporal:
            return frames, q
    return best_frames, best_q
