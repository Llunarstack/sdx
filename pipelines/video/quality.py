"""Segment quality gates: temporal consistency, sharpness."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .types import SegmentQualityReport
from .video_io import read_frame_rgb

__all__ = ["score_segment_quality", "score_temporal_consistency", "score_frame_sharpness"]


def score_frame_sharpness(rgb: np.ndarray) -> float:
    gray = rgb[..., :3].astype(np.float32).mean(axis=2) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return float(min(1.0, (gx + gy) * 0.5 * 4.0))


def score_temporal_consistency(frame_paths: Sequence[Path]) -> float:
    if len(frame_paths) < 2:
        return 1.0
    diffs: List[float] = []
    prev = read_frame_rgb(frame_paths[0]).astype(np.float32)
    for p in frame_paths[1:]:
        curr = read_frame_rgb(p).astype(np.float32)
        d = float(np.mean(np.abs(curr - prev)) / 255.0)
        diffs.append(d)
        prev = curr
    mean_d = float(np.mean(diffs))
    # Lower diff variance + moderate mean diff = good
    var_d = float(np.var(diffs))
    score = 1.0 - min(1.0, mean_d * 2.5 + var_d * 4.0)
    return max(0.0, score)


def score_segment_quality(
    frame_paths: Sequence[Path],
    *,
    segment_index: int = 0,
    min_temporal: float = 0.35,
    min_sharpness: float = 0.08,
) -> SegmentQualityReport:
    paths = list(frame_paths)
    temporal = score_temporal_consistency(paths)
    sharpness = float(np.mean([score_frame_sharpness(read_frame_rgb(p)) for p in paths[: min(8, len(paths))]]))
    notes: List[str] = []
    if temporal < min_temporal:
        notes.append(f"low temporal consistency ({temporal:.3f})")
    if sharpness < min_sharpness:
        notes.append(f"low sharpness ({sharpness:.3f})")
    passed = temporal >= min_temporal and sharpness >= min_sharpness
    return SegmentQualityReport(
        segment_index=segment_index,
        temporal_score=temporal,
        sharpness_score=sharpness,
        passed=passed,
        notes=notes,
    )
