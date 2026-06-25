"""Repair drifted frames via neighbor blending and optional re-edit hints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .semantic_drift import find_drift_frames, score_sequence_drift
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["DriftRepairReport", "repair_sequence_drift", "build_drift_repair_prompt"]


@dataclass(slots=True)
class DriftRepairReport:
    drift_indices: List[int]
    repaired_count: int
    max_drift: float
    mean_drift: float


def build_drift_repair_prompt(base_prompt: str, *, drift_score: float) -> str:
    extra = "consistent identity, stable colors, no morphing"
    if drift_score > 0.75:
        extra += ", match reference exactly, reduce variation"
    return f"{base_prompt}, {extra}".strip(", ")


def repair_sequence_drift(
    frame_paths: List[Path],
    *,
    anchor_path: Optional[str | Path] = None,
    threshold: float = 0.55,
    blend_strength: float = 0.45,
) -> DriftRepairReport:
    """Blend drifted frames toward temporal neighbors + anchor."""
    if len(frame_paths) < 2:
        scores = score_sequence_drift(frame_paths, anchor_path)
        mx = max((s for _, s in scores), default=0.0)
        return DriftRepairReport([], 0, mx, mx)

    anchor = None
    if anchor_path and Path(anchor_path).is_file():
        anchor = read_frame_rgb(anchor_path)
    else:
        anchor = read_frame_rgb(frame_paths[0])

    drift_idx = find_drift_frames(frame_paths, anchor_path=anchor_path, threshold=threshold)
    scores = score_sequence_drift(frame_paths, anchor_path)
    all_scores = [s for _, s in scores]
    repaired = 0
    rgbs = [read_frame_rgb(p) for p in frame_paths]
    b = float(max(0.0, min(1.0, blend_strength)))

    for i in drift_idx:
        refs: List[np.ndarray] = []
        if i > 0:
            refs.append(rgbs[i - 1])
        if i + 1 < len(rgbs):
            refs.append(rgbs[i + 1])
        if not refs:
            continue
        neighbor = np.mean([r.astype(np.float32) for r in refs], axis=0)
        fixed = rgbs[i].astype(np.float32) * (1.0 - b) + neighbor * (b * 0.65) + anchor.astype(np.float32) * (b * 0.35)
        rgbs[i] = np.clip(fixed, 0, 255).astype(np.uint8)
        save_frame_rgb(frame_paths[i], rgbs[i])
        repaired += 1

    return DriftRepairReport(
        drift_indices=drift_idx,
        repaired_count=repaired,
        max_drift=float(max(all_scores) if all_scores else 0.0),
        mean_drift=float(np.mean(all_scores) if all_scores else 0.0),
    )
