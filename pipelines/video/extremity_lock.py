"""
Extremity lock — temporal hand / limb coherence (Kling body-part module analog).

Hands are still the #1 user hate across Veo / Kling / Runway: melted fingers,
topology pops, and structure that appears then dissolves. We:

1. Score each frame with the cheap hand-structure heuristic.
2. Flag temporal collapses (good → mush).
3. Soft-blend extremity bands from the nearest high-score frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = [
    "ExtremityReport",
    "score_extremity_coherence",
    "apply_extremity_lock",
]


@dataclass(slots=True)
class ExtremityReport:
    score: float
    mean_hand: float = 1.0
    stability: float = 1.0
    collapsed_frames: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _hand_score(rgb: np.ndarray) -> float:
    from utils.quality.hand_verifier import score_hand_structure

    return float(score_hand_structure(rgb))


def score_extremity_coherence(
    frame_paths: list[Path] | list[str],
    *,
    sample_every: int = 1,
    collapse_drop: float = 0.22,
) -> ExtremityReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return ExtremityReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    scores = [_hand_score(read_frame_rgb(paths[i])) for i in idxs]
    mean_h = float(np.mean(scores))
    # Stability: 1 - normalized consecutive drop magnitude
    drops = [max(0.0, scores[i - 1] - scores[i]) for i in range(1, len(scores))]
    collapsed = sum(1 for d in drops if d >= collapse_drop)
    mean_drop = float(np.mean(drops)) if drops else 0.0
    stability = float(np.clip(1.0 - mean_drop * 2.5 - collapsed * 0.08, 0.0, 1.0))
    score = float(np.clip(0.55 * mean_h + 0.45 * stability, 0.0, 1.0))
    notes = []
    if collapsed:
        notes.append(f"hand_collapses={collapsed}")
    if mean_h < 0.4:
        notes.append(f"weak_hand_structure={mean_h:.2f}")
    return ExtremityReport(score=score, mean_hand=mean_h, stability=stability, collapsed_frames=collapsed, notes=notes)


def _extremity_mask(h: int, w: int) -> np.ndarray:
    """Lower 45% + side margins — where hands/arms usually live."""
    m = np.zeros((h, w), dtype=np.float32)
    m[int(h * 0.55) :, :] = 1.0
    m[:, : max(1, w // 4)] = np.maximum(m[:, : max(1, w // 4)], 0.85)
    m[:, -max(1, w // 4) :] = np.maximum(m[:, -max(1, w // 4) :], 0.85)
    return m


def apply_extremity_lock(
    frame_paths: list[Path],
    *,
    strength: float = 0.5,
    collapse_drop: float = 0.22,
) -> ExtremityReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return ExtremityReport(score=1.0)
    scores = [_hand_score(read_frame_rgb(p)) for p in paths]
    donor_i = int(np.argmax(scores))
    donor = read_frame_rgb(paths[donor_i]).astype(np.float32)
    h, w = donor.shape[:2]
    mask = _extremity_mask(h, w)[..., None]
    repaired = 0
    for i, p in enumerate(paths):
        if i == 0:
            continue
        drop = scores[i - 1] - scores[i]
        if drop < collapse_drop and scores[i] >= 0.35:
            continue
        # Prefer previous good frame if better than global donor for local continuity
        src_i = i - 1 if scores[i - 1] > scores[donor_i] * 0.9 else donor_i
        src = read_frame_rgb(paths[src_i]).astype(np.float32)
        curr = read_frame_rgb(p).astype(np.float32)
        a = float(np.clip(strength, 0.0, 1.0))
        out = curr * (1.0 - mask * a) + src * (mask * a)
        save_frame_rgb(p, np.clip(out, 0, 255).astype(np.uint8))
        repaired += 1
    after = score_extremity_coherence(paths, collapse_drop=collapse_drop)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"extremity_repaired={repaired}")
    return after
