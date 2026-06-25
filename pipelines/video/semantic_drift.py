"""Detect per-frame semantic / appearance drift in sequences."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .video_io import read_frame_rgb

__all__ = [
    "color_histogram",
    "histogram_distance",
    "frame_drift_score",
    "score_sequence_drift",
    "find_drift_frames",
]


def color_histogram(rgb: np.ndarray, *, bins: int = 16) -> np.ndarray:
    hists: List[np.ndarray] = []
    for c in range(3):
        h, _ = np.histogram(rgb[..., c], bins=bins, range=(0, 256))
        h = h.astype(np.float32)
        h /= h.sum() + 1e-6
        hists.append(h)
    return np.concatenate(hists)


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 = identical, 1 = very different."""
    d = float(np.sum(np.abs(a - b)) * 0.5)
    return min(1.0, max(0.0, d))


def _structural_proxy(a: np.ndarray, b: np.ndarray) -> float:
    """Cheap SSIM proxy via gradient correlation."""
    ga = a.astype(np.float32).mean(axis=2)
    gb = b.astype(np.float32).mean(axis=2)
    ga = ga - ga.mean()
    gb = gb - gb.mean()
    num = float(np.sum(ga * gb))
    den = float(np.sqrt(np.sum(ga * ga) * np.sum(gb * gb)) + 1e-6)
    sim = num / den
    return 1.0 - min(1.0, max(0.0, (sim + 1.0) * 0.5))


def frame_drift_score(
    frame_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    prev_rgb: Optional[np.ndarray] = None,
) -> float:
    """Higher = more drift from reference (and optionally previous frame)."""
    h_ref = color_histogram(reference_rgb)
    h_cur = color_histogram(frame_rgb)
    color_d = histogram_distance(h_cur, h_ref)
    struct_d = _structural_proxy(frame_rgb, reference_rgb)
    score = 0.55 * color_d + 0.45 * struct_d
    if prev_rgb is not None:
        h_prev = color_histogram(prev_rgb)
        temporal_d = histogram_distance(h_cur, h_prev)
        score = 0.7 * score + 0.3 * temporal_d
    return float(min(1.0, max(0.0, score)))


def score_sequence_drift(
    frame_paths: Sequence[Path],
    anchor_path: Optional[str | Path] = None,
) -> List[Tuple[int, float]]:
    if not frame_paths:
        return []
    ref = read_frame_rgb(anchor_path) if anchor_path and Path(anchor_path).is_file() else read_frame_rgb(frame_paths[0])
    prev = None
    out: List[Tuple[int, float]] = []
    for i, fp in enumerate(frame_paths):
        rgb = read_frame_rgb(fp)
        out.append((i, frame_drift_score(rgb, ref, prev_rgb=prev)))
        prev = rgb
    return out


def find_drift_frames(
    frame_paths: Sequence[Path],
    *,
    anchor_path: Optional[str | Path] = None,
    threshold: float = 0.55,
) -> List[int]:
    scores = score_sequence_drift(frame_paths, anchor_path)
    return [i for i, s in scores if s >= threshold]
