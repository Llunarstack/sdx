"""
HF shimmer guard — LTX-class high-frequency foliage/texture flicker.

Creators hate shimmering trees/grass from VAE temporal instability. We detect
high-frequency bands whose frame-to-frame energy varies without global motion,
then temporally median-filter those bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["ShimmerReport", "score_hf_shimmer", "apply_hf_deshimmer"]


@dataclass(slots=True)
class ShimmerReport:
    score: float
    shimmer_energy: float = 0.0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _hf_map(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=2)
    # Laplacian-ish high frequency
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return np.abs(lap)


def score_hf_shimmer(frame_paths: list[Path] | list[str], *, sample_every: int = 1) -> ShimmerReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3:
        return ShimmerReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    maps = [_hf_map(read_frame_rgb(paths[i])) for i in idxs]
    # Temporal variance of HF energy
    stack = np.stack(maps, axis=0)
    tvar = stack.var(axis=0)
    # Global motion proxy
    rgbs = [read_frame_rgb(paths[i]).astype(np.float32) for i in idxs]
    gmot = float(np.mean([np.mean(np.abs(rgbs[i] - rgbs[i - 1])) / 255.0 for i in range(1, len(rgbs))]))
    shimmer = float(tvar.mean())
    # Penalize high HF temporal variance when global motion is low
    pen = shimmer / (8.0 + gmot * 40.0)
    score = float(np.clip(1.0 - pen, 0.0, 1.0))
    notes = []
    if score < 0.7:
        notes.append(f"hf_shimmer={shimmer:.2f}")
    return ShimmerReport(score=score, shimmer_energy=shimmer, notes=notes)


def apply_hf_deshimmer(
    frame_paths: list[Path],
    *,
    strength: float = 0.55,
    window: int = 3,
) -> ShimmerReport:
    """Temporal median on high-frequency residual (keeps low-freq structure)."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3:
        return ShimmerReport(score=1.0)
    frames = [read_frame_rgb(p).astype(np.float32) for p in paths]
    w = max(3, int(window) | 1)  # odd
    half = w // 2
    a = float(np.clip(strength, 0.0, 1.0))
    repaired = 0
    for i in range(len(frames)):
        lo = max(0, i - half)
        hi = min(len(frames), i + half + 1)
        neighborhood = np.stack(frames[lo:hi], axis=0)
        med = np.median(neighborhood, axis=0)
        # Only blend where HF temporal variance is high
        hf_stack = np.stack([_hf_map(frames[j].astype(np.uint8)) for j in range(lo, hi)], axis=0)
        tvar = hf_stack.var(axis=0)
        thr = float(np.percentile(tvar, 70))
        mask = (tvar >= thr).astype(np.float32)[..., None] * a
        out = frames[i] * (1.0 - mask) + med * mask
        if float(mask.mean()) > 0.02:
            save_frame_rgb(paths[i], np.clip(out, 0, 255).astype(np.uint8))
            repaired += 1
    after = score_hf_shimmer(paths)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"deshimmered={repaired}")
    return after
