"""
Occlusion resolve — soft FG/BG layering when subjects cross.

Closed models blend overlapping people into one mushy silhouette. We estimate
a coarse depth proxy (lower = closer for ground-plane bias + brightness) and
prefer nearer pixels when two massy regions overlap mid-clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["OcclusionReport", "score_occlusion_coherence", "apply_occlusion_resolve"]


@dataclass(slots=True)
class OcclusionReport:
    score: float
    conflict_frames: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _depth_proxy(rgb: np.ndarray) -> np.ndarray:
    """Higher = closer. Ground-plane: lower in frame = closer; slight brightness bias."""
    h, w = rgb.shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    gray = rgb.astype(np.float32).mean(axis=2) / 255.0
    return 0.7 * yy + 0.3 * (1.0 - gray)


def _edge_energy(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
    gy[1:, :] = np.abs(g[1:, :] - g[:-1, :])
    return gx + gy


def score_occlusion_coherence(frame_paths: list[Path] | list[str], *, sample_every: int = 2) -> OcclusionReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return OcclusionReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    conflicts = 0
    scores: list[float] = []
    for i in idxs:
        rgb = read_frame_rgb(paths[i])
        e = _edge_energy(rgb)
        # Soft collisions: high edge energy in mid band with low local contrast coherence
        h, w = e.shape
        mid = e[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        dens = float(mid.mean())
        # Penalize extremely dense mid edges (mushy overlaps)
        s = float(np.clip(1.0 - max(0.0, dens - 25.0) / 40.0, 0.0, 1.0))
        if dens > 40.0:
            conflicts += 1
        scores.append(s)
    mean = float(np.mean(scores)) if scores else 1.0
    notes = [f"occlusion_conflicts={conflicts}"] if conflicts else []
    return OcclusionReport(score=mean, conflict_frames=conflicts, notes=notes)


def apply_occlusion_resolve(
    frame_paths: list[Path],
    *,
    strength: float = 0.4,
) -> OcclusionReport:
    """
    Prefer nearer (lower-in-frame) structure when blending with previous frame
    in high-conflict mid regions — reduces silhouette mush across overlaps.
    """
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return OcclusionReport(score=1.0)
    a = float(np.clip(strength, 0.0, 1.0))
    repaired = 0
    prev = read_frame_rgb(paths[0]).astype(np.float32)
    prev_d = _depth_proxy(prev.astype(np.uint8))
    for p in paths[1:]:
        curr = read_frame_rgb(p).astype(np.float32)
        d = _depth_proxy(curr.astype(np.uint8))
        e = _edge_energy(curr.astype(np.uint8))
        thr = float(np.percentile(e, 75))
        conflict = (e >= thr).astype(np.float32)
        # Where current is closer than previous, keep current; else soft-blend prev structure
        closer = (d >= prev_d).astype(np.float32)
        w = conflict * a * (1.0 - closer)
        w3 = w[..., None]
        out = curr * (1.0 - w3) + prev * w3
        if float(w.mean()) > 0.03:
            save_frame_rgb(p, np.clip(out, 0, 255).astype(np.uint8))
            repaired += 1
            curr = out
        prev = curr
        prev_d = _depth_proxy(curr.astype(np.uint8))
    after = score_occlusion_coherence(paths, sample_every=1)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"occlusion_resolved={repaired}")
    return after
