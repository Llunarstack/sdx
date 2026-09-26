"""
Glyph lock — temporal logo / on-screen text stability.

Product shots (and title cards) melt letters and warp logos — a top complaint
on Kling/Veo product b-roll. We:

1. Find high-frequency "glyph-ish" ROIs on the first good frame.
2. Track edge-hash distance over time (Hamming on binarized edges).
3. Freeze melted ROIs by reinjecting the donor glyph patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = [
    "GlyphReport",
    "find_glyph_rois",
    "score_glyph_stability",
    "apply_glyph_lock",
]


@dataclass(slots=True)
class GlyphROI:
    y0: int
    x0: int
    y1: int
    x1: int
    edge_hash: np.ndarray  # flat binary


@dataclass(slots=True)
class GlyphReport:
    score: float
    melt_events: int = 0
    repaired: int = 0
    rois: int = 0
    notes: list[str] = field(default_factory=list)


def _edge_map(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
    gy[1:, :] = np.abs(g[1:, :] - g[:-1, :])
    return gx + gy


def _hash_patch(edge: np.ndarray, y0: int, x0: int, y1: int, x1: int, size: int = 16) -> np.ndarray:
    from PIL import Image

    patch = edge[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((size * size,), dtype=np.uint8)
    im = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    # Fixed mid threshold — mean-adaptive hashes are too invariant under mush
    thr = max(6.0, float(np.percentile(arr, 60)))
    return (arr > thr).astype(np.uint8).ravel()


def find_glyph_rois(rgb: np.ndarray, *, max_rois: int = 4, grid: int = 6) -> list[GlyphROI]:
    """Peak high-frequency cells as logo/text candidates."""
    edge = _edge_map(rgb)
    h, w = edge.shape
    cell_h, cell_w = max(1, h // grid), max(1, w // grid)
    cands: list[tuple[float, int, int, int, int]] = []
    global_mean = float(edge.mean()) + 1e-6
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * cell_h, gx * cell_w
            y1, x1 = min(h, y0 + cell_h), min(w, x0 + cell_w)
            patch = edge[y0:y1, x0:x1]
            energy = float(patch.mean())
            contrast = energy / global_mean
            # Glyphs: locally elevated edge energy vs frame
            if energy < 5.0 or contrast < 1.35:
                continue
            cands.append((energy * contrast, y0, x0, y1, x1))
    cands.sort(key=lambda t: -t[0])
    rois: list[GlyphROI] = []
    for _, y0, x0, y1, x1 in cands[:max_rois]:
        rois.append(GlyphROI(y0=y0, x0=x0, y1=y1, x1=x1, edge_hash=_hash_patch(edge, y0, x0, y1, x1)))
    return rois


def _hamming(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return 1.0
    return float(np.mean(a[:n] != b[:n]))


def score_glyph_stability(
    frame_paths: list[Path] | list[str],
    *,
    melt_threshold: float = 0.28,
    sample_every: int = 2,
) -> GlyphReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return GlyphReport(score=1.0)
    donor = read_frame_rgb(paths[0])
    rois = find_glyph_rois(donor)
    if not rois:
        return GlyphReport(score=1.0, notes=["no_glyph_rois"])
    melts = 0
    dists: list[float] = []
    idxs = list(range(0, len(paths), max(1, sample_every)))
    for i in idxs[1:]:
        edge = _edge_map(read_frame_rgb(paths[i]))
        for roi in rois:
            h = _hash_patch(edge, roi.y0, roi.x0, roi.y1, roi.x1)
            d = _hamming(roi.edge_hash, h)
            dists.append(d)
            if d >= melt_threshold:
                melts += 1
    mean_d = float(np.mean(dists)) if dists else 0.0
    score = float(np.clip(1.0 - mean_d * 2.2 - melts * 0.04, 0.0, 1.0))
    notes = []
    if melts:
        notes.append(f"glyph_melts={melts}")
    return GlyphReport(score=score, melt_events=melts, rois=len(rois), notes=notes)


def apply_glyph_lock(
    frame_paths: list[Path],
    *,
    strength: float = 0.7,
    melt_threshold: float = 0.28,
) -> GlyphReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return GlyphReport(score=1.0)
    donor = read_frame_rgb(paths[0])
    rois = find_glyph_rois(donor)
    if not rois:
        return GlyphReport(score=1.0, notes=["no_glyph_rois"])
    repaired = 0
    for p in paths[1:]:
        curr = read_frame_rgb(p)
        edge = _edge_map(curr)
        changed = False
        for roi in rois:
            h = _hash_patch(edge, roi.y0, roi.x0, roi.y1, roi.x1)
            if _hamming(roi.edge_hash, h) < melt_threshold:
                continue
            patch = donor[roi.y0 : roi.y1, roi.x0 : roi.x1]
            region = curr[roi.y0 : roi.y1, roi.x0 : roi.x1].astype(np.float32)
            a = float(np.clip(strength, 0.0, 1.0))
            curr[roi.y0 : roi.y1, roi.x0 : roi.x1] = np.clip(
                region * (1.0 - a) + patch.astype(np.float32) * a, 0, 255
            ).astype(np.uint8)
            changed = True
            repaired += 1
        if changed:
            save_frame_rgb(p, curr)
    after = score_glyph_stability(paths, melt_threshold=melt_threshold)
    after.repaired = repaired
    after.rois = len(rois)
    if repaired:
        after.notes.append(f"glyph_repaired={repaired}")
    return after
