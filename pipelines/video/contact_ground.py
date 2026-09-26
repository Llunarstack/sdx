"""
Contact ground — kill the floating-character look competitors still ship.

Hailuo/Kling/Veo all produce feet that hover or clip. We:

1. Detect subject mass above the floor band.
2. Score missing contact shadows (AO darkening at foot/ground).
3. Paint soft contact shadows + slight ground darkening under feet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["ContactReport", "score_contact_ground", "apply_contact_ground"]


@dataclass(slots=True)
class ContactReport:
    score: float
    float_events: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _foot_band(rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Lower 22% of frame as candidate foot/ground interface."""
    h, w = rgb.shape[:2]
    y0 = int(h * 0.78)
    return rgb[y0:, :], y0, h


def _subject_mask_lower(rgb: np.ndarray) -> np.ndarray:
    """Cheap subject proxy: pixels darker/contrasty vs floor median."""
    band, _, _ = _foot_band(rgb)
    gray = band.astype(np.float32).mean(axis=2)
    med = float(np.median(gray))
    # Subject feet usually darker or much brighter than flat floor
    mask = (np.abs(gray - med) > 18.0).astype(np.float32)
    return mask


def score_contact_ground(frame_paths: list[Path] | list[str], *, sample_every: int = 2) -> ContactReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        return ContactReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    floats = 0
    scores: list[float] = []
    for i in idxs:
        rgb = read_frame_rgb(paths[i])
        band, y0, h = _foot_band(rgb)
        gray = band.astype(np.float32).mean(axis=2)
        mask = _subject_mask_lower(rgb)
        if mask.mean() < 0.02:
            scores.append(0.7)
            continue
        # Halo just below subject mass (contact AO zone)
        bh, bw = gray.shape
        # Bottom row of subject columns
        cols = np.where(mask.max(axis=0) > 0.5)[0]
        if cols.size == 0:
            scores.append(0.7)
            continue
        # For each subject column, compare pixel under foot tip vs floor further down
        aos: list[float] = []
        for c in cols[:: max(1, len(cols) // 8)]:
            rows = np.where(mask[:, c] > 0.5)[0]
            if rows.size == 0:
                continue
            tip = int(rows.max())
            under = tip + 1
            far = min(bh - 1, tip + max(3, bh // 6))
            if under >= bh:
                continue
            aos.append(float(gray[far, c] - gray[under, c]) / 255.0)
        if not aos:
            scores.append(0.6)
            continue
        ao = float(np.mean(aos))
        # Positive ao = under tip darker than far floor = contact shadow present
        s = float(np.clip(0.4 + ao * 5.0, 0.0, 1.0))
        if ao < 0.015:
            floats += 1
        scores.append(s)
    mean = float(np.mean(scores)) if scores else 1.0
    notes = [f"float_events={floats}"] if floats else []
    return ContactReport(score=mean, float_events=floats, notes=notes)


def apply_contact_ground(
    frame_paths: list[Path],
    *,
    strength: float = 0.55,
) -> ContactReport:
    """Paint soft elliptical contact shadows under lower-band subject mass."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        return ContactReport(score=1.0)
    repaired = 0
    a = float(np.clip(strength, 0.0, 1.0))
    for p in paths:
        rgb = read_frame_rgb(p).astype(np.float32)
        h, w = rgb.shape[:2]
        mask = _subject_mask_lower(rgb.astype(np.uint8))
        if mask.mean() < 0.015:
            continue
        # Centroid of foot mass
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        # Shadow sits just below foot mass in full frame coords
        y0 = int(h * 0.78)
        cy = y0 + float(ys.mean()) + max(2.0, h * 0.01)
        rx = max(4.0, (xs.max() - xs.min()) * 0.55)
        ry = max(2.0, h * 0.012)
        yy, xx = np.mgrid[0:h, 0:w]
        ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        shadow = np.clip(1.0 - ell, 0.0, 1.0)[..., None] * (0.35 * a)
        # Only darken, never brighten
        out = rgb * (1.0 - shadow)
        save_frame_rgb(p, np.clip(out, 0, 255).astype(np.uint8))
        repaired += 1
    after = score_contact_ground(paths, sample_every=1)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"contact_repaired={repaired}")
    return after
