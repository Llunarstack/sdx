"""
Element Binding — multi-reference identity anchors (Kling-inspired, detector-free).

Closed models still morph faces past ~60° turns and lose wardrobe color mid-clip.
We build a cheap multi-ref fingerprint (HSV hist + spatial chroma grid) and:

1. Score every frame against the bound identity.
2. Soft-reanchor drifted frames toward the nearest high-scoring frame's subject band.

No face detector required — works offline and on product/prop subjects too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = [
    "IdentityFingerprint",
    "IdentityBindReport",
    "build_fingerprint",
    "bind_from_refs",
    "score_identity_bind",
    "apply_identity_bind",
]


@dataclass(slots=True)
class IdentityFingerprint:
    """Compact appearance prior from one or more reference crops."""

    hsv_hist: np.ndarray  # (48,)
    chroma_grid: np.ndarray  # (4, 4, 3) mean RGB in subject band
    band: tuple[float, float, float, float] = (0.12, 0.08, 0.88, 0.78)  # x0,y0,x1,y1 norm


@dataclass(slots=True)
class IdentityBindReport:
    score: float
    drift_frames: int = 0
    repaired: int = 0
    per_frame: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _subject_band(rgb: np.ndarray, band: tuple[float, float, float, float]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = band
    xa, xb = int(w * x0), int(w * x1)
    ya, yb = int(h * y0), int(h * y1)
    xa, xb = max(0, min(w - 1, xa)), max(1, min(w, xb))
    ya, yb = max(0, min(h - 1, ya)), max(1, min(h, yb))
    return rgb[ya:yb, xa:xb]


def _hsv_hist(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(rgb[..., :3].astype(np.uint8)).convert("HSV")
    arr = np.asarray(im, dtype=np.float32)
    h = arr[..., 0].ravel()
    s = arr[..., 1].ravel()
    v = arr[..., 2].ravel()
    hh, _ = np.histogram(h, bins=bins, range=(0, 255), density=True)
    hs, _ = np.histogram(s, bins=bins, range=(0, 255), density=True)
    hv, _ = np.histogram(v, bins=bins, range=(0, 255), density=True)
    hist = np.concatenate([hh, hs, hv]).astype(np.float32)
    hist /= hist.sum() + 1e-8
    return hist


def _chroma_grid(rgb: np.ndarray, grid: int = 4) -> np.ndarray:
    from PIL import Image

    small = np.asarray(Image.fromarray(rgb[..., :3].astype(np.uint8)).resize((grid * 8, grid * 8)), dtype=np.float32)
    out = np.zeros((grid, grid, 3), dtype=np.float32)
    cell = 8
    for gy in range(grid):
        for gx in range(grid):
            patch = small[gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell]
            out[gy, gx] = patch.reshape(-1, 3).mean(axis=0)
    return out / 255.0


def build_fingerprint(rgb: np.ndarray, *, band: tuple[float, float, float, float] | None = None) -> IdentityFingerprint:
    b = band or (0.12, 0.08, 0.88, 0.78)
    crop = _subject_band(rgb, b)
    if crop.size == 0:
        crop = rgb
    return IdentityFingerprint(hsv_hist=_hsv_hist(crop), chroma_grid=_chroma_grid(crop), band=b)


def bind_from_refs(
    refs: list[str | Path] | list[np.ndarray],
    *,
    band: tuple[float, float, float, float] | None = None,
) -> IdentityFingerprint:
    """Average fingerprints across multiple angle/expression refs (Element Binding)."""
    fps: list[IdentityFingerprint] = []
    for r in refs:
        if isinstance(r, (str, Path)):
            rgb = read_frame_rgb(r)
        else:
            rgb = np.asarray(r)
        fps.append(build_fingerprint(rgb, band=band))
    if not fps:
        raise ValueError("bind_from_refs requires at least one reference")
    hist = np.mean([f.hsv_hist for f in fps], axis=0)
    hist /= hist.sum() + 1e-8
    grid = np.mean([f.chroma_grid for f in fps], axis=0)
    return IdentityFingerprint(hsv_hist=hist.astype(np.float32), chroma_grid=grid.astype(np.float32), band=fps[0].band)


def _similarity(fp: IdentityFingerprint, rgb: np.ndarray) -> float:
    other = build_fingerprint(rgb, band=fp.band)
    hist_sim = float(np.sqrt(fp.hsv_hist * other.hsv_hist).sum())
    chroma = 1.0 - float(np.mean(np.abs(fp.chroma_grid - other.chroma_grid)))
    return float(np.clip(0.55 * hist_sim + 0.45 * chroma, 0.0, 1.0))


def score_identity_bind(
    frame_paths: list[Path] | list[str],
    fingerprint: IdentityFingerprint | None = None,
    *,
    anchor: str | Path | None = None,
    refs: list[str | Path] | None = None,
    sample_every: int = 2,
    drift_threshold: float = 0.62,
) -> IdentityBindReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 1:
        return IdentityBindReport(score=1.0)
    if fingerprint is None:
        if refs:
            fingerprint = bind_from_refs(refs)
        elif anchor and Path(anchor).is_file():
            fingerprint = build_fingerprint(read_frame_rgb(anchor))
        else:
            fingerprint = build_fingerprint(read_frame_rgb(paths[0]))
    idxs = list(range(0, len(paths), max(1, sample_every)))
    if idxs[-1] != len(paths) - 1:
        idxs.append(len(paths) - 1)
    scores = [_similarity(fingerprint, read_frame_rgb(paths[i])) for i in idxs]
    full = [0.0] * len(paths)
    for j, i in enumerate(idxs):
        full[i] = scores[j]
    for i in range(len(paths)):
        if full[i] == 0.0:
            nearest = min(idxs, key=lambda k: abs(k - i))
            full[i] = full[nearest]
    drift = sum(1 for s in scores if s < drift_threshold)
    mean = float(np.mean(scores)) if scores else 1.0
    notes = []
    if drift:
        notes.append(f"identity_drift_frames={drift}")
    return IdentityBindReport(score=mean, drift_frames=drift, per_frame=full, notes=notes)


def apply_identity_bind(
    frame_paths: list[Path],
    *,
    fingerprint: IdentityFingerprint | None = None,
    anchor: str | Path | None = None,
    refs: list[str | Path] | None = None,
    strength: float = 0.45,
    drift_threshold: float = 0.62,
) -> IdentityBindReport:
    """Re-anchor drifted frames toward the best-matching frame's subject band."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return IdentityBindReport(score=1.0)
    report = score_identity_bind(
        paths, fingerprint, anchor=anchor, refs=refs, sample_every=1, drift_threshold=drift_threshold
    )
    fp = fingerprint
    if fp is None:
        if refs:
            fp = bind_from_refs(refs)
        elif anchor and Path(anchor).is_file():
            fp = build_fingerprint(read_frame_rgb(anchor))
        else:
            fp = build_fingerprint(read_frame_rgb(paths[0]))
    donor_i = int(np.argmax(report.per_frame)) if report.per_frame else 0
    donor = read_frame_rgb(paths[donor_i])
    donor_band = _subject_band(donor, fp.band)
    repaired = 0
    from PIL import Image

    for i, p in enumerate(paths):
        if report.per_frame[i] >= drift_threshold:
            continue
        curr = read_frame_rgb(p)
        h, w = curr.shape[:2]
        x0, y0, x1, y1 = fp.band
        xa, xb = int(w * x0), int(w * x1)
        ya, yb = int(h * y0), int(h * y1)
        patch = np.asarray(Image.fromarray(donor_band).resize((max(1, xb - xa), max(1, yb - ya)), Image.BILINEAR))
        region = curr[ya:yb, xa:xb].astype(np.float32)
        a = float(np.clip(strength, 0.0, 1.0))
        if patch.shape[:2] == region.shape[:2]:
            curr[ya:yb, xa:xb] = np.clip(region * (1.0 - a) + patch.astype(np.float32) * a, 0, 255).astype(np.uint8)
            save_frame_rgb(p, curr)
            repaired += 1
    after = score_identity_bind(paths, fp, sample_every=1, drift_threshold=drift_threshold)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"identity_bind_repaired={repaired}")
    return after
