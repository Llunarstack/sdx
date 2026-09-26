"""
Artifact critic — GenVID-inspired axes, but actionable for SDX retry.

Competitors hide failures behind pretty demos. We score Appearance / Motion /
Camera explicitly and fail the segment when thresholds breach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb

__all__ = [
    "ArtifactReport",
    "score_appearance_artifacts",
    "score_motion_artifacts",
    "score_camera_artifacts",
    "score_video_artifacts",
]


@dataclass(slots=True)
class ArtifactReport:
    """Higher scores = cleaner video (fewer artifacts)."""

    appearance: float
    motion: float
    camera: float
    overall: float
    flicker: float = 1.0
    color_drift: float = 1.0
    deformation: float = 1.0
    appear_disappear: float = 1.0
    shake: float = 1.0
    notes: list[str] = field(default_factory=list)


def _sample_paths(paths: list[Path], max_n: int = 24) -> list[Path]:
    if len(paths) <= max_n:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_n).astype(int)
    return [paths[i] for i in idx]


def score_appearance_artifacts(frame_paths: list[Path]) -> tuple[float, float, float, list[str]]:
    """Returns (appearance_score, flicker, color_drift, notes)."""
    paths = _sample_paths([Path(p) for p in frame_paths if Path(p).is_file()])
    notes: list[str] = []
    if len(paths) < 2:
        return 1.0, 1.0, 1.0, notes
    frames = [read_frame_rgb(p).astype(np.float32) for p in paths]
    # Flicker: high-frequency frame-to-frame energy after low-pass
    diffs = []
    for a, b in zip(frames[:-1], frames[1:]):
        diffs.append(float(np.mean(np.abs(a - b)) / 255.0))
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs))
    # Good motion has moderate mean_d; flicker has high var relative to mean
    flicker = float(np.clip(1.0 - (var_d / (mean_d + 1e-4)) * 0.35 - var_d * 8.0, 0.0, 1.0))
    # Alternating / soap-opera chatter: even↔odd oscillation with huge swing
    if len(frames) >= 4:
        even = np.stack([frames[i] for i in range(0, len(frames), 2)])
        odd = np.stack([frames[i] for i in range(1, len(frames), 2)])
        n = min(len(even), len(odd))
        parity = float(np.mean(np.abs(even[:n] - odd[:n])) / 255.0)
        # Penalize strong even/odd disagreement (classic gen flicker)
        if parity > 0.18:
            flicker = float(np.clip(flicker * (1.0 - parity) - parity * 0.5, 0.0, 1.0))
    # Color drift: mean RGB shift across clip
    means = [f.reshape(-1, 3).mean(axis=0) for f in frames]
    drift = float(np.mean(np.abs(means[-1] - means[0])) / 255.0)
    color = float(np.clip(1.0 - drift * 6.0, 0.0, 1.0))
    if flicker < 0.55:
        notes.append(f"flicker={flicker:.2f}")
    if color < 0.55:
        notes.append(f"color_drift={color:.2f}")
    appearance = 0.55 * flicker + 0.45 * color
    return appearance, flicker, color, notes


def score_motion_artifacts(frame_paths: list[Path]) -> tuple[float, float, float, list[str]]:
    """Returns (motion_score, deformation, appear_disappear, notes)."""
    from .permanence import score_object_permanence

    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    notes: list[str] = []
    if len(paths) < 3:
        return 1.0, 1.0, 1.0, notes
    perm = score_object_permanence(paths)
    appear_disappear = float(perm.score)
    if perm.disappear_events or perm.appear_events:
        notes.extend(perm.notes)

    # Deformation proxy: local patch correlation collapse between frames
    sample = _sample_paths(paths, 12)
    corrs: list[float] = []
    for a_p, b_p in zip(sample[:-1], sample[1:]):
        a = read_frame_rgb(a_p).astype(np.float32)
        b = read_frame_rgb(b_p).astype(np.float32)
        # Center crop correlation
        h, w = a.shape[:2]
        y0, x0 = h // 4, w // 4
        y1, x1 = 3 * h // 4, 3 * w // 4
        pa = a[y0:y1, x0:x1].ravel()
        pb = b[y0:y1, x0:x1].ravel()
        pa = (pa - pa.mean()) / (pa.std() + 1e-6)
        pb = (pb - pb.mean()) / (pb.std() + 1e-6)
        corrs.append(float(np.mean(pa * pb)))
    # Extreme correlation drops without being near-zero global change → deform
    deform = float(np.clip(np.mean(corrs) * 0.5 + 0.5, 0.0, 1.0))
    if deform < 0.45:
        notes.append(f"deformation={deform:.2f}")
    motion = 0.55 * appear_disappear + 0.45 * deform
    return motion, deform, appear_disappear, notes


def score_camera_artifacts(frame_paths: list[Path]) -> tuple[float, float, list[str]]:
    """Returns (camera_score, shake, notes)."""
    paths = _sample_paths([Path(p) for p in frame_paths if Path(p).is_file()], 16)
    notes: list[str] = []
    if len(paths) < 3:
        return 1.0, 1.0, notes
    # Global translation proxy via phase-ish centroid of edges
    cents: list[tuple[float, float]] = []
    for p in paths:
        g = read_frame_rgb(p).astype(np.float32).mean(axis=2)
        gy = np.abs(np.diff(g, axis=0)).mean(axis=1)
        gx = np.abs(np.diff(g, axis=1)).mean(axis=0)
        ys = np.arange(len(gy), dtype=np.float32)
        xs = np.arange(len(gx), dtype=np.float32)
        cy = float((ys * gy).sum() / (gy.sum() + 1e-6))
        cx = float((xs * gx).sum() / (gx.sum() + 1e-6))
        cents.append((cy, cx))
    jumps = [
        ((cents[i][0] - cents[i - 1][0]) ** 2 + (cents[i][1] - cents[i - 1][1]) ** 2) ** 0.5
        for i in range(1, len(cents))
    ]
    # Smooth camera → low variance of jumps; shake → high variance
    jvar = float(np.var(jumps)) if jumps else 0.0
    shake = float(np.clip(1.0 - jvar / 50.0, 0.0, 1.0))
    # Static death: almost zero motion entire clip
    jmean = float(np.mean(jumps)) if jumps else 0.0
    static_pen = 0.0 if jmean > 0.4 else 0.25
    camera = float(np.clip(shake - static_pen, 0.0, 1.0))
    if shake < 0.5:
        notes.append(f"camera_shake={shake:.2f}")
    if static_pen > 0:
        notes.append("near_static_camera")
    return camera, shake, notes


def score_video_artifacts(frame_paths: list[Path] | list[str]) -> ArtifactReport:
    paths = [Path(p) for p in frame_paths]
    app, flicker, color, n1 = score_appearance_artifacts(paths)
    mot, deform, appear_d, n2 = score_motion_artifacts(paths)
    cam, shake, n3 = score_camera_artifacts(paths)
    overall = float(0.35 * app + 0.45 * mot + 0.20 * cam)
    notes = n1 + n2 + n3
    return ArtifactReport(
        appearance=app,
        motion=mot,
        camera=cam,
        overall=overall,
        flicker=flicker,
        color_drift=color,
        deformation=deform,
        appear_disappear=appear_d,
        shake=shake,
        notes=notes,
    )
