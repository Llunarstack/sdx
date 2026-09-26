"""
Motion shutter — cinematic directional blur competitors fake badly.

Kling boasts physically-correct shutter blur; most models either mush or
omit it. We estimate local flow from frame pairs and apply anisotropic blur
along motion for film/realistic grammars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["ShutterReport", "apply_motion_shutter", "score_motion_shutter"]


@dataclass(slots=True)
class ShutterReport:
    score: float = 1.0
    applied: int = 0
    mean_flow: float = 0.0
    notes: list[str] = field(default_factory=list)


def _block_flow(a: np.ndarray, b: np.ndarray, *, block: int = 16) -> tuple[float, float]:
    """Coarse translational flow via SSD on center crop."""
    ha, wa = a.shape[:2]
    y0, x0 = ha // 4, wa // 4
    y1, x1 = 3 * ha // 4, 3 * wa // 4
    pa = a[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
    pb = b[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
    best = 1e18
    bdx = bdy = 0
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            shifted = np.roll(np.roll(pb, dy, axis=0), dx, axis=1)
            err = float(np.mean((pa - shifted) ** 2))
            if err < best:
                best = err
                bdx, bdy = dx, dy
    return float(bdx), float(bdy)


def _directional_blur(rgb: np.ndarray, dx: float, dy: float, *, amount: float) -> np.ndarray:
    """Cheap multi-tap blur along (dx, dy)."""
    mag = (dx * dx + dy * dy) ** 0.5
    if mag < 0.4 or amount < 0.05:
        return rgb
    taps = max(2, min(7, int(round(mag * amount * 1.5))))
    acc = rgb.astype(np.float32) * 0.0
    wsum = 0.0
    for t in range(-taps, taps + 1):
        wt = 1.0 - abs(t) / (taps + 1)
        ox = int(round(dx * t / max(taps, 1) * amount))
        oy = int(round(dy * t / max(taps, 1) * amount))
        shifted = np.roll(np.roll(rgb, oy, axis=0), ox, axis=1).astype(np.float32)
        acc += shifted * wt
        wsum += wt
    return np.clip(acc / max(wsum, 1e-6), 0, 255).astype(np.uint8)


def apply_motion_shutter(
    frame_paths: list[Path],
    *,
    amount: float = 0.45,
    min_flow: float = 0.8,
) -> ShutterReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return ShutterReport()
    applied = 0
    flows: list[float] = []
    prev = read_frame_rgb(paths[0])
    for p in paths[1:]:
        curr = read_frame_rgb(p)
        dx, dy = _block_flow(prev, curr)
        mag = (dx * dx + dy * dy) ** 0.5
        flows.append(mag)
        if mag >= min_flow:
            out = _directional_blur(curr, dx, dy, amount=amount)
            save_frame_rgb(p, out)
            applied += 1
            curr = out
        prev = curr
    mean_f = float(np.mean(flows)) if flows else 0.0
    # Score: applied when motion warrants it
    score = 1.0 if mean_f < min_flow else float(np.clip(applied / max(len(paths) - 1, 1), 0.4, 1.0))
    notes = [f"shutter_applied={applied}"] if applied else ["shutter_skip_low_motion"]
    return ShutterReport(score=score, applied=applied, mean_flow=mean_f, notes=notes)


def score_motion_shutter(frame_paths: list[Path] | list[str]) -> ShutterReport:
    """Soft score: reward moderate directional streaking vs isotropic mush."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return ShutterReport(score=1.0)
    # Proxy: if consecutive frames differ anisotropically, shutter looks "alive"
    a = read_frame_rgb(paths[0]).astype(np.float32)
    b = read_frame_rgb(paths[min(1, len(paths) - 1)]).astype(np.float32)
    d = np.abs(a - b).mean(axis=2)
    gy = np.abs(np.diff(d, axis=0)).mean()
    gx = np.abs(np.diff(d, axis=1)).mean()
    aniso = abs(gx - gy) / (gx + gy + 1e-6)
    score = float(np.clip(0.5 + aniso, 0.0, 1.0))
    return ShutterReport(score=score, notes=[f"aniso={aniso:.2f}"])
