"""
Lip-sync driver — correlate mouth-band motion with audio energy (Wan/Hailuo claim).

Neural viseme models need weights; we score + soft-nudge the lower-face band so
silent frames don't flap and loud frames aren't frozen mouths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["LipSyncReport", "score_lip_sync", "apply_lip_sync_nudge"]


@dataclass(slots=True)
class LipSyncReport:
    score: float
    correlation: float = 0.0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _mouth_activity(rgb: np.ndarray) -> float:
    """Edge energy in lower-center face band."""
    h, w = rgb.shape[:2]
    y0, y1 = int(h * 0.55), int(h * 0.78)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    band = rgb[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
    if band.size < 4:
        return 0.0
    gx = np.abs(np.diff(band, axis=1)).mean()
    gy = np.abs(np.diff(band, axis=0)).mean()
    return float(gx + gy)


def score_lip_sync(
    frame_paths: list[Path] | list[str],
    audio_energy: np.ndarray | None = None,
    *,
    wav_path: str | Path | None = None,
    fps: float = 24.0,
) -> LipSyncReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return LipSyncReport(score=1.0)
    acts = np.asarray([_mouth_activity(read_frame_rgb(p)) for p in paths], dtype=np.float32)
    if acts.max() > 1e-6:
        acts = acts / acts.max()
    if audio_energy is None and wav_path:
        from .native_audio_track import energy_envelope

        audio_energy = energy_envelope(wav_path, fps=fps, duration_sec=len(paths) / max(fps, 1e-6))
    if audio_energy is None or len(audio_energy) < 2:
        # No audio: penalize high mouth chatter (talking without sound)
        chatter = float(np.std(acts))
        score = float(np.clip(1.0 - chatter * 2.0, 0.0, 1.0))
        return LipSyncReport(score=score, correlation=0.0, notes=["no_audio_energy"])
    # Resample audio to frame count
    idx = np.linspace(0, len(audio_energy) - 1, len(acts)).astype(int)
    ae = audio_energy[idx]
    if ae.std() < 1e-6 or acts.std() < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(acts, ae)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    score = float(np.clip(0.5 + 0.5 * corr, 0.0, 1.0))
    notes = []
    if corr < 0.15:
        notes.append(f"lip_corr={corr:.2f}")
    return LipSyncReport(score=score, correlation=corr, notes=notes)


def apply_lip_sync_nudge(
    frame_paths: list[Path],
    audio_energy: np.ndarray,
    *,
    strength: float = 0.35,
) -> LipSyncReport:
    """
    Soft temporal blend of mouth band toward neighbors so activity tracks energy:
    high energy → prefer sharper (current) mouth; low energy → blend toward calmer neighbor.
    """
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3 or audio_energy is None or len(audio_energy) < 2:
        return LipSyncReport(score=1.0)
    frames = [read_frame_rgb(p).astype(np.float32) for p in paths]
    idx = np.linspace(0, len(audio_energy) - 1, len(frames)).astype(int)
    ae = audio_energy[idx]
    if ae.max() > 1e-6:
        ae = ae / ae.max()
    a = float(np.clip(strength, 0.0, 1.0))
    repaired = 0
    for i in range(1, len(frames) - 1):
        h, w = frames[i].shape[:2]
        y0, y1 = int(h * 0.55), int(h * 0.78)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        # Low energy → blend toward average of neighbors (closed-ish)
        calm = 0.5 * (frames[i - 1] + frames[i + 1])
        w_calm = (1.0 - float(ae[i])) * a
        if w_calm < 0.05:
            continue
        region = frames[i][y0:y1, x0:x1]
        frames[i][y0:y1, x0:x1] = region * (1.0 - w_calm) + calm[y0:y1, x0:x1] * w_calm
        save_frame_rgb(paths[i], np.clip(frames[i], 0, 255).astype(np.uint8))
        repaired += 1
    after = score_lip_sync(paths, audio_energy=ae)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"lip_nudge={repaired}")
    return after
