"""Temporal-aware keyframe pick — continuity bias toward the previous keyframe."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb

__all__ = [
    "pixel_continuity_score",
    "pick_keyframe_with_continuity",
    "harmonize_keyframe_pair",
]


def pixel_continuity_score(candidate: np.ndarray, previous: np.ndarray) -> float:
    """
    Higher = more similar to previous keyframe (less flicker risk).

    Uses downsampled RGB L2 in [0, 1].
    """
    a = np.asarray(candidate, dtype=np.float32)
    b = np.asarray(previous, dtype=np.float32)
    if a.shape != b.shape:
        from PIL import Image

        b = np.asarray(
            Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
    # Downsample for speed
    step = max(1, min(a.shape[0], a.shape[1]) // 64)
    a_s = a[::step, ::step, :3] / 255.0
    b_s = b[::step, ::step, :3] / 255.0
    rms = float(np.sqrt(np.mean((a_s - b_s) ** 2)))
    return float(np.clip(1.0 - rms * 2.5, 0.0, 1.0))


def pick_keyframe_with_continuity(
    candidate_paths: list[str | Path],
    *,
    previous_keyframe: str | Path | None = None,
    quality_scores: list[float] | None = None,
    continuity_weight: float = 0.35,
) -> tuple[int, list[float]]:
    """
    Pick best candidate index blending quality scores with continuity to previous KF.

    If ``quality_scores`` is None, uses edge sharpness as a cheap proxy.
    """
    paths = [Path(p) for p in candidate_paths]
    n = len(paths)
    if n == 0:
        return 0, []
    if n == 1:
        return 0, [1.0]

    if quality_scores is None:
        from .quality import score_frame_sharpness

        quality_scores = [score_frame_sharpness(read_frame_rgb(p)) for p in paths]
    q = np.asarray(quality_scores, dtype=np.float64)
    if q.size != n:
        q = np.resize(q, n)
    q_min, q_max = float(q.min()), float(q.max())
    qn = (q - q_min) / (q_max - q_min + 1e-8) if q_max > q_min else np.ones(n)

    w = float(max(0.0, min(1.0, continuity_weight)))
    if previous_keyframe and Path(previous_keyframe).is_file() and w > 1e-6:
        prev = read_frame_rgb(previous_keyframe)
        cont = np.asarray(
            [pixel_continuity_score(read_frame_rgb(p), prev) for p in paths],
            dtype=np.float64,
        )
        combined = (1.0 - w) * qn + w * cont
    else:
        combined = qn

    best = int(np.argmax(combined))
    return best, [float(x) for x in combined]


def harmonize_keyframe_pair(
    current_path: str | Path,
    previous_path: str | Path,
    *,
    strength: float = 0.25,
) -> Path:
    """
    Soft-blend current keyframe toward previous for identity continuity
    (cross-keyframe identity before interpolate).
    """
    cur_p = Path(current_path)
    prev_p = Path(previous_path)
    if not cur_p.is_file() or not prev_p.is_file():
        return cur_p
    s = float(max(0.0, min(0.6, strength)))
    if s < 1e-6:
        return cur_p
    cur = read_frame_rgb(cur_p).astype(np.float32)
    prev = read_frame_rgb(prev_p).astype(np.float32)
    if prev.shape != cur.shape:
        from PIL import Image

        prev = np.asarray(
            Image.fromarray(prev.astype(np.uint8)).resize((cur.shape[1], cur.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        )
    out = cur * (1.0 - s) + prev * s
    from .video_io import save_frame_rgb

    save_frame_rgb(cur_p, np.clip(out, 0, 255).astype(np.uint8))
    return cur_p
