"""Motion analysis: optical flow, camera motion score, motion templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .video_io import read_frame_rgb

__all__ = [
    "MotionProfile",
    "compute_flow_magnitude",
    "estimate_motion_score",
    "extract_motion_profile",
    "warp_frame_with_flow",
]


@dataclass(slots=True)
class MotionProfile:
    mean_flow: float
    max_flow: float
    pan_x: float
    pan_y: float
    zoom_hint: float
    frame_count: int


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python required for motion helpers") from exc


def compute_flow_magnitude(frame_a: np.ndarray, frame_b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Dense optical flow magnitude between consecutive RGB frames."""
    cv2 = _require_cv2()
    g0 = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return flow, float(np.mean(mag))


def estimate_motion_score(frame_paths: List[Path], *, sample_pairs: int = 8) -> float:
    """0–1 motion intensity proxy from sampled frame pairs."""
    if len(frame_paths) < 2:
        return 0.0
    step = max(1, len(frame_paths) // max(1, sample_pairs))
    mags: List[float] = []
    for i in range(0, len(frame_paths) - 1, step):
        a = read_frame_rgb(frame_paths[i])
        b = read_frame_rgb(frame_paths[min(len(frame_paths) - 1, i + step)])
        _, m = compute_flow_magnitude(a, b)
        mags.append(m)
    if not mags:
        return 0.0
    return float(min(1.0, np.mean(mags) / 8.0))


def extract_motion_profile(frame_paths: List[Path], *, max_pairs: int = 12) -> MotionProfile:
    if len(frame_paths) < 2:
        return MotionProfile(0.0, 0.0, 0.0, 0.0, 0.0, len(frame_paths))
    step = max(1, len(frame_paths) // max(1, max_pairs))
    mags: List[float] = []
    pan_x: List[float] = []
    pan_y: List[float] = []
    for i in range(0, len(frame_paths) - 1, step):
        a = read_frame_rgb(frame_paths[i])
        b = read_frame_rgb(frame_paths[min(len(frame_paths) - 1, i + step)])
        flow, m = compute_flow_magnitude(a, b)
        mags.append(m)
        pan_x.append(float(np.mean(flow[..., 0])))
        pan_y.append(float(np.mean(flow[..., 1])))
    mean_m = float(np.mean(mags)) if mags else 0.0
    return MotionProfile(
        mean_flow=mean_m,
        max_flow=float(max(mags) if mags else 0.0),
        pan_x=float(np.mean(pan_x) if pan_x else 0.0),
        pan_y=float(np.mean(pan_y) if pan_y else 0.0),
        zoom_hint=float(min(1.0, abs(np.mean(pan_x or [0])) + abs(np.mean(pan_y or [0])))),
        frame_count=len(frame_paths),
    )


def warp_frame_with_flow(
    source_rgb: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Warp source by flow field (motion-guided propagation)."""
    cv2 = _require_cv2()
    h, w = source_rgb.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0] * alpha).astype(np.float32)
    map_y = (grid_y + flow[..., 1] * alpha).astype(np.float32)
    warped = cv2.remap(
        cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
