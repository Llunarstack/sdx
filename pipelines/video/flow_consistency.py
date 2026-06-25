"""Forward-backward flow consistency repair for interpolated frames."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .motion import compute_flow_magnitude
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["repair_flow_consistency"]


def _fb_error(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    h, w = flow_fwd.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_fwd[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_fwd[..., 1]).astype(np.float32)
    try:
        import cv2

        bx = cv2.remap(flow_bwd[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        by = cv2.remap(flow_bwd[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        err = np.sqrt((flow_fwd[..., 0] + bx) ** 2 + (flow_fwd[..., 1] + by) ** 2)
        return err.astype(np.float32)
    except ImportError:
        return np.zeros((h, w), dtype=np.float32)


def repair_flow_consistency(frame_paths: List[Path], *, threshold: float = 2.5, blend: float = 0.45) -> List[Path]:
    """Blend inconsistent pixels toward previous frame (reduces ghosting)."""
    if len(frame_paths) < 2:
        return frame_paths
    prev = read_frame_rgb(frame_paths[0])
    for i in range(1, len(frame_paths)):
        curr = read_frame_rgb(frame_paths[i])
        try:
            f_fwd, _ = compute_flow_magnitude(prev, curr)
            f_bwd, _ = compute_flow_magnitude(curr, prev)
            err = _fb_error(f_fwd, f_bwd)
            mask = (err > threshold).astype(np.float32)
            if mask.max() > 0:
                m3 = mask[..., None]
                b = float(max(0.0, min(1.0, blend)))
                fixed = curr.astype(np.float32) * (1.0 - m3 * b) + prev.astype(np.float32) * (m3 * b)
                curr = np.clip(fixed, 0, 255).astype(np.uint8)
                save_frame_rgb(frame_paths[i], curr)
        except Exception:
            pass
        prev = curr
    return frame_paths
