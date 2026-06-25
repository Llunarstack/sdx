"""Lock identity regions across frames (face/body from anchor)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["apply_identity_lock_to_sequence", "apply_propagated_identity_lock", "composite_with_mask"]


def composite_with_mask(
    frame_rgb: np.ndarray,
    anchor_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.85,
) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() > 1.0:
        m = m / 255.0
    if m.shape[:2] != frame_rgb.shape[:2]:
        import cv2

        m = cv2.resize(m, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    s = float(max(0.0, min(1.0, strength)))
    m3 = m[..., None]
    out = frame_rgb.astype(np.float32) * (1.0 - m3 * s) + anchor_rgb.astype(np.float32) * (m3 * s)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_identity_lock_to_sequence(
    frame_paths: List[Path],
    anchor_path: str | Path,
    mask_path: Optional[str | Path] = None,
    *,
    box: Optional[tuple[float, float, float, float]] = None,
    strength: float = 0.85,
) -> List[Path]:
    """Re-apply anchor pixels inside mask/box on every frame."""
    from .mask_propagate import load_mask_binary, rasterize_box_mask

    anchor = read_frame_rgb(anchor_path)
    if not frame_paths:
        return []
    h, w = read_frame_rgb(frame_paths[0]).shape[:2]
    if mask_path and Path(mask_path).is_file():
        mask0 = load_mask_binary(mask_path, shape=(h, w))
    elif box is not None:
        mask0 = rasterize_box_mask(w, h, box)
    else:
        # Default: center subject box
        mask0 = rasterize_box_mask(w, h, (0.25, 0.05, 0.75, 0.95))
    for fp in frame_paths:
        rgb = read_frame_rgb(fp)
        locked = composite_with_mask(rgb, anchor, mask0, strength=strength)
        save_frame_rgb(fp, locked)
    return frame_paths


def apply_propagated_identity_lock(
    frame_paths: List[Path],
    anchor_path: str | Path,
    *,
    mask_path: Optional[str | Path] = None,
    box: Optional[tuple[float, float, float, float]] = None,
    strength: float = 0.85,
    work_dir: Optional[Path] = None,
) -> List[Path]:
    """Per-frame mask propagation then anchor composite (tracks moving subjects)."""
    from .mask_propagate import load_mask_binary, propagate_mask_sequence, rasterize_box_mask

    if not frame_paths:
        return []
    anchor = read_frame_rgb(anchor_path)
    h, w = read_frame_rgb(frame_paths[0]).shape[:2]
    if mask_path and Path(mask_path).is_file():
        mask0 = load_mask_binary(mask_path, shape=(h, w))
    elif box is not None:
        mask0 = rasterize_box_mask(w, h, box)
    else:
        mask0 = rasterize_box_mask(w, h, (0.25, 0.05, 0.75, 0.95))

    mask_dir = (work_dir or frame_paths[0].parent) / "_prop_masks"
    if mask_dir.exists():
        import shutil

        shutil.rmtree(mask_dir)
    mask_paths = propagate_mask_sequence(mask0, frame_paths, out_dir=mask_dir)
    for fp, mp in zip(frame_paths, mask_paths[: len(frame_paths)]):
        m = load_mask_binary(mp, shape=(h, w))
        rgb = read_frame_rgb(fp)
        locked = composite_with_mask(rgb, anchor, m, strength=strength)
        save_frame_rgb(fp, locked)
    return frame_paths
