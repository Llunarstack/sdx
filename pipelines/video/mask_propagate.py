"""Propagate entity masks across frames using optical flow."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .motion import compute_flow_magnitude
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["load_mask_binary", "propagate_mask_sequence", "rasterize_box_mask"]


def load_mask_binary(path: str | Path, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    from PIL import Image

    m = np.array(Image.open(path).convert("L"))
    if shape is not None and (m.shape[0], m.shape[1]) != shape:
        import cv2

        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.float32)


def rasterize_box_mask(
    width: int,
    height: int,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=np.float32)
    ix0 = int(max(0, min(width - 1, x0 * width)))
    iy0 = int(max(0, min(height - 1, y0 * height)))
    ix1 = int(max(0, min(width, x1 * width)))
    iy1 = int(max(0, min(height, y1 * height)))
    mask[iy0:iy1, ix0:ix1] = 1.0
    return mask


def propagate_mask_sequence(
    mask0: np.ndarray,
    frame_paths: Sequence[Path],
    *,
    out_dir: Path,
) -> List[Path]:
    """Forward-warp initial mask through each frame pair's optical flow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    paths: List[Path] = []
    curr = mask0.astype(np.float32)
    prev_rgb = read_frame_rgb(frame_paths[0])
    h, w = prev_rgb.shape[:2]
    if curr.shape[:2] != (h, w):
        curr = cv2.resize(curr, (w, h), interpolation=cv2.INTER_NEAREST)
    p0 = out_dir / "mask_000001.png"
    save_frame_rgb(p0, (curr * 255).astype(np.uint8)[..., None].repeat(3, axis=2))
    paths.append(p0)
    for i in range(1, len(frame_paths)):
        a = read_frame_rgb(frame_paths[i - 1])
        b = read_frame_rgb(frame_paths[i])
        flow, _ = compute_flow_magnitude(a, b)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(curr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        curr = np.clip(warped, 0, 1)
        fp = out_dir / f"mask_{i + 1:06d}.png"
        save_frame_rgb(fp, (curr * 255).astype(np.uint8)[..., None].repeat(3, axis=2))
        paths.append(fp)
    return paths
