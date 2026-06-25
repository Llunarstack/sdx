"""Apply reference clip motion to edited anchor frames (motion template path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from .motion import compute_flow_magnitude, warp_frame_with_flow
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["motion_template_sequence", "retime_frame_list"]


def retime_frame_list(frames: List[np.ndarray], target_count: int) -> List[np.ndarray]:
    if not frames or target_count <= 0:
        return []
    if len(frames) == target_count:
        return frames
    idx = np.linspace(0, len(frames) - 1, target_count).astype(int)
    return [frames[i] for i in idx]


def motion_template_sequence(
    anchor_path: str | Path,
    source_frame_paths: Sequence[Path],
    out_dir: str | Path,
    *,
    target_count: int,
    blend_source: float = 0.15,
    motion_brush: Optional[Any] = None,
) -> List[Path]:
    """
    Warp edited anchor forward using optical flow from source clip.

    Keeps motion from reference while preserving anchor appearance.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not source_frame_paths:
        rgb = read_frame_rgb(anchor_path)
        paths = []
        for i in range(target_count):
            fp = out / f"frame_{i + 1:06d}.png"
            save_frame_rgb(fp, rgb)
            paths.append(fp)
        return paths

    brush_arr = None
    brush_mode = "motion_only"
    if motion_brush is not None:
        from .motion_brush import load_motion_brush, parse_motion_brush, warp_with_motion_brush

        spec = motion_brush if hasattr(motion_brush, "mask_path") else parse_motion_brush(motion_brush)
        if spec:
            h0, w0 = read_frame_rgb(anchor_path).shape[:2]
            brush_arr = load_motion_brush(spec, width=w0, height=h0)
            brush_mode = spec.mode

    curr = read_frame_rgb(anchor_path).astype(np.float32)
    acc: List[np.ndarray] = [curr.astype(np.uint8)]
    n = min(len(source_frame_paths), max(target_count, 2))
    for i in range(1, n):
        a = read_frame_rgb(source_frame_paths[i - 1])
        b = read_frame_rgb(source_frame_paths[i])
        try:
            flow, _ = compute_flow_magnitude(a, b)
            if brush_arr is not None:
                from .motion_brush import warp_with_motion_brush

                warped = warp_with_motion_brush(
                    curr.astype(np.uint8), flow, brush_arr, mode=brush_mode, alpha=1.0
                ).astype(np.float32)
            else:
                warped = warp_frame_with_flow(curr.astype(np.uint8), flow, alpha=1.0).astype(np.float32)
            if blend_source > 0:
                src = b.astype(np.float32)
                if src.shape == warped.shape:
                    warped = (1.0 - blend_source) * warped + blend_source * src
            curr = warped
        except Exception:
            curr = read_frame_rgb(source_frame_paths[i]).astype(np.float32)
        acc.append(np.clip(curr, 0, 255).astype(np.uint8))

    acc = retime_frame_list(acc, target_count)
    paths: List[Path] = []
    for i, rgb in enumerate(acc):
        fp = out / f"frame_{i + 1:06d}.png"
        save_frame_rgb(fp, rgb)
        paths.append(fp)
    return paths
