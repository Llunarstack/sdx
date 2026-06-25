"""First–last-frame video (FLF2V) interpolation between anchor endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .depth_interpolate import depth_aware_blend
from .interpolate import interpolate_sequence
from .motion import compute_flow_magnitude, warp_frame_with_flow
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["interpolate_flf2v_sequence", "prepare_flf2v_keyframes"]


def prepare_flf2v_keyframes(
    start_path: str | Path,
    end_path: str | Path,
    out_dir: str | Path,
    *,
    mid_keyframe_paths: Optional[Sequence[Path]] = None,
) -> List[Path]:
    """Ordered keyframes: start → mids → end."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_p = out / "flf_start.png"
    end_p = out / "flf_end.png"
    save_frame_rgb(start_p, read_frame_rgb(start_path))
    save_frame_rgb(end_p, read_frame_rgb(end_path))
    keys = [start_p]
    if mid_keyframe_paths:
        keys.extend([Path(p) for p in mid_keyframe_paths if Path(p).is_file()])
    keys.append(end_p)
    return keys


def interpolate_flf2v_sequence(
    start_path: str | Path,
    end_path: str | Path,
    target_frame_count: int,
    out_dir: str | Path,
    *,
    mid_keyframe_paths: Optional[Sequence[Path]] = None,
    source_frame_paths: Optional[Sequence[Path]] = None,
    use_depth: bool = True,
    use_flow: bool = True,
) -> List[Path]:
    """
    Constrain segment between ``start_path`` and ``end_path``.

    Optional source frames guide mid-segment motion; endpoints are pinned.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if target_frame_count <= 0:
        return []

    start_rgb = read_frame_rgb(start_path)
    end_rgb = read_frame_rgb(end_path)
    keys = prepare_flf2v_keyframes(start_path, end_path, out / "keys", mid_keyframe_paths=mid_keyframe_paths)

    if len(keys) == 2 and source_frame_paths and len(source_frame_paths) >= 3 and use_flow:
        n = target_frame_count
        paths: List[Path] = []
        for i in range(n):
            t = i / max(1, n - 1)
            if i == 0:
                rgb = start_rgb
            elif i == n - 1:
                rgb = end_rgb
            else:
                src_i = min(len(source_frame_paths) - 1, int(round(t * (len(source_frame_paths) - 1))))
                src_rgb = read_frame_rgb(source_frame_paths[src_i])
                try:
                    flow_s, _ = compute_flow_magnitude(start_rgb, src_rgb)
                    warped = warp_frame_with_flow(start_rgb, flow_s, alpha=t)
                    if use_depth:
                        rgb = depth_aware_blend(warped, end_rgb, t)
                    else:
                        rgb = depth_aware_blend(warped, src_rgb, 0.5)
                except Exception:
                    rgb = (
                        depth_aware_blend(start_rgb, end_rgb, t)
                        if use_depth
                        else ((1 - t) * start_rgb.astype(np.float32) + t * end_rgb.astype(np.float32)).astype(np.uint8)
                    )
            fp = out / f"frame_{i + 1:06d}.png"
            save_frame_rgb(fp, rgb)
            paths.append(fp)
        return paths

    from .depth_interpolate import interpolate_sequence_depth

    if use_depth and len(keys) >= 2:
        return interpolate_sequence_depth(keys, target_frame_count, out, use_flow=use_flow)
    return interpolate_sequence(keys, target_frame_count, out, use_flow=use_flow)
