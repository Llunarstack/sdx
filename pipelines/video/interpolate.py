"""Frame interpolation between keyframes (flow-guided + blend fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .motion import compute_flow_magnitude, warp_frame_with_flow
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["blend_frames", "interpolate_sequence", "interpolate_between_keyframes"]


def blend_frames(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(max(0.0, min(1.0, t)))
    return np.clip((1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32), 0, 255).astype(np.uint8)


def interpolate_between_keyframes(
    frame_a_path: str | Path,
    frame_b_path: str | Path,
    out_dir: str | Path,
    *,
    count: int,
    use_flow: bool = True,
    prefix: str = "mid",
) -> List[Path]:
    """Generate ``count`` intermediate frames between A and B."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = read_frame_rgb(frame_a_path)
    b = read_frame_rgb(frame_b_path)
    flow = None
    if use_flow:
        try:
            flow, _ = compute_flow_magnitude(a, b)
        except Exception:
            flow = None
    paths: List[Path] = []
    for i in range(1, count + 1):
        t = i / (count + 1)
        if flow is not None:
            warped = warp_frame_with_flow(a, flow, alpha=t)
            rgb = blend_frames(warped, b, t * 0.35)
        else:
            rgb = blend_frames(a, b, t)
        fp = out / f"{prefix}_{i:04d}.png"
        save_frame_rgb(fp, rgb)
        paths.append(fp)
    return paths


def interpolate_sequence(
    keyframe_paths: Sequence[str | Path],
    target_frame_count: int,
    out_dir: str | Path,
    *,
    use_flow: bool = True,
) -> List[Path]:
    """
    Expand sparse keyframes to ``target_frame_count`` frames via segment interpolation.
    """
    keys = [Path(p) for p in keyframe_paths if Path(p).is_file()]
    if not keys:
        return []
    if len(keys) == 1:
        rgb = read_frame_rgb(keys[0])
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(target_frame_count):
            fp = out / f"frame_{i + 1:06d}.png"
            save_frame_rgb(fp, rgb)
            paths.append(fp)
        return paths
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments = len(keys) - 1
    per_seg = max(1, target_frame_count // segments)
    all_paths: List[Path] = []
    all_paths.append(keys[0])
    for s in range(segments):
        mids = interpolate_between_keyframes(
            keys[s],
            keys[s + 1],
            out / f"seg_{s:02d}",
            count=max(0, per_seg - 2),
            use_flow=use_flow,
            prefix="mid",
        )
        all_paths.extend(mids)
        all_paths.append(keys[s + 1])
    # Retime to exact count
    if len(all_paths) > target_frame_count:
        idx = np.linspace(0, len(all_paths) - 1, target_frame_count).astype(int)
        all_paths = [all_paths[i] for i in idx]
    elif len(all_paths) < target_frame_count:
        last = all_paths[-1]
        while len(all_paths) < target_frame_count:
            all_paths.append(last)
    final: List[Path] = []
    for i, src in enumerate(all_paths):
        fp = out / f"frame_{i + 1:06d}.png"
        if src.resolve() != fp.resolve():
            save_frame_rgb(fp, read_frame_rgb(src))
        else:
            fp = src
        final.append(fp)
    return final
