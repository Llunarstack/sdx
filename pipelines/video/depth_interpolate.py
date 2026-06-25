"""Edge-aware depth-proxy interpolation between keyframes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["depth_proxy", "interpolate_sequence_depth"]


def depth_proxy(rgb: np.ndarray) -> np.ndarray:
    """Cheap depth stand-in: inverse luminance + Sobel edge emphasis."""
    gray = rgb[..., :3].astype(np.float32).mean(axis=2) / 255.0
    inv = 1.0 - gray
    try:
        import cv2

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-6)
        depth = np.clip(0.65 * inv + 0.35 * edge, 0, 1)
    except ImportError:
        depth = inv
    return depth.astype(np.float32)


def depth_aware_blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Occlusion-aware blend using depth proxy (closer layer wins at mid-t)."""
    t = float(max(0.0, min(1.0, t)))
    da = depth_proxy(a)
    db = depth_proxy(b)
    # Soft winner-take-most: nearer surface dominates cross-fade
    wa = (1.0 - t) * (1.0 + da * 0.75)
    wb = t * (1.0 + db * 0.75)
    wsum = wa + wb + 1e-6
    wa, wb = wa / wsum, wb / wsum
    out = wa[..., None] * a.astype(np.float32) + wb[..., None] * b.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def interpolate_sequence_depth(
    keyframe_paths: Sequence[str | Path],
    target_frame_count: int,
    out_dir: str | Path,
    *,
    use_flow: bool = True,
) -> List[Path]:
    """Like ``interpolate_sequence`` but mid-frames use depth-aware blending."""
    from .interpolate import interpolate_sequence

    keys = [Path(p) for p in keyframe_paths if Path(p).is_file()]
    if len(keys) <= 1:
        return interpolate_sequence(keyframe_paths, target_frame_count, out_dir, use_flow=use_flow)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments = len(keys) - 1
    per_seg = max(1, target_frame_count // segments)
    all_paths: List[Path] = [keys[0]]
    for s in range(segments):
        a = read_frame_rgb(keys[s])
        b = read_frame_rgb(keys[s + 1])
        seg_dir = out / f"seg_{s:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        mids: List[Path] = []
        count = max(0, per_seg - 2)
        flow = None
        if use_flow:
            try:
                from .motion import compute_flow_magnitude, warp_frame_with_flow

                flow, _ = compute_flow_magnitude(a, b)
            except Exception:
                flow = None
        for i in range(1, count + 1):
            t = i / (count + 1)
            if flow is not None:
                warped = warp_frame_with_flow(a, flow, alpha=t)
                rgb = depth_aware_blend(warped, b, t * 0.45)
            else:
                rgb = depth_aware_blend(a, b, t)
            fp = seg_dir / f"mid_{i:04d}.png"
            save_frame_rgb(fp, rgb)
            mids.append(fp)
        all_paths.extend(mids)
        all_paths.append(keys[s + 1])

    if len(all_paths) > target_frame_count:
        idx = np.linspace(0, len(all_paths) - 1, target_frame_count).astype(int)
        all_paths = [all_paths[i] for i in idx]
    elif len(all_paths) < target_frame_count:
        all_paths.extend([all_paths[-1]] * (target_frame_count - len(all_paths)))

    final: List[Path] = []
    for i, src in enumerate(all_paths[:target_frame_count]):
        fp = out / f"frame_{i + 1:06d}.png"
        save_frame_rgb(fp, read_frame_rgb(src))
        final.append(fp)
    return final
