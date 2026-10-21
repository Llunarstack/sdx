"""Frame interpolation between keyframes (flow-guided + blend fallback).

Also holds what used to be: depth_interpolate, flf2v.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .motion import compute_flow_magnitude, warp_frame_with_flow
from .video_io import read_frame_rgb, save_frame_rgb

# ==================================================================================================
# interpolate
# ==================================================================================================


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
) -> list[Path]:
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
    paths: list[Path] = []
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
) -> list[Path]:
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
    all_paths: list[Path] = []
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
    final: list[Path] = []
    for i, src in enumerate(all_paths):
        fp = out / f"frame_{i + 1:06d}.png"
        if src.resolve() != fp.resolve():
            save_frame_rgb(fp, read_frame_rgb(src))
        else:
            fp = src
        final.append(fp)
    return final


# ==================================================================================================
# depth_interpolate: Edge-aware depth-proxy interpolation between keyframes.
# ==================================================================================================


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
) -> list[Path]:
    """Like ``interpolate_sequence`` but mid-frames use depth-aware blending."""

    keys = [Path(p) for p in keyframe_paths if Path(p).is_file()]
    if len(keys) <= 1:
        return interpolate_sequence(keyframe_paths, target_frame_count, out_dir, use_flow=use_flow)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments = len(keys) - 1
    per_seg = max(1, target_frame_count // segments)
    all_paths: list[Path] = [keys[0]]
    for s in range(segments):
        a = read_frame_rgb(keys[s])
        b = read_frame_rgb(keys[s + 1])
        seg_dir = out / f"seg_{s:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        mids: list[Path] = []
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

    final: list[Path] = []
    for i, src in enumerate(all_paths[:target_frame_count]):
        fp = out / f"frame_{i + 1:06d}.png"
        save_frame_rgb(fp, read_frame_rgb(src))
        final.append(fp)
    return final


# ==================================================================================================
# flf2v: First–last-frame video (FLF2V) interpolation between anchor endpoints.
# ==================================================================================================


def prepare_flf2v_keyframes(
    start_path: str | Path,
    end_path: str | Path,
    out_dir: str | Path,
    *,
    mid_keyframe_paths: Sequence[Path] | None = None,
) -> list[Path]:
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
    mid_keyframe_paths: Sequence[Path] | None = None,
    source_frame_paths: Sequence[Path] | None = None,
    use_depth: bool = True,
    use_flow: bool = True,
) -> list[Path]:
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
        paths: list[Path] = []
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

    if use_depth and len(keys) >= 2:
        return interpolate_sequence_depth(keys, target_frame_count, out, use_flow=use_flow)
    return interpolate_sequence(keys, target_frame_count, out, use_flow=use_flow)


__all__ = [
    "blend_frames",
    "interpolate_sequence",
    "interpolate_between_keyframes",
    "depth_proxy",
    "interpolate_sequence_depth",
    "interpolate_flf2v_sequence",
    "prepare_flf2v_keyframes",
]
