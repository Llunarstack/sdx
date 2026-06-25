"""Per-rig-box regional motion: limbs move independently from reference flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .mask_propagate import rasterize_box_mask
from .motion import compute_flow_magnitude
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["load_rig_boxes", "apply_regional_motion", "feather_mask"]


def load_rig_boxes(rig_path: str | Path) -> List[Tuple[str, Tuple[float, float, float, float], bool]]:
    data = json.loads(Path(rig_path).read_text(encoding="utf-8"))
    out: List[Tuple[str, Tuple[float, float, float, float], bool]] = []
    for r in data.get("regions", []):
        if "box" not in r:
            continue
        box = tuple(float(x) for x in r["box"][:4])
        lock = str(r.get("reference_mode", "")).lower() == "identity" or bool(r.get("lock"))
        out.append((str(r.get("name", "part")), box, lock))
    return out


def feather_mask(mask: np.ndarray, radius: int = 6) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    try:
        import cv2

        k = max(3, radius * 2 + 1)
        return cv2.GaussianBlur(m, (k, k), 0)
    except ImportError:
        return m


def _crop_box(rgb: np.ndarray, box: Tuple[float, float, float, float]) -> tuple[np.ndarray, int, int, int, int]:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    ix0 = int(max(0, min(w - 1, x0 * w)))
    iy0 = int(max(0, min(h - 1, y0 * h)))
    ix1 = int(max(ix0 + 1, min(w, x1 * w)))
    iy1 = int(max(iy0 + 1, min(h, y1 * h)))
    return rgb[iy0:iy1, ix0:ix1], ix0, iy0, ix1, iy1


def _paste_box(base: np.ndarray, patch: np.ndarray, ix0: int, iy0: int, ix1: int, iy1: int, alpha: np.ndarray) -> None:
    ph, pw = patch.shape[:2]
    bh, bw = iy1 - iy0, ix1 - ix0
    if ph != bh or pw != bw:
        try:
            import cv2

            patch = cv2.resize(patch, (bw, bh), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            return
    a = alpha[:bh, :bw]
    if a.ndim == 2:
        a = a[..., None]
    region = base[iy0:iy1, ix0:ix1].astype(np.float32)
    blended = region * (1.0 - a) + patch.astype(np.float32) * a
    base[iy0:iy1, ix0:ix1] = np.clip(blended, 0, 255).astype(np.uint8)


def apply_regional_motion(
    edited_anchor_path: str | Path,
    source_frame_paths: Sequence[Path],
    out_dir: str | Path,
    rig_path: str | Path,
    *,
    target_count: int,
    locked_only: bool = False,
) -> List[Path]:
    """
    Warp rig regions independently using source clip flow, composite onto anchor.

    Unlocked regions follow full-frame motion; locked regions get independent warp.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    boxes = load_rig_boxes(rig_path)
    if not boxes or not source_frame_paths:
        from .motion_transfer import motion_template_sequence

        return motion_template_sequence(edited_anchor_path, source_frame_paths, out, target_count=target_count)

    anchor = read_frame_rgb(edited_anchor_path).astype(np.float32)
    h, w = anchor.shape[:2]
    part_states: dict[str, np.ndarray] = {}
    part_masks: dict[str, np.ndarray] = {}
    for name, box, lock in boxes:
        if locked_only and not lock:
            continue
        crop, ix0, iy0, ix1, iy1 = _crop_box(anchor.astype(np.uint8), box)
        part_states[name] = crop.astype(np.float32)
        m = rasterize_box_mask(w, h, box)
        part_masks[name] = feather_mask(m, radius=8)

    acc: List[np.ndarray] = []
    n = min(len(source_frame_paths), max(target_count, 2))
    curr_full = anchor.copy()
    acc.append(np.clip(curr_full, 0, 255).astype(np.uint8))

    for i in range(1, n):
        a = read_frame_rgb(source_frame_paths[i - 1])
        b = read_frame_rgb(source_frame_paths[i])
        try:
            flow, _ = compute_flow_magnitude(a, b)
        except Exception:
            flow = None
        frame = curr_full.copy()
        for name, box, lock in boxes:
            if locked_only and not lock:
                continue
            if name not in part_states:
                continue
            crop = part_states[name]
            ch, cw = crop.shape[:2]
            if flow is not None:
                # Regional mean flow inside box
                m = part_masks[name] > 0.05
                if m.any():
                    fx = float(np.mean(flow[..., 0][m]))
                    fy = float(np.mean(flow[..., 1][m]))
                else:
                    fx, fy = 0.0, 0.0
                try:
                    import cv2

                    mtx = np.float32([[1, 0, fx], [0, 1, fy]])
                    warped = cv2.warpAffine(crop.astype(np.uint8), mtx, (cw, ch), borderMode=cv2.BORDER_REPLICATE)
                    part_states[name] = warped.astype(np.float32)
                except ImportError:
                    pass
            _, ix0, iy0, ix1, iy1 = _crop_box(frame.astype(np.uint8), box)
            _paste_box(frame, part_states[name].astype(np.uint8), ix0, iy0, ix1, iy1, part_masks[name])
        curr_full = frame.astype(np.float32)
        acc.append(np.clip(curr_full, 0, 255).astype(np.uint8))

    from .motion_transfer import retime_frame_list

    acc = retime_frame_list(acc, target_count)
    paths: List[Path] = []
    for j, rgb in enumerate(acc):
        fp = out / f"frame_{j + 1:06d}.png"
        save_frame_rgb(fp, rgb)
        paths.append(fp)
    return paths
