"""Pose / skeleton control image generation for sample.py ControlNet."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["pose_from_rig_boxes", "write_pose_control_image", "pose_control_args"]


def pose_from_rig_boxes(
    width: int,
    height: int,
    parts: Sequence[Tuple[str, Tuple[float, float, float, float]]],
) -> np.ndarray:
    """Draw stick-figure style pose map from normalized rig boxes."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    centers: dict[str, tuple[int, int]] = {}
    for name, box in parts:
        x0, y0, x1, y1 = box
        cx = int((x0 + x1) * 0.5 * width)
        cy = int((y0 + y1) * 0.5 * height)
        centers[name] = (cx, cy)
        ix0, iy0 = int(x0 * width), int(y0 * height)
        ix1, iy1 = int(x1 * width), int(y1 * height)
        canvas[iy0:iy1, ix0:ix1] = (40, 40, 40)
    try:
        import cv2

        joints = [
            ("head", "torso"),
            ("torso", "left_arm"),
            ("torso", "right_arm"),
            ("torso", "legs"),
        ]
        for a, b in joints:
            if a in centers and b in centers:
                cv2.line(canvas, centers[a], centers[b], (255, 255, 255), 2, cv2.LINE_AA)
        for c in centers.values():
            cv2.circle(canvas, c, 4, (255, 255, 255), -1, cv2.LINE_AA)
    except ImportError:
        pass
    return canvas


def write_pose_control_image(
    parts: Sequence[Tuple[str, Tuple[float, float, float, float]]],
    out_path: str | Path,
    *,
    width: int = 512,
    height: int = 512,
) -> Path:
    from .video_io import save_frame_rgb

    rgb = pose_from_rig_boxes(width, height, parts)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    save_frame_rgb(op, rgb)
    return op


def pose_control_args(control_image_path: str | Path) -> List[str]:
    p = str(control_image_path)
    return ["--control-image", p, "--control-type", "pose", "--control-scale", "0.65"]
