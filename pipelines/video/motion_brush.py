"""Motion brush — restrict optical flow to painted regions (Runway Motion Brush style)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .mask_propagate import load_mask_binary, rasterize_box_mask

__all__ = [
    "MotionBrushSpec",
    "apply_brush_to_flow",
    "load_motion_brush",
    "parse_motion_brush",
    "warp_with_motion_brush",
]


@dataclass(slots=True)
class MotionBrushSpec:
    mask_path: str = ""
    box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    mode: str = "motion_only"  # motion_only | freeze_brush | background_only
    feather: int = 8


def parse_motion_brush(raw: Any) -> Optional[MotionBrushSpec]:
    if not raw:
        return None
    if isinstance(raw, str):
        return MotionBrushSpec(mask_path=raw)
    if not isinstance(raw, Mapping):
        return None
    box_raw = raw.get("box") or raw.get("region")
    box = (0.0, 0.0, 0.0, 0.0)
    if isinstance(box_raw, (list, tuple)) and len(box_raw) >= 4:
        box = tuple(float(x) for x in box_raw[:4])
    return MotionBrushSpec(
        mask_path=str(raw.get("mask") or raw.get("mask_path") or ""),
        box=box,
        mode=str(raw.get("mode") or "motion_only").lower(),
        feather=int(raw.get("feather") or 8),
    )


def load_motion_brush(
    spec: MotionBrushSpec,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Return float mask 0–1."""
    if spec.mask_path and Path(spec.mask_path).is_file():
        if spec.mask_path.endswith(".json"):
            data = json.loads(Path(spec.mask_path).read_text(encoding="utf-8"))
            inner = parse_motion_brush(data)
            if inner:
                return load_motion_brush(inner, width=width, height=height)
        m = load_mask_binary(spec.mask_path, shape=(height, width))
    elif spec.box != (0.0, 0.0, 0.0, 0.0):
        m = rasterize_box_mask(width, height, spec.box)
    else:
        m = np.zeros((height, width), dtype=np.float32)
    if spec.feather > 0:
        try:
            import cv2

            k = max(3, spec.feather * 2 + 1)
            m = cv2.GaussianBlur(m, (k, k), 0)
        except ImportError:
            pass
    return np.clip(m, 0, 1).astype(np.float32)


def apply_brush_to_flow(
    flow: np.ndarray,
    brush: np.ndarray,
    *,
    mode: str = "motion_only",
) -> np.ndarray:
    """Zero or attenuate flow outside/inside brush depending on mode."""
    m = brush.astype(np.float32)
    m3 = m[..., None] if m.ndim == 2 else m
    out = flow.astype(np.float32).copy()
    if mode == "background_only":
        out = out * (1.0 - m3)
    elif mode == "freeze_brush":
        out = out * m3
    else:
        out = out * m3
    return out


def warp_with_motion_brush(
    source_rgb: np.ndarray,
    flow: np.ndarray,
    brush: np.ndarray,
    *,
    mode: str = "motion_only",
    alpha: float = 1.0,
) -> np.ndarray:
    from .motion import warp_frame_with_flow

    masked_flow = apply_brush_to_flow(flow, brush, mode=mode)
    warped = warp_frame_with_flow(source_rgb, masked_flow, alpha=alpha).astype(np.float32)
    if mode == "motion_only":
        m = brush[..., None] if brush.ndim == 2 else brush
        base = source_rgb.astype(np.float32)
        return np.clip(base * (1.0 - m) + warped * m, 0, 255).astype(np.uint8)
    return warped.astype(np.uint8)
