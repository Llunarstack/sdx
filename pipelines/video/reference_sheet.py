"""Auto reference sheet from one hero image (front / profile / full-body crops)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["ReferenceSheet", "build_reference_sheet", "write_reference_sheet_manifest"]


@dataclass(slots=True)
class ReferenceSheet:
    source: str
    views: List[str]
    boxes: List[Tuple[str, Tuple[float, float, float, float]]]
    manifest_path: str = ""


def _subject_box(rgb: np.ndarray) -> Tuple[float, float, float, float]:
    from .auto_rig import _center_of_mass_box

    tmp = Path("_tmp_rig_probe.png")
    try:
        save_frame_rgb(tmp, rgb)
        box = _center_of_mass_box(tmp)
        if box:
            return box
    finally:
        tmp.unlink(missing_ok=True)
    return (0.2, 0.05, 0.8, 0.95)


def _crop_norm(rgb: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    ix0 = int(max(0, min(w - 1, x0 * w)))
    iy0 = int(max(0, min(h - 1, y0 * h)))
    ix1 = int(max(ix0 + 1, min(w, x1 * w)))
    iy1 = int(max(iy0 + 1, min(h, y1 * h)))
    crop = rgb[iy0:iy1, ix0:ix1]
    try:
        import cv2

        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        return crop


def build_reference_sheet(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    target_size: Optional[tuple[int, int]] = None,
) -> ReferenceSheet:
    """
    Heuristic multi-view sheet from a single image.

    Produces front (center), profile (horizontal offset), full-body (wider box) crops
    resized back to frame size — enough for multi-ref consistency like Runway/Kling.
    """
    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb = read_frame_rgb(src)
    h, w = rgb.shape[:2]
    if target_size:
        tw, th = target_size
        try:
            import cv2

            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pass

    subj = _subject_box(rgb)
    sx0, sy0, sx1, sy1 = subj
    sw, sh = sx1 - sx0, sy1 - sy0
    cx = (sx0 + sx1) * 0.5

    views_spec = [
        ("front", (sx0 + sw * 0.05, sy0, sx1 - sw * 0.05, sy1)),
        ("profile", (max(0.0, cx - sw * 0.55), sy0, min(1.0, cx + sw * 0.15), sy1)),
        (
            "full_body",
            (
                max(0.0, sx0 - sw * 0.12),
                max(0.0, sy0 - sh * 0.05),
                min(1.0, sx1 + sw * 0.12),
                min(1.0, sy1 + sh * 0.02),
            ),
        ),
    ]
    paths: List[str] = []
    boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for name, box in views_spec:
        view = _crop_norm(rgb, box)
        fp = out / f"{name}.png"
        save_frame_rgb(fp, view)
        paths.append(str(fp))
        boxes.append((name, box))

    manifest = out / "reference_sheet.json"
    write_reference_sheet_manifest(
        ReferenceSheet(source=str(src.resolve()), views=paths, boxes=boxes, manifest_path=str(manifest)),
        manifest,
    )
    return ReferenceSheet(source=str(src.resolve()), views=paths, boxes=boxes, manifest_path=str(manifest))


def write_reference_sheet_manifest(sheet: ReferenceSheet, path: str | Path) -> Path:
    p = Path(path)
    payload = {
        "source": sheet.source,
        "views": [{"name": n, "path": vp, "box": list(b)} for (n, b), vp in zip(sheet.boxes, sheet.views)],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
