"""
Rasterize per-box sketches (vector strokes or image files) for regional box prompting.

Sketches are drawn inside each region's bounding box; paired with the region ``prompt``
they implement Ideogram-style *draw + describe* per zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore

__all__ = [
    "SketchStroke",
    "parse_strokes",
    "spec_has_sketches",
    "rasterize_region_sketch",
    "build_region_sketch_masks",
    "build_composite_sketch_pil",
    "build_composite_sketch_tensor",
    "sketch_augmented_prompt",
    "apply_sketch_to_region_mask",
]


@dataclass(frozen=True, slots=True)
class SketchStroke:
    """One polyline inside a region box (coordinates relative to the box, 0–1)."""

    points: Tuple[Tuple[float, float], ...]
    width: float = 0.025  # fraction of box min side


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def parse_strokes(raw: Any) -> List[SketchStroke]:
    if not isinstance(raw, list):
        return []
    out: List[SketchStroke] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        pts_raw = item.get("points", item.get("path", []))
        if not isinstance(pts_raw, list) or len(pts_raw) < 2:
            continue
        pts: List[Tuple[float, float]] = []
        for p in pts_raw:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((_clamp01(float(p[0])), _clamp01(float(p[1]))))
        if len(pts) < 2:
            continue
        w = float(item.get("width", item.get("stroke_width", 0.025)) or 0.025)
        out.append(SketchStroke(points=tuple(pts), width=max(0.002, min(0.2, w))))
    return out


def spec_has_sketches(spec: Any) -> bool:
    for r in spec.regions:
        if r.sketch_path or r.strokes:
            return True
    return False


def _resolve_sketch_path(region: Any, source_dir: Optional[Path]) -> Optional[Path]:
    raw = (region.sketch_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_file() and source_dir is not None:
        p = source_dir / raw
    return p if p.is_file() else None


def _box_pixel_bounds(region: Any, height: int, width: int) -> Tuple[int, int, int, int]:
    y0 = int(region.y1 * height)
    y1 = max(y0 + 1, int(region.y2 * height))
    x0 = int(region.x1 * width)
    x1 = max(x0 + 1, int(region.x2 * width))
    return x0, y0, x1, y1


def _draw_strokes_on_image(
    img: "Image.Image",
    region: Any,
    strokes: Sequence[SketchStroke],
    *,
    color: int = 255,
) -> None:
    if ImageDraw is None:
        raise RuntimeError("Pillow is required for box sketch rasterization")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = _box_pixel_bounds(region, img.height, img.width)
    bw, bh = x1 - x0, y1 - y0
    for stroke in strokes:
        px_pts: List[Tuple[float, float]] = []
        for u, v in stroke.points:
            px_pts.append((x0 + u * bw, y0 + v * bh))
        line_w = max(1.0, stroke.width * min(bw, bh))
        draw.line(px_pts, fill=color, width=int(round(line_w)), joint="curve")


def _paste_sketch_image(
    canvas: "Image.Image",
    region: Any,
    sketch_path: Path,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for box sketch rasterization")
    sketch = Image.open(sketch_path).convert("L")
    x0, y0, x1, y1 = _box_pixel_bounds(region, canvas.height, canvas.width)
    bw, bh = x1 - x0, y1 - y0
    sketch = sketch.resize((bw, bh), Image.Resampling.LANCZOS)
    canvas.paste(sketch, (x0, y0))


def rasterize_region_sketch(
    region: Any,
    height: int,
    width: int,
    *,
    source_dir: Optional[Path] = None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Rasterize one region's sketch to ``(1, 1, H, W)`` in ``[0, 1]``.

    White-on-black; undrawn areas inside the box are dim (0.15) so the whole box
    still receives regional CFG, with stronger weight on drawn lines.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for box sketch rasterization")

    canvas = Image.new("L", (width, height), color=0)
    x0, y0, x1, y1 = _box_pixel_bounds(region, height, width)
    # faint fill so the box zone is active even without strokes
    box_layer = Image.new("L", (x1 - x0, y1 - y0), color=int(0.15 * 255))
    canvas.paste(box_layer, (x0, y0))

    sp = _resolve_sketch_path(region, source_dir)
    if sp is not None:
        _paste_sketch_image(canvas, region, sp)
    if region.strokes:
        _draw_strokes_on_image(canvas, region, region.strokes, color=255)

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return t.clamp(0.0, 1.0)


def build_region_sketch_masks(
    spec: Any,
    latent_h: int,
    latent_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Stack per-region sketch masks ``(R, 1, H, W)`` or ``None`` if no sketches."""
    if not spec_has_sketches(spec):
        return None
    source_dir = getattr(spec, "source_dir", None)
    masks: List[torch.Tensor] = []
    for region in spec.regions:
        if region.sketch_path or region.strokes:
            masks.append(
                rasterize_region_sketch(
                    region,
                    latent_h,
                    latent_w,
                    source_dir=source_dir,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            masks.append(torch.zeros(1, 1, latent_h, latent_w, device=device, dtype=dtype))
    return torch.cat(masks, dim=0)


def build_composite_sketch_pil(
    spec: Any,
    pixel_size: int,
    *,
    source_dir: Optional[Path] = None,
) -> "Image.Image":
    """Full-frame RGB scribble control image (white lines on black) at ``pixel_size``."""
    if Image is None:
        raise RuntimeError("Pillow is required for box sketch rasterization")
    src = source_dir or getattr(spec, "source_dir", None)
    acc = np.zeros((pixel_size, pixel_size), dtype=np.float32)
    for region in spec.regions:
        if not (region.sketch_path or region.strokes):
            continue
        layer = rasterize_region_sketch(
            region,
            pixel_size,
            pixel_size,
            source_dir=src,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        acc = np.maximum(acc, layer.squeeze().numpy())
    mono = Image.fromarray((acc * 255.0).astype(np.uint8))
    return Image.merge("RGB", (mono, mono, mono))


def build_composite_sketch_tensor(
    spec: Any,
    pixel_size: int,
    *,
    source_dir: Optional[Path] = None,
    device: torch.device,
) -> torch.Tensor:
    """Control-net style tensor ``(3, H, W)`` normalized to ``[-1, 1]``."""
    pil = build_composite_sketch_pil(spec, pixel_size, source_dir=source_dir)
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=torch.float32)


def sketch_augmented_prompt(region: Any) -> str:
    """Append a light hint when a sketch accompanies the text description."""
    base = (region.prompt or "").strip()
    if not (region.sketch_path or region.strokes):
        return base
    hint = "following the drawn layout in this region"
    if hint.lower() in base.lower():
        return base
    return f"{base}, {hint}" if base else hint


def apply_sketch_to_region_mask(
    region_mask: torch.Tensor,
    sketch_mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """
    Boost regional influence where the user drew.

    ``weight`` in ``[0, 1]`` — higher = stronger adherence on stroke pixels.
    """
    w = float(max(0.0, min(1.0, weight)))
    if w <= 0.0:
        return region_mask
    if sketch_mask.shape[-2:] != region_mask.shape[-2:]:
        sketch_mask = F.interpolate(
            sketch_mask,
            size=region_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    boost = 1.0 + w * sketch_mask.clamp(0.0, 1.0)
    out = region_mask * boost
    return out.clamp(0.0, 1.0)
