"""Ideogram-style box + prompt regional generation: layout parsing, per-box sketches, masks and the regional CFG forward."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore


@dataclass(frozen=True, slots=True)
class SketchStroke:
    """One polyline inside a region box (coordinates relative to the box, 0–1)."""

    points: tuple[tuple[float, float], ...]
    width: float = 0.025  # fraction of box min side


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def parse_strokes(raw: Any) -> list[SketchStroke]:
    if not isinstance(raw, list):
        return []
    out: list[SketchStroke] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        pts_raw = item.get("points", item.get("path", []))
        if not isinstance(pts_raw, list) or len(pts_raw) < 2:
            continue
        pts: list[tuple[float, float]] = []
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


def _resolve_sketch_path(region: Any, source_dir: Path | None) -> Path | None:
    raw = (region.sketch_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_file() and source_dir is not None:
        p = source_dir / raw
    return p if p.is_file() else None


def _box_pixel_bounds(region: Any, height: int, width: int) -> tuple[int, int, int, int]:
    y0 = int(region.y1 * height)
    y1 = max(y0 + 1, int(region.y2 * height))
    x0 = int(region.x1 * width)
    x1 = max(x0 + 1, int(region.x2 * width))
    return x0, y0, x1, y1


def _draw_strokes_on_image(
    img: Image.Image,
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
        px_pts: list[tuple[float, float]] = []
        for u, v in stroke.points:
            px_pts.append((x0 + u * bw, y0 + v * bh))
        line_w = max(1.0, stroke.width * min(bw, bh))
        draw.line(px_pts, fill=color, width=int(round(line_w)), joint="curve")


def _paste_sketch_image(
    canvas: Image.Image,
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
    source_dir: Path | None = None,
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
) -> torch.Tensor | None:
    """Stack per-region sketch masks ``(R, 1, H, W)`` or ``None`` if no sketches."""
    if not spec_has_sketches(spec):
        return None
    source_dir = getattr(spec, "source_dir", None)
    masks: list[torch.Tensor] = []
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
    source_dir: Path | None = None,
) -> Image.Image:
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
    source_dir: Path | None = None,
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


if TYPE_CHECKING:
    from utils.generation.guidance import GuidanceConfig


@dataclass(slots=True)
class BoxRegion:
    """One spatial box with its own prompt and optional sketch."""

    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    prompt: str
    negative: str = ""
    priority: int = 5
    sketch_path: str = ""
    strokes: tuple[SketchStroke, ...] = ()
    sketch_weight: float = 0.7
    reference_path: str = ""
    reference_weight: float = 0.8
    reference_mode: str = "identity"


@dataclass(slots=True)
class RegionalInjectConfig:
    """
    Regional-Prompting-FLUX-style inject schedule.

    ``mask_inject_steps``: apply per-region blend for the first N sampler steps.
    ``base_ratio``: weight kept on global CFG vs regional (lower = stronger regions).
    """

    mask_inject_steps: int = 10
    base_ratio: float = 0.15
    use_coordinate_tokens: bool = False
    lamic_isolation: bool = False


@dataclass(slots=True)
class BoxLayoutSpec:
    """Parsed box layout file."""

    global_prompt: str = ""
    global_negative: str = ""
    regions: list[BoxRegion] = field(default_factory=list)
    feather_px: int = 8
    overlap_mode: str = "priority"  # priority | divide
    source_dir: Path | None = None
    inject: RegionalInjectConfig = field(default_factory=RegionalInjectConfig)


@dataclass(slots=True)
class RegionalCFGPlan:
    """Runtime plan passed into ``sample_loop``."""

    region_masks: torch.Tensor  # (R, 1, H, W)
    bg_mask: torch.Tensor  # (1, 1, H, W)
    region_cond_embs: torch.Tensor  # (R, L, D)
    region_names: tuple[str, ...] = ()
    region_sketches: torch.Tensor | None = None  # (R, 1, H, W)
    region_sketch_weights: tuple[float, ...] = ()
    inject: RegionalInjectConfig = field(default_factory=RegionalInjectConfig)
    fusion_weights: tuple[float, ...] = ()  # per-step LAMIC fusion (optional)
    cache: dict = field(default_factory=dict, repr=False, compare=False)  # step-invariant tensors


def _parse_box(raw: Any) -> tuple[float, float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    elif isinstance(raw, Mapping):
        x1 = float(raw.get("x_min", raw.get("x1", 0.0)))
        y1 = float(raw.get("y_min", raw.get("y1", 0.0)))
        x2 = float(raw.get("x_max", raw.get("x2", 1.0)))
        y2 = float(raw.get("y_max", raw.get("y2", 1.0)))
    else:
        raise ValueError(f"invalid box: {raw!r}")
    x1, x2 = sorted((_clamp01(x1), _clamp01(x2)))
    y1, y2 = sorted((_clamp01(y1), _clamp01(y2)))
    if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
        raise ValueError(f"degenerate box [{x1},{y1},{x2},{y2}]")
    return x1, y1, x2, y2


def _parse_region_dict(d: Mapping[str, Any], idx: int) -> BoxRegion:
    name = str(d.get("name", d.get("id", f"region_{idx}")) or f"region_{idx}").strip()
    box_raw = d.get("box", d.get("bbox", d))
    if isinstance(box_raw, Mapping) and "box" not in box_raw and "bbox" not in box_raw:
        if not any(k in box_raw for k in ("x_min", "x1", "x_max", "x2")):
            box_raw = d.get("box", d.get("bbox", [0, 0, 1, 1]))
    x1, y1, x2, y2 = _parse_box(box_raw if box_raw is not d else d)
    prompt = str(d.get("prompt", d.get("text", d.get("caption", d.get("description", "")))) or "").strip()
    negative = str(d.get("negative", d.get("negative_prompt", "")) or "").strip()
    priority = int(d.get("priority", 5))
    sketch_path = str(d.get("sketch", d.get("sketch_path", d.get("drawing", d.get("draw", "")))) or "").strip()
    strokes = tuple(parse_strokes(d.get("strokes", d.get("paths", []))))
    sketch_weight = float(d.get("sketch_weight", d.get("draw_weight", 0.7)) or 0.7)
    reference_path = str(d.get("reference", d.get("reference_image", d.get("ref", ""))) or "").strip()
    reference_weight = float(d.get("reference_weight", d.get("ref_weight", 0.8)) or 0.8)
    reference_mode = str(d.get("reference_mode", "identity") or "identity").strip()
    return BoxRegion(
        name=name,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        prompt=prompt,
        negative=negative,
        priority=priority,
        sketch_path=sketch_path,
        strokes=strokes,
        sketch_weight=max(0.0, min(1.0, sketch_weight)),
        reference_path=reference_path,
        reference_weight=max(0.0, min(1.0, reference_weight)),
        reference_mode=reference_mode,
    )


def _ideogram_regions(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Ideogram ``compositional_deconstruction.elements`` → region dicts.

    Ideogram bboxes are ``[y_min, x_min, y_max, x_max]`` on a 0–1000 grid.
    """
    comp = data.get("compositional_deconstruction")
    elements = comp.get("elements") if isinstance(comp, Mapping) else None
    out: list[dict[str, Any]] = []
    for i, el in enumerate(elements if isinstance(elements, list) else []):
        bbox = el.get("bbox") if isinstance(el, Mapping) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        y1, x1, y2, x2 = (float(v) / 1000.0 for v in bbox[:4])
        is_text = str(el.get("type", "")).lower() == "text"
        text = str(el.get("text", "") or "").strip()
        desc = str(el.get("desc", el.get("description", "")) or "").strip()
        parts = [f'lettering reads exactly: "{text}"' if text else "", desc]
        out.append(
            {
                "name": el.get("name") or f"{'text' if is_text else 'obj'}_{i + 1}",
                "box": [x1, y1, x2, y2],
                "prompt": ", ".join(p for p in parts if p),
                "priority": el.get("priority", 12 if is_text else 6),
            }
        )
    return out


def parse_box_layout(data: Mapping[str, Any]) -> BoxLayoutSpec:
    if not isinstance(data, Mapping):
        raise TypeError("box layout must be a JSON object")
    regions_raw = data.get("regions", data.get("boxes")) or _ideogram_regions(data)
    if not isinstance(regions_raw, list) or not regions_raw:
        raise ValueError("box layout requires a non-empty 'regions' list")
    regions = [_parse_region_dict(r, i) for i, r in enumerate(regions_raw) if isinstance(r, Mapping)]
    if not regions:
        raise ValueError("no valid regions in box layout")
    for r in regions:
        if not r.prompt:
            raise ValueError(f"region {r.name!r} is missing 'prompt' (describe what you drew)")
    inject = RegionalInjectConfig(
        mask_inject_steps=max(0, int(data.get("mask_inject_steps", 10) or 10)),
        base_ratio=float(max(0.0, min(1.0, data.get("base_ratio", 0.15) or 0.15))),
        use_coordinate_tokens=bool(data.get("coordinate_tokens", data.get("use_loc_tokens", False))),
        lamic_isolation=bool(data.get("lamic_isolation", False)),
    )
    global_prompt = str(data.get("global_prompt", data.get("prompt", data.get("description", ""))) or "").strip()
    style = data.get("style")
    palette = (style.get("color_palette") if isinstance(style, Mapping) else None) or data.get("color_palette")
    if isinstance(palette, list) and palette:
        pin = "color palette locked to " + ", ".join(str(c) for c in palette)
        global_prompt = f"{global_prompt}, {pin}" if global_prompt else pin
    return BoxLayoutSpec(
        global_prompt=global_prompt,
        global_negative=str(data.get("global_negative", data.get("negative", "")) or "").strip(),
        regions=regions,
        feather_px=max(0, int(data.get("feather", data.get("feather_px", 8)) or 0)),
        overlap_mode=str(data.get("overlap_mode", "priority") or "priority").strip().lower(),
        source_dir=None,
        inject=inject,
    )


def load_box_layout_file(path: str | Path) -> BoxLayoutSpec:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"box layout not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(raw, dict):
        raise ValueError("box layout JSON must be an object at the root")
    spec = parse_box_layout(raw)
    spec.source_dir = p.parent
    return spec


def layout_text_from_regions(spec: BoxLayoutSpec) -> str:
    """Merge regions into ``[layout]`` training-style text (text-only fallback)."""
    parts = [f"{r.name}: {r.prompt}" for r in spec.regions]
    layout = " | ".join(parts)
    gp = (spec.global_prompt or "").strip()
    if gp:
        return f"{gp}. [layout] {layout}"
    return f"[layout] {layout}"


def _box_mask_tensor(
    region: BoxRegion,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y0 = int(region.y1 * height)
    y1 = max(y0 + 1, int(region.y2 * height))
    x0 = int(region.x1 * width)
    x1 = max(x0 + 1, int(region.x2 * width))
    m = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    m[:, :, y0:y1, x0:x1] = 1.0
    return m


def _feather_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    if radius_px <= 0:
        return mask
    k = radius_px * 2 + 1
    pad = radius_px
    blurred = F.avg_pool2d(
        F.pad(mask, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=k,
        stride=1,
    )
    return blurred.clamp(0.0, 1.0)


def build_latent_region_masks(
    spec: BoxLayoutSpec,
    latent_h: int,
    latent_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    pixel_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-region masks and background mask at latent resolution.

    Returns ``(region_masks, bg_mask)`` with shapes ``(R, 1, H, W)`` and ``(1, 1, H, W)``.
    """
    px = int(pixel_size or max(latent_h, latent_w) * 8)
    feather_latent = max(0, int(round(spec.feather_px * latent_h / max(px, 1))))

    ordered = sorted(spec.regions, key=lambda r: r.priority, reverse=True)
    masks: list[torch.Tensor] = []
    occupied = torch.zeros(1, 1, latent_h, latent_w, device=device, dtype=dtype)

    for region in ordered:
        raw = _box_mask_tensor(region, latent_h, latent_w, device=device, dtype=dtype)
        if spec.overlap_mode == "priority":
            raw = raw * (1.0 - occupied)
            occupied = (occupied + raw).clamp(0.0, 1.0)
        raw = _feather_mask(raw, feather_latent)
        masks.append(raw)

    if not masks:
        bg = torch.ones(1, 1, latent_h, latent_w, device=device, dtype=dtype)
        return torch.zeros(0, 1, latent_h, latent_w, device=device, dtype=dtype), bg

    stacked = torch.cat(masks, dim=0)
    if spec.overlap_mode == "divide":
        total = stacked.sum(dim=0, keepdim=True).clamp(min=1e-8)
        stacked = stacked / total

    region_sum = stacked.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
    bg = (1.0 - region_sum).clamp(0.0, 1.0)
    return stacked, bg


def expand_model_kwargs_batch(
    model_kwargs: dict[str, Any],
    batch: int,
    *,
    repeat_index: int | None = None,
) -> dict[str, Any]:
    """Expand tensor kwargs from batch B to ``batch`` (typically R×B)."""
    out: dict[str, Any] = {}
    for k, v in model_kwargs.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if v.shape[0] == batch:
            out[k] = v
        elif v.shape[0] == 1:
            out[k] = v.expand(batch, *v.shape[1:])
        elif repeat_index is not None and v.shape[0] > 1:
            out[k] = v[repeat_index : repeat_index + 1].expand(batch, *v.shape[1:])
        else:
            out[k] = v[:1].expand(batch, *v.shape[1:])
    return out


def encode_regional_plan(
    spec: BoxLayoutSpec,
    *,
    encode_fn,
    device: torch.device,
    latent_h: int,
    latent_w: int,
    pixel_size: int,
    base_negative: str = "",
) -> RegionalCFGPlan:
    """
    Encode per-region prompts and build latent masks.

    ``encode_fn(captions: List[str]) -> Tensor`` must return ``(N, L, D)`` embeddings.
    """

    try:
        from frontier.layout import bind_coordinates_to_prompt
    except ImportError:
        bind_coordinates_to_prompt = None  # type: ignore

    region_masks, bg_mask = build_latent_region_masks(
        spec,
        latent_h,
        latent_w,
        device=device,
        pixel_size=pixel_size,
    )
    prompts = []
    for r in spec.regions:
        p = sketch_augmented_prompt(r)
        if spec.inject.use_coordinate_tokens and bind_coordinates_to_prompt is not None:
            p = bind_coordinates_to_prompt(p, (r.x1, r.y1, r.x2, r.y2))
        prompts.append(p)
    region_neg = [r.negative or base_negative or spec.global_negative for r in spec.regions]
    pos_emb = encode_fn(prompts)
    if pos_emb.shape[0] != len(prompts):
        raise RuntimeError("encode_fn returned unexpected batch size for regional prompts")
    _ = region_neg
    region_sketches = build_region_sketch_masks(spec, latent_h, latent_w, device=device, dtype=torch.float32)
    sketch_weights = tuple(float(r.sketch_weight) for r in spec.regions)

    fusion_weights: tuple[float, ...] = ()
    if spec.inject.lamic_isolation:
        try:
            from frontier.layout import RegionFusionSchedule, fusion_weight_at_step

            sched = RegionFusionSchedule()
            n_steps = max(4, int(pixel_size // 32))  # proxy; caller may override via plan
            fusion_weights = tuple(fusion_weight_at_step(i, n_steps, sched) for i in range(n_steps))
        except ImportError:
            fusion_weights = ()

    return RegionalCFGPlan(
        region_masks=region_masks.to(device=device),
        bg_mask=bg_mask.to(device=device),
        region_cond_embs=pos_emb.to(device=device),
        region_names=tuple(r.name for r in spec.regions),
        region_sketches=region_sketches.to(device=device) if region_sketches is not None else None,
        region_sketch_weights=sketch_weights,
        inject=spec.inject,
        fusion_weights=fusion_weights,
    )


def _blend_masks(
    plan: RegionalCFGPlan, h: int, w: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Background mask and per-region masks (sketch-boosted) at ``h x w``; built once per plan and grid."""
    key = ("masks", h, w, dtype, device)
    if key not in plan.cache:
        masks = plan.region_masks.to(device=device, dtype=dtype)
        if masks.shape[-2:] != (h, w):
            masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        bg = plan.bg_mask.to(device=device, dtype=dtype)
        if bg.shape[-2:] != (h, w):
            bg = F.interpolate(bg, size=(h, w), mode="bilinear", align_corners=False)
        sketches = plan.region_sketches
        per = []
        for i in range(int(plan.region_cond_embs.shape[0])):
            m = masks[i : i + 1]
            if sketches is not None and i < sketches.shape[0] and sketches[i : i + 1].max() > 1e-4:
                sw = plan.region_sketch_weights[i] if i < len(plan.region_sketch_weights) else 0.7
                m = apply_sketch_to_region_mask(m, sketches[i : i + 1], sw)
            per.append(m)
        plan.cache[key] = (bg, per)
    return plan.cache[key]


def regional_cfg_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    *,
    model_kwargs_cond: dict[str, Any],
    model_kwargs_uncond: dict[str, Any],
    plan: RegionalCFGPlan,
    cfg_scale: float,
    block_cache: Any = None,
    sample_step: int = 0,
    guidance: GuidanceConfig | None = None,
    x0_affine: tuple[Any, Any] | None = None,
    sigma: float | None = None,
) -> torch.Tensor:
    """
    Spatial blend of global CFG + per-region CFG predictions through the guidance stack.

    Two model forwards: one uncond, one batched (global + all regions). Regions share the uncond
    branch, so APG momentum (per-trajectory state) is not used here.
    """
    from utils.generation.guidance import GuidanceConfig, combine_guided_prediction

    B, C, H, W = x.shape
    R = int(plan.region_cond_embs.shape[0])
    device = x.device

    def _model_out(
        x_in: torch.Tensor,
        kwargs: dict[str, Any],
        *,
        t_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ts = t_in if t_in is not None else t_batch
        kw = dict(kwargs)
        if block_cache is not None:
            out = model(x_in, ts, block_cache=block_cache, **kw)
        else:
            out = model(x_in, ts, **kw)
        if out.shape[1] > x_in.shape[1]:
            out = out[:, : x_in.shape[1]]
        return out

    g = guidance or GuidanceConfig()
    out_uncond = _model_out(x, model_kwargs_uncond) if g.cfg_active(sigma) else None
    stack_kw = dict(cfg_scale=float(cfg_scale), config=g, x0_affine=x0_affine)

    global_emb = model_kwargs_cond.get("encoder_hidden_states")
    if global_emb is None:
        raise ValueError("regional_cfg_forward requires encoder_hidden_states in model_kwargs_cond")
    hit = plan.cache.get("emb")
    if hit is not None and hit[0] is global_emb:
        stacked_emb = hit[1]
    else:
        stacked_emb = torch.cat([global_emb[:1], plan.region_cond_embs], dim=0)
        plan.cache["emb"] = (global_emb, stacked_emb)
    n_batch = R + 1
    x_n = x[:1].expand(n_batch, -1, -1, -1)
    t_n = t_batch[:1].expand(n_batch)
    mk_n = expand_model_kwargs_batch(model_kwargs_cond, n_batch)
    mk_n["encoder_hidden_states"] = stacked_emb

    outs = _model_out(x_n, mk_n, t_in=t_n)
    out_global = outs[0:1].expand(B, -1, -1, -1)
    out_regions = outs[1:]

    guided_global = combine_guided_prediction(out_global, out_uncond, x, **stack_kw)

    inj = plan.inject
    if inj.mask_inject_steps > 0 and int(sample_step) >= int(inj.mask_inject_steps):
        return guided_global

    bg, masks = _blend_masks(plan, H, W, x.dtype, device)
    result = guided_global * bg
    for i, m in enumerate(masks):
        out_r = out_regions[i : i + 1].expand(B, -1, -1, -1)
        result = result + combine_guided_prediction(out_r, out_uncond, x, **stack_kw) * m

    br = float(max(0.0, min(1.0, inj.base_ratio)))
    return guided_global * br + result * (1.0 - br)
