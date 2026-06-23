"""
Ideogram-style **box + prompt** regional generation at inference.

Each region is a normalized bounding box ``[x1, y1, x2, y2]`` in ``[0, 1]`` with its own
prompt. Optionally include a **sketch** per box (vector ``strokes`` or ``sketch`` image path)
to draw the layout and describe it in text — draw + describe per region.

During denoising, classifier-free guidance is applied per region and blended in
latent space (Attention-Couple / Regional-Prompter style) — no DiT architecture change.

Use with ``sample.py --box-layout path.json``. See ``examples/box_layout.example.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .regional_box_sketch import SketchStroke, parse_strokes

__all__ = [
    "BoxRegion",
    "BoxLayoutSpec",
    "RegionalCFGPlan",
    "RegionalInjectConfig",
    "load_box_layout_file",
    "parse_box_layout",
    "build_latent_region_masks",
    "layout_text_from_regions",
    "encode_regional_plan",
    "regional_cfg_forward",
    "expand_model_kwargs_batch",
]


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
    strokes: Tuple[SketchStroke, ...] = ()
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
    regions: List[BoxRegion] = field(default_factory=list)
    feather_px: int = 8
    overlap_mode: str = "priority"  # priority | divide
    source_dir: Optional[Path] = None
    inject: RegionalInjectConfig = field(default_factory=RegionalInjectConfig)


@dataclass(slots=True)
class RegionalCFGPlan:
    """Runtime plan passed into ``sample_loop``."""

    region_masks: torch.Tensor  # (R, 1, H, W)
    bg_mask: torch.Tensor  # (1, 1, H, W)
    region_cond_embs: torch.Tensor  # (R, L, D)
    region_names: Tuple[str, ...] = ()
    region_sketches: Optional[torch.Tensor] = None  # (R, 1, H, W)
    region_sketch_weights: Tuple[float, ...] = ()
    inject: RegionalInjectConfig = field(default_factory=RegionalInjectConfig)
    fusion_weights: Tuple[float, ...] = ()  # per-step LAMIC fusion (optional)


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _parse_box(raw: Any) -> Tuple[float, float, float, float]:
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
    sketch_path = str(
        d.get("sketch", d.get("sketch_path", d.get("drawing", d.get("draw", "")))) or ""
    ).strip()
    strokes = tuple(parse_strokes(d.get("strokes", d.get("paths", []))))
    sketch_weight = float(d.get("sketch_weight", d.get("draw_weight", 0.7)) or 0.7)
    reference_path = str(
        d.get("reference", d.get("reference_image", d.get("ref", ""))) or ""
    ).strip()
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


def parse_box_layout(data: Mapping[str, Any]) -> BoxLayoutSpec:
    if not isinstance(data, Mapping):
        raise TypeError("box layout must be a JSON object")
    regions_raw = data.get("regions", data.get("boxes", []))
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
    return BoxLayoutSpec(
        global_prompt=str(data.get("global_prompt", data.get("prompt", "")) or "").strip(),
        global_negative=str(data.get("global_negative", data.get("negative", "")) or "").strip(),
        regions=regions,
        feather_px=max(0, int(data.get("feather", data.get("feather_px", 8)) or 0)),
        overlap_mode=str(data.get("overlap_mode", "priority") or "priority").strip().lower(),
        source_dir=None,
        inject=inject,
    )


def load_box_layout_file(path: Union[str, Path]) -> BoxLayoutSpec:
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
    pixel_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-region masks and background mask at latent resolution.

    Returns ``(region_masks, bg_mask)`` with shapes ``(R, 1, H, W)`` and ``(1, 1, H, W)``.
    """
    px = int(pixel_size or max(latent_h, latent_w) * 8)
    feather_latent = max(0, int(round(spec.feather_px * latent_h / max(px, 1))))

    ordered = sorted(spec.regions, key=lambda r: r.priority, reverse=True)
    masks: List[torch.Tensor] = []
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
    model_kwargs: Dict[str, Any],
    batch: int,
    *,
    repeat_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Expand tensor kwargs from batch B to ``batch`` (typically R×B)."""
    out: Dict[str, Any] = {}
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
    from .regional_box_sketch import build_region_sketch_masks, sketch_augmented_prompt

    try:
        from frontier.layout.coordinate_bind import bind_coordinates_to_prompt
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
    region_sketches = build_region_sketch_masks(
        spec, latent_h, latent_w, device=device, dtype=torch.float32
    )
    sketch_weights = tuple(float(r.sketch_weight) for r in spec.regions)

    fusion_weights: Tuple[float, ...] = ()
    if spec.inject.lamic_isolation:
        try:
            from frontier.layout.lamic_schedule import RegionFusionSchedule, fusion_weight_at_step

            sched = RegionFusionSchedule()
            n_steps = max(4, int(pixel_size // 32))  # proxy; caller may override via plan
            fusion_weights = tuple(
                fusion_weight_at_step(i, n_steps, sched) for i in range(n_steps)
            )
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


def regional_cfg_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    *,
    model_kwargs_cond: Dict[str, Any],
    model_kwargs_uncond: Dict[str, Any],
    plan: RegionalCFGPlan,
    cfg_scale: float,
    cfg_rescale: float = 0.0,
    block_cache: Any = None,
    sample_step: int = 0,
    total_steps: int = 1,
    **guidance_kw: Any,
) -> torch.Tensor:
    """
    Spatial blend of global CFG + per-region CFG predictions.

    Two model forwards: one uncond, one batched (global + all regions).
    """
    from utils.generation.cfg_batched import combine_cfg_outputs
    from .regional_box_sketch import apply_sketch_to_region_mask

    B, C, H, W = x.shape
    R = int(plan.region_cond_embs.shape[0])
    device = x.device

    def _model_out(x_in: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
        kw = dict(kwargs)
        if block_cache is not None:
            out = model(x_in, t_batch, block_cache=block_cache, **kw)
        else:
            out = model(x_in, t_batch, **kw)
        if out.shape[1] > x_in.shape[1]:
            out = out[:, : x_in.shape[1]]
        return out

    out_uncond = _model_out(x, model_kwargs_uncond)

    global_emb = model_kwargs_cond.get("encoder_hidden_states")
    if global_emb is None:
        raise ValueError("regional_cfg_forward requires encoder_hidden_states in model_kwargs_cond")
    stacked_emb = torch.cat([global_emb[:1], plan.region_cond_embs], dim=0)
    n_batch = R + 1
    x_n = x[:1].expand(n_batch, -1, -1, -1)
    t_n = t_batch[:1].expand(n_batch)
    mk_n = expand_model_kwargs_batch(model_kwargs_cond, n_batch)
    mk_n["encoder_hidden_states"] = stacked_emb

    outs = _model_out(x_n, mk_n)
    out_global = outs[0:1].expand(B, -1, -1, -1)
    out_regions = outs[1:]

    guided_global = combine_cfg_outputs(
        out_global,
        out_uncond,
        x,
        cfg_scale=float(cfg_scale),
        cfg_rescale=float(cfg_rescale),
        sample_step=int(sample_step),
        total_steps=int(total_steps),
        **guidance_kw,
    )

    inj = plan.inject
    if inj.mask_inject_steps > 0 and int(sample_step) >= int(inj.mask_inject_steps):
        return guided_global

    masks = plan.region_masks.to(device=device, dtype=x.dtype)
    if masks.shape[-2:] != (H, W):
        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
    bg = plan.bg_mask.to(device=device, dtype=x.dtype)
    if bg.shape[-2:] != (H, W):
        bg = F.interpolate(bg, size=(H, W), mode="bilinear", align_corners=False)

    result = guided_global * bg
    for i in range(R):
        out_r = out_regions[i : i + 1].expand(B, -1, -1, -1)
        guided_r = combine_cfg_outputs(
            out_r,
            out_uncond,
            x,
            cfg_scale=float(cfg_scale),
            cfg_rescale=float(cfg_rescale),
            sample_step=int(sample_step),
            total_steps=int(total_steps),
            **guidance_kw,
        )
        m = masks[i : i + 1]
        if plan.region_sketches is not None and i < plan.region_sketches.shape[0]:
            sw = (
                plan.region_sketch_weights[i]
                if i < len(plan.region_sketch_weights)
                else 0.7
            )
            if plan.region_sketches[i : i + 1].max() > 1e-4:
                m = apply_sketch_to_region_mask(m, plan.region_sketches[i : i + 1], sw)
        result = result + guided_r * m

    br = float(max(0.0, min(1.0, inj.base_ratio)))
    return guided_global * br + result * (1.0 - br)
