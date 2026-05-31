"""Batched classifier-free guidance (one model forward for cond + uncond)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from utils.generation.guidance_session import GuidanceSession
from utils.generation.guidance_stack import SpectralGuidanceEMA, combine_guided_prediction


def merge_cfg_model_kwargs(
    model_kwargs_cond: Dict[str, Any],
    model_kwargs_uncond: Dict[str, Any],
) -> Dict[str, Any]:
    """Stack cond/uncond tensor kwargs along batch dim (2×B)."""
    merged: Dict[str, Any] = {}
    for k, v_cond in model_kwargs_cond.items():
        v_uncond = model_kwargs_uncond.get(k, v_cond)
        if isinstance(v_cond, torch.Tensor) and isinstance(v_uncond, torch.Tensor):
            merged[k] = torch.cat([v_cond, v_uncond], dim=0)
        else:
            merged[k] = v_cond
    return merged


def combine_cfg_outputs(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    x: torch.Tensor,
    *,
    cfg_scale: float,
    cfg_rescale: float = 0.0,
    zeresfdg_strength: float = 0.0,
    fdg_strength: float = 0.0,
    fdg_cutoff_frac: float = 0.15,
    apg_parallel_eta: float = -1.0,
    rcfgpp_tangent: float = 0.0,
    cfg_zero_star: bool = False,
    cfg_pp_lambda: float = 0.0,
    tcfg_damping: float = 0.0,
    cfg_skip_early_frac: float = 0.0,
    cfg_skip_late_frac: float = 0.0,
    sample_step: int = 0,
    total_steps: int = 1,
    cfg_zero_init_frac: float = 0.04,
    spectral_ema: Optional[SpectralGuidanceEMA] = None,
    guidance_session: Optional[GuidanceSession] = None,
) -> torch.Tensor:
    """Apply guidance stack to cond/uncond predictions."""
    return combine_guided_prediction(
        out_cond,
        out_uncond,
        x,
        cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale,
        zeresfdg_strength=zeresfdg_strength,
        fdg_strength=fdg_strength,
        fdg_cutoff_frac=fdg_cutoff_frac,
        apg_parallel_eta=apg_parallel_eta,
        rcfgpp_tangent=rcfgpp_tangent,
        cfg_zero_star=cfg_zero_star,
        cfg_pp_lambda=cfg_pp_lambda,
        tcfg_damping=tcfg_damping,
        cfg_skip_early_frac=cfg_skip_early_frac,
        cfg_skip_late_frac=cfg_skip_late_frac,
        sample_step=sample_step,
        total_steps=total_steps,
        cfg_zero_init_frac=cfg_zero_init_frac,
        spectral_ema=spectral_ema,
        guidance_session=guidance_session,
    )


def batched_cfg_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    *,
    model_kwargs_cond: Dict[str, Any],
    model_kwargs_uncond: Dict[str, Any],
    cfg_scale: float,
    cfg_rescale: float = 0.0,
    zeresfdg_strength: float = 0.0,
    fdg_strength: float = 0.0,
    fdg_cutoff_frac: float = 0.15,
    apg_parallel_eta: float = -1.0,
    block_cache: Any = None,
    rcfgpp_tangent: float = 0.0,
    cfg_zero_star: bool = False,
    cfg_pp_lambda: float = 0.0,
    tcfg_damping: float = 0.0,
    cfg_skip_early_frac: float = 0.0,
    cfg_skip_late_frac: float = 0.0,
    sample_step: int = 0,
    total_steps: int = 1,
    cfg_zero_init_frac: float = 0.04,
    spectral_ema: Optional[SpectralGuidanceEMA] = None,
    guidance_session: Optional[GuidanceSession] = None,
    slg_scale: float = 0.0,
    slg_skip_blocks: Tuple[int, ...] = (),
    slg_active: bool = True,
    guidance_probe: Any = None,
    dbc_separate_cfg: bool = True,
) -> torch.Tensor:
    """Single ``model`` forward for CFG (same math as two sequential forwards)."""
    x2 = torch.cat([x, x], dim=0)
    t2 = torch.cat([t_batch, t_batch], dim=0)
    kwargs = merge_cfg_model_kwargs(model_kwargs_cond, model_kwargs_uncond)
    model_kw = dict(kwargs)
    if block_cache is not None:
        model_kw["dbc_separate_cfg"] = bool(dbc_separate_cfg)
        out = model(x2, t2, block_cache=block_cache, **model_kw)
    else:
        out = model(x2, t2, **model_kw)
    if out.shape[1] > x.shape[1]:
        out = out[:, : x.shape[1]]
    out_cond, out_uncond = out.chunk(2, dim=0)
    if guidance_probe is not None:
        guidance_probe.note(out_cond, out_uncond, step=int(sample_step))
    if float(slg_scale) > 0.0 and slg_skip_blocks and bool(slg_active):
        from utils.generation.slg_guidance import slg_combine

        skip_kw = dict(model_kwargs_cond)
        if block_cache is not None:
            skip_kw["dbc_separate_cfg"] = bool(dbc_separate_cfg)
        out_skip = model(x, t_batch, skip_blocks=slg_skip_blocks, block_cache=block_cache, **skip_kw)
        if out_skip.shape[1] > x.shape[1]:
            out_skip = out_skip[:, : x.shape[1]]
        return slg_combine(
            out_cond,
            out_uncond,
            out_skip,
            cfg_scale=float(cfg_scale),
            slg_scale=float(slg_scale),
            cfg_rescale=float(cfg_rescale),
        )
    return combine_cfg_outputs(
        out_cond,
        out_uncond,
        x,
        cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale,
        zeresfdg_strength=zeresfdg_strength,
        fdg_strength=fdg_strength,
        fdg_cutoff_frac=fdg_cutoff_frac,
        apg_parallel_eta=apg_parallel_eta,
        rcfgpp_tangent=rcfgpp_tangent,
        cfg_zero_star=cfg_zero_star,
        cfg_pp_lambda=cfg_pp_lambda,
        tcfg_damping=tcfg_damping,
        cfg_skip_early_frac=cfg_skip_early_frac,
        cfg_skip_late_frac=cfg_skip_late_frac,
        sample_step=sample_step,
        total_steps=total_steps,
        cfg_zero_init_frac=cfg_zero_init_frac,
        spectral_ema=spectral_ema,
        guidance_session=guidance_session,
    )


__all__ = [
    "GuidanceSession",
    "SpectralGuidanceEMA",
    "batched_cfg_forward",
    "combine_cfg_outputs",
    "combine_guided_prediction",
    "merge_cfg_model_kwargs",
]
