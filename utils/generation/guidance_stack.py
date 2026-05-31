"""
Unified **guidance stack** dispatcher (ZeResFDG, FDG, APG, CFG-Zero*, Rectified-CFG++).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from utils.generation.zeresfdg import SpectralGuidanceEMA

if TYPE_CHECKING:
    from utils.generation.guidance_session import GuidanceSession


def combine_guided_prediction(
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
    guidance_session: Optional["GuidanceSession"] = None,
) -> torch.Tensor:
    """Single entry point for all guidance variants used by ``cfg_batched``."""
    if out_cond.shape != x.shape and out_cond.shape[1] > x.shape[1]:
        out_cond = out_cond[:, : x.shape[1]]
        out_uncond = out_uncond[:, : x.shape[1]]

    progress = float(sample_step) / float(max(int(total_steps) - 1, 1)) if int(total_steps) > 1 else 0.0
    if float(cfg_skip_early_frac) > 0.0 or float(cfg_skip_late_frac) > 0.0:
        from utils.generation.cfg_interval import should_apply_cfg

        if not should_apply_cfg(
            progress,
            skip_early_frac=float(cfg_skip_early_frac),
            skip_late_frac=float(cfg_skip_late_frac),
        ):
            return out_cond

    if bool(cfg_zero_star):
        from utils.generation.cfg_zero_star import cfg_zero_star_combine

        return cfg_zero_star_combine(
            out_cond,
            out_uncond,
            cfg_scale=float(cfg_scale),
            sample_step=int(sample_step),
            total_steps=int(total_steps),
            zero_init_frac=float(cfg_zero_init_frac),
        )

    if float(zeresfdg_strength) > 0.0:
        from utils.generation.zeresfdg import apply_zeresfdg_cfg

        return apply_zeresfdg_cfg(
            out_cond,
            out_uncond,
            cfg_scale=float(cfg_scale),
            cfg_rescale=float(cfg_rescale),
            fdg_cutoff_frac=float(fdg_cutoff_frac),
            spectral_ema=spectral_ema,
            strength=float(zeresfdg_strength),
        )

    if float(cfg_pp_lambda) > 0.0:
        from utils.generation.cfg_pp import cfg_pp_combine

        return cfg_pp_combine(
            out_cond,
            out_uncond,
            cfg_lambda=float(cfg_pp_lambda),
            cfg_rescale=float(cfg_rescale),
        )

    if float(tcfg_damping) > 0.0:
        from utils.generation.tcfg import tcfg_combine

        return tcfg_combine(
            out_cond,
            out_uncond,
            cfg_scale=float(cfg_scale),
            damping=float(tcfg_damping),
            cfg_rescale=float(cfg_rescale),
        )

    if float(fdg_strength) > 0.0:
        from utils.superior.frequency_cfg import apply_fdg_cfg

        return apply_fdg_cfg(
            out_cond,
            out_uncond,
            cfg_scale=float(cfg_scale),
            cfg_rescale=float(cfg_rescale),
            fdg_strength=float(fdg_strength),
            cutoff_frac=float(fdg_cutoff_frac),
        )

    if float(apg_parallel_eta) >= 0.0:
        from utils.generation.apg_guidance import apg_guidance_delta

        mom = None
        beta = 0.0
        if guidance_session is not None and float(guidance_session.apg_momentum_beta) > 0.0:
            mom = guidance_session.prev_apg_delta
            beta = float(guidance_session.apg_momentum_beta)
        delta = out_cond - out_uncond
        apg_delta = apg_guidance_delta(
            delta,
            out_cond,
            parallel_eta=float(apg_parallel_eta),
            cfg_rescale=float(cfg_rescale),
            momentum_delta=mom,
            momentum_beta=beta,
        )
        if guidance_session is not None:
            guidance_session.note_apg_delta(apg_delta)
        return out_uncond + float(cfg_scale) * apg_delta

    if float(rcfgpp_tangent) > 0.0:
        from utils.generation.rectified_cfgpp import rectified_cfgpp_combine

        return rectified_cfgpp_combine(
            out_cond,
            out_uncond,
            cfg_scale=float(cfg_scale),
            cfg_rescale=float(cfg_rescale),
            tangent_norm=float(rcfgpp_tangent),
        )

    delta = out_cond - out_uncond
    if cfg_rescale > 0:
        sig = delta.std() + 1e-8
        delta = delta / max(sig / float(cfg_rescale), 1.0)
    return out_uncond + float(cfg_scale) * delta


__all__ = ["SpectralGuidanceEMA", "combine_guided_prediction"]
