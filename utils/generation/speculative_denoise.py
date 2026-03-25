"""
**Speculative** denoising helper: cheap draft prediction + optional acceptance (same backbone).

This does **not** load a separate tiny DiT. It uses **two** forwards on the **same** ``model``:
one at full CFG and one at a **scaled-down** guidance (draft). If draft and full predictions are
close in L2, optionally blend toward the draft for a small wall-clock win on some steps.

Intended for experimentation; tune ``draft_cfg_scale`` and ``blend_if_close`` carefully.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch


def cfg_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    *,
    model_kwargs_cond: Dict,
    model_kwargs_uncond: Optional[Dict],
    cfg_scale: float,
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    """One CFG-combined prediction (cond / uncond), channel-trimmed to ``x``."""
    if model_kwargs_uncond is not None and cfg_scale != 1.0 and cfg_scale > 0:
        oc = model(x, t_batch, **model_kwargs_cond)
        ou = model(x, t_batch, **model_kwargs_uncond)
        if oc.shape != x.shape and oc.shape[1] > x.shape[1]:
            oc, ou = oc[:, : x.shape[1]], ou[:, : x.shape[1]]
        delta = oc - ou
        if cfg_rescale > 0:
            sig = delta.std() + 1e-8
            delta = delta / max(sig / cfg_rescale, 1.0)
        return ou + float(cfg_scale) * delta
    o = model(x, t_batch, **model_kwargs_cond)
    if o.shape != x.shape and o.shape[1] > x.shape[1]:
        o = o[:, : x.shape[1]]
    return o


def speculative_cfg_prediction(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    *,
    model_kwargs_cond: Dict,
    model_kwargs_uncond: Optional[Dict],
    cfg_scale: float,
    draft_cfg_scale: float,
    cfg_rescale: float = 0.0,
    close_thresh: float = 0.0,
    blend_on_close: float = 0.35,
) -> torch.Tensor:
    """
    If ``close_thresh > 0`` and ``|pred_full - pred_draft|`` mean is below threshold, return
    ``(1-a)*pred_full + a*pred_draft`` with ``a = blend_on_close``; else return ``pred_full``.

    Always runs **two** model forwards when CFG uses cond+uncond (draft then full).
    """
    pf = cfg_predict(
        model,
        x,
        t_batch,
        model_kwargs_cond=model_kwargs_cond,
        model_kwargs_uncond=model_kwargs_uncond,
        cfg_scale=float(cfg_scale),
        cfg_rescale=cfg_rescale,
    )
    if (
        model_kwargs_uncond is None
        or float(draft_cfg_scale) <= 0
        or abs(float(draft_cfg_scale) - float(cfg_scale)) < 1e-6
    ):
        return pf
    pd = cfg_predict(
        model,
        x,
        t_batch,
        model_kwargs_cond=model_kwargs_cond,
        model_kwargs_uncond=model_kwargs_uncond,
        cfg_scale=float(draft_cfg_scale),
        cfg_rescale=cfg_rescale,
    )
    if float(close_thresh) <= 0.0:
        return pf
    delta = (pf - pd).abs().mean()
    if float(delta.detach().cpu().item()) < float(close_thresh):
        a = float(max(0.0, min(1.0, blend_on_close)))
        return (1.0 - a) * pf + a * pd
    return pf


__all__ = ["cfg_predict", "speculative_cfg_prediction"]
