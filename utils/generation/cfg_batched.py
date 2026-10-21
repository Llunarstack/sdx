"""Batched classifier-free guidance: cond, uncond and PAG branches in one model forward."""

from __future__ import annotations

from typing import Any

import torch

from utils.generation.guidance_stack import GuidanceConfig, GuidanceState, combine_guided_prediction


def merge_cfg_model_kwargs(*parts: dict[str, Any]) -> dict[str, Any]:
    """Concatenate per-branch tensor kwargs along the batch dim; non-tensors come from the first part."""
    first = parts[0]
    merged: dict[str, Any] = {}
    for k, v0 in first.items():
        vals = [p.get(k, v0) for p in parts]
        merged[k] = torch.cat(vals, dim=0) if all(isinstance(v, torch.Tensor) for v in vals) else v0
    return merged


def batched_cfg_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    *,
    model_kwargs_cond: dict[str, Any],
    model_kwargs_uncond: dict[str, Any] | None,
    cfg_scale: float,
    guidance: GuidanceConfig | None = None,
    state: GuidanceState | None = None,
    x0_affine: tuple[Any, Any] | None = None,
    sigma: float | None = None,
    step: int = 0,
    total_steps: int = 1,
    flow: bool = False,
    block_cache: Any = None,
    guidance_probe: Any = None,
    dbc_separate_cfg: bool = True,
    amp: bool = False,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Guided prediction from one batched forward (same math as separate per-branch forwards).

    Skip-layer guidance needs its own forward because it drops whole blocks. Outside the guidance
    interval the uncond branch is skipped, except with a block cache, whose cached residuals need a
    fixed batch layout. ``amp`` runs the forwards under CUDA autocast (``autocast_dtype``, default
    bf16 when supported, else fp16) and returns ``x.dtype``; ignored off-GPU.
    """
    g = guidance or GuidanceConfig()
    if g.zero_init_active(step, total_steps, flow=flow):
        return torch.zeros_like(x)
    amp_dtype = None
    if amp and x.is_cuda:
        amp_dtype = autocast_dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    def _forward(xb: torch.Tensor, tb: torch.Tensor, **kw: Any) -> torch.Tensor:
        if amp_dtype is None:
            out = model(xb, tb, **kw)
        else:
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(xb, tb, **kw).to(x.dtype)
        return out[:, : x.shape[1]]

    in_interval = g.cfg_active(sigma)
    run_uncond = float(cfg_scale) != 1.0 and bool(model_kwargs_uncond) and (in_interval or block_cache is not None)
    pag_in_batch = g.pag and block_cache is None
    parts = [model_kwargs_cond]
    if run_uncond:
        parts.append(model_kwargs_uncond)
    if pag_in_batch:
        parts.append(model_kwargs_cond)
    n = len(parts)
    kw = merge_cfg_model_kwargs(*parts)
    if pag_in_batch:
        kw.update(pag_blocks=g.pag_blocks, pag_from=(n - 1) * x.shape[0])
    if block_cache is not None:
        kw.update(block_cache=block_cache, dbc_separate_cfg=bool(dbc_separate_cfg))
    out = _forward(torch.cat([x] * n), torch.cat([t_batch] * n), **kw)
    chunks = out.chunk(n, dim=0)
    out_cond = chunks[0]
    out_uncond = chunks[1] if run_uncond else None
    if guidance_probe is not None and out_uncond is not None:
        guidance_probe.note(out_cond, out_uncond, step=int(step))
    if not in_interval:
        out_uncond = None

    perturbed: list[tuple[float, torch.Tensor]] = []
    if g.pag:
        out_pag = (
            chunks[-1]
            if pag_in_batch
            else _forward(x, t_batch, pag_blocks=g.pag_blocks, pag_from=0, **model_kwargs_cond)
        )
        perturbed.append((g.pag_scale, out_pag))
    if g.slg_active(step, total_steps):
        perturbed.append((g.slg_scale, _forward(x, t_batch, skip_blocks=g.slg_blocks, **model_kwargs_cond)))
    return combine_guided_prediction(
        out_cond,
        out_uncond,
        x,
        cfg_scale=float(cfg_scale),
        config=g,
        x0_affine=x0_affine,
        perturbed=perturbed,
        state=state,
    )


__all__ = ["batched_cfg_forward", "merge_cfg_model_kwargs"]
