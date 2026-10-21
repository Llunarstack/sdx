"""Composable classifier-free guidance stack.

Every enabled technique runs, in this order, instead of a first-match switch:

1. Unconditional-branch fixes in the model's output space, where their papers define them:
   TCFG (Kwon et al., 2025) and the CFG-Zero* optimized scale (Fan et al., 2025).
2. Both branches mapped to denoised space with the sampler's affine map ``x0 = a*x + b*out``.
3. APG (Sadat et al., ICLR 2025): reverse momentum, per-sample norm threshold, then the
   parallel/orthogonal split against the conditional x0, ``D_c + (w-1)*(d_orth + eta*d_par)``.
4. FDG (Sadat et al., 2025): a two-band Laplacian split of the guidance term with a lower
   low-frequency weight and a higher high-frequency weight; with APG on, it projects per band.
5. Perturbation terms ``s*(D_c - D_perturbed)``: PAG (Ahn et al., 2024) and skip-layer guidance.
6. CFG rescale (Lin et al., WACV 2024): per-sample std match to the conditional x0, blended by phi.

The guidance interval (Kynkäänniemi et al., NeurIPS 2024) and CFG-Zero* zero-init decide which
forwards run at all, so ``cfg_batched.batched_cfg_forward`` applies them before calling the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class GuidanceConfig:
    """Per-run guidance settings. Defaults are plain CFG."""

    rescale: float = 0.0
    apg_eta: float = -1.0
    apg_momentum: float = 0.0
    apg_norm_threshold: float = 0.0
    fdg_low: float = 1.0
    fdg_high: float = 1.0
    cfg_zero_star: bool = False
    cfg_zero_init_frac: float = 0.04
    tcfg_damping: float = 0.0
    interval: tuple[float, float] | None = None
    pag_scale: float = 0.0
    pag_blocks: tuple[int, ...] = ()
    slg_scale: float = 0.0
    slg_blocks: tuple[int, ...] = ()
    slg_window: tuple[float, float] = (0.01, 0.30)

    @classmethod
    def build(
        cls,
        *,
        depth: int = 28,
        interval: str | Sequence[float] | None = None,
        pag_blocks: str | Sequence[int] = "",
        slg_blocks: str | Sequence[int] = "",
        **kw: Any,
    ) -> GuidanceConfig:
        """Config from CLI-style values (``"lo,hi"`` interval, ``"auto"``/``"7,8"`` block lists)."""
        return cls(
            interval=parse_sigma_interval(interval),
            pag_blocks=parse_block_indices(pag_blocks, depth=depth, auto=(max(0, int(depth) // 2),)),
            slg_blocks=parse_block_indices(slg_blocks, depth=depth, auto=_slg_auto_blocks(depth)),
            **kw,
        )

    @property
    def apg(self) -> bool:
        return self.apg_eta >= 0.0

    @property
    def fdg(self) -> bool:
        return self.fdg_low != 1.0 or self.fdg_high != 1.0

    @property
    def pag(self) -> bool:
        return self.pag_scale != 0.0 and bool(self.pag_blocks)

    def cfg_active(self, sigma: float | None) -> bool:
        """Guidance interval: CFG only for noise levels ``sigma`` in ``(lo, hi]``."""
        if self.interval is None or sigma is None:
            return True
        lo, hi = self.interval
        return lo < float(sigma) <= hi

    def slg_active(self, step: int, total_steps: int) -> bool:
        if self.slg_scale == 0.0 or not self.slg_blocks:
            return False
        p = 1.0 if total_steps <= 1 else step / (total_steps - 1)
        return self.slg_window[0] <= p <= self.slg_window[1]

    def zero_init_active(self, step: int, total_steps: int, *, flow: bool) -> bool:
        """CFG-Zero* zero-init: the first steps predict zero velocity. Valid only for flow matching."""
        if not (self.cfg_zero_star and flow) or self.cfg_zero_init_frac <= 0.0:
            return False
        return step < max(1, round(self.cfg_zero_init_frac * max(1, total_steps)))


@dataclass
class GuidanceState:
    """Cross-step state of one sampling trajectory (APG momentum buffer)."""

    apg_momentum: torch.Tensor | None = None


def parse_sigma_interval(spec: str | Sequence[float] | None) -> tuple[float, float] | None:
    """``"0.28,5.42"`` -> ``(0.28, 5.42)``; empty / ``none`` -> ``None``."""
    if spec is None:
        return None
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "none", "off"):
            return None
        spec = [float(p) for p in s.replace(";", ",").split(",")]
    lo, hi = (float(v) for v in spec)
    if not 0.0 <= lo < hi:
        raise ValueError(f"guidance interval needs 0 <= lo < hi, got ({lo}, {hi})")
    return lo, hi


def _slg_auto_blocks(depth: int) -> tuple[int, ...]:
    """Three blocks around 55% of the depth (SD3.5 Medium skips 7,8,9)."""
    d = max(1, int(depth))
    mid = int(round(0.55 * (d - 1)))
    return tuple(sorted({max(0, mid - 1), mid, min(d - 1, mid + 1)}))


def parse_block_indices(spec: str | Sequence[int], *, depth: int = 28, auto: tuple[int, ...] = ()) -> tuple[int, ...]:
    """``"7,8,9"`` / ``"auto"`` / ``""`` -> sorted in-range block indices."""
    if not isinstance(spec, str):
        idx = [int(v) for v in spec]
    else:
        s = spec.strip().lower()
        if s in ("", "none"):
            return ()
        if s == "auto":
            return tuple(auto)
        idx = [int(p) for p in s.replace(";", ",").split(",") if p.strip().isdigit()]
    return tuple(sorted({i for i in idx if 0 <= i < int(depth)}))


def project(v: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample ``(parallel, orthogonal)`` parts of ``v`` w.r.t. ``ref`` (APG, computed in float64)."""
    dims = tuple(range(1, v.ndim))
    v64 = v.double()
    unit = F.normalize(ref.double(), dim=dims)
    par = (v64 * unit).sum(dim=dims, keepdim=True) * unit
    return par.to(v.dtype), (v64 - par).to(v.dtype)


def rescale_guided(guided: torch.Tensor, cond: torch.Tensor, phi: float) -> torch.Tensor:
    """CFG rescale (Lin et al., 2024): ``phi * guided*std(cond)/std(guided) + (1-phi) * guided``, per sample."""
    dims = tuple(range(1, guided.ndim))
    ratio = cond.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True).clamp_min(1e-8)
    return guided * (float(phi) * ratio + (1.0 - float(phi)))


def cfg_zero_star_scale(out_cond: torch.Tensor, out_uncond: torch.Tensor) -> torch.Tensor:
    """CFG-Zero* optimized scale ``s* = <c, u> / ||u||^2`` per sample (Fan et al., 2025)."""
    dims = tuple(range(1, out_cond.ndim))
    dot = (out_cond * out_uncond).sum(dim=dims, keepdim=True)
    return dot / ((out_uncond * out_uncond).sum(dim=dims, keepdim=True) + 1e-8)


def tcfg_uncond(out_cond: torch.Tensor, out_uncond: torch.Tensor, *, damping: float = 1.0) -> torch.Tensor:
    """TCFG (Kwon et al., 2025): drop the unconditional prediction's component along the second
    right-singular vector of ``[cond; uncond]``. ``damping < 1`` removes only that fraction."""
    m = torch.stack([out_cond, out_uncond], dim=1).flatten(2).float()
    _, _, vh = torch.linalg.svd(m, full_matrices=False)
    coef = out_uncond.flatten(1).float().unsqueeze(1) @ vh.transpose(1, 2)
    coef[..., 1] *= 1.0 - float(damping)
    return (coef @ vh).reshape(out_uncond.shape).to(out_uncond.dtype)


def _binomial_blur(x: torch.Tensor) -> torch.Tensor:
    k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype) / 16.0
    k = (k1[:, None] * k1[None, :]).expand(x.shape[1], 1, 5, 5)
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), k, groups=x.shape[1])


def laplacian_split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-level Laplacian pyramid (Burt & Adelson) over the last two dims: ``(low, high)``, ``low + high == x``."""
    if x.ndim < 4:
        raise ValueError(f"frequency split needs spatial (B, C, ..., H, W) tensors, got shape {tuple(x.shape)}")
    flat = x.reshape(x.shape[0], -1, *x.shape[-2:])
    h, w = flat.shape[-2:]
    down = F.interpolate(_binomial_blur(flat), size=((h + 1) // 2, (w + 1) // 2), mode="bilinear", align_corners=False)
    low = _binomial_blur(F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)).reshape(x.shape)
    return low, x - low


def _x0_coeffs(x0_affine: tuple[Any, Any] | None, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x0_affine is None:
        return x.new_zeros(()), x.new_ones(())
    a, b = (torch.as_tensor(v, device=x.device, dtype=x.dtype) for v in x0_affine)
    # b -> 0 (flow at s=0) makes x0 independent of the output; guide the raw output instead.
    tiny = b.abs() < 1e-6
    return torch.where(tiny, torch.zeros_like(a), a), torch.where(tiny, torch.ones_like(b), b)


def _apg_accumulate(delta: torch.Tensor, g: GuidanceConfig, state: GuidanceState | None) -> torch.Tensor:
    if g.apg_momentum != 0.0 and state is not None:
        prev = state.apg_momentum
        if prev is not None and prev.shape == delta.shape:
            delta = delta + g.apg_momentum * prev
        state.apg_momentum = delta.detach()
    if g.apg_norm_threshold > 0.0:
        norm = delta.flatten(1).norm(dim=1).view(-1, *([1] * (delta.ndim - 1)))
        delta = delta * (g.apg_norm_threshold / norm.clamp_min(1e-12)).clamp(max=1.0)
    return delta


def combine_guided_prediction(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor | None,
    x: torch.Tensor,
    *,
    cfg_scale: float,
    config: GuidanceConfig | None = None,
    x0_affine: tuple[Any, Any] | None = None,
    perturbed: Sequence[tuple[float, torch.Tensor]] = (),
    state: GuidanceState | None = None,
) -> torch.Tensor:
    """Guided model output from the conditional, unconditional and perturbed-conditional predictions.

    ``out_uncond=None`` drops the CFG term (outside the guidance interval). ``x0_affine=(a, b)`` maps a
    model output to its denoised estimate ``a*x + b*out``; APG and rescale run in that space.
    ``perturbed`` holds ``(scale, out_perturbed)`` pairs (PAG, SLG). ``state`` carries APG momentum.
    """
    g = config or GuidanceConfig()
    c = x.shape[1]
    w = float(cfg_scale)
    with torch.autocast(device_type=x.device.type, enabled=False):
        out_cond = out_cond[:, :c]
        if out_uncond is not None:
            out_uncond = out_uncond[:, :c]
            if g.tcfg_damping > 0.0:
                out_uncond = tcfg_uncond(out_cond, out_uncond, damping=g.tcfg_damping)
            if g.cfg_zero_star:
                out_uncond = out_uncond * cfg_zero_star_scale(out_cond, out_uncond)
        a, b = _x0_coeffs(x0_affine if (g.apg or g.rescale > 0.0) else None, x)
        d_cond = a * x + b * out_cond
        guided = d_cond
        if out_uncond is not None and w != 1.0:
            delta = d_cond - (a * x + b * out_uncond)
            if g.apg:
                delta = _apg_accumulate(delta, g, state)
            if g.fdg:
                refs = laplacian_split(d_cond) if g.apg else (d_cond, d_cond)
                bands = zip(laplacian_split(delta), refs, (g.fdg_low, g.fdg_high))
            else:
                bands = ((delta, d_cond, 1.0),)
            for d_k, ref_k, r_k in bands:
                if g.apg:
                    par, orth = project(d_k, ref_k)
                    d_k = orth + g.apg_eta * par
                guided = guided + (w - 1.0) * float(r_k) * d_k
        for scale, out_p in perturbed:
            guided = guided + float(scale) * b * (out_cond - out_p[:, :c])
        if g.rescale > 0.0:
            guided = rescale_guided(guided, d_cond, g.rescale)
        return (guided - a * x) / b


__all__ = [
    "GuidanceConfig",
    "GuidanceState",
    "cfg_zero_star_scale",
    "combine_guided_prediction",
    "laplacian_split",
    "parse_block_indices",
    "parse_sigma_interval",
    "project",
    "rescale_guided",
    "tcfg_uncond",
]
