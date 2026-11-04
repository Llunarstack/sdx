"""Sampling-time guidance: the composable CFG stack, batched CFG forward, CFG-rejection probe, latent filters, frontier kwargs and CLIP alignment."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from utils.modeling.architecture import CFGRejectionTracker, pick_best_candidate_index


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
    """Cross-step state of one sampling trajectory (APG momentum buffer, batched-CFG kwargs cache)."""

    apg_momentum: torch.Tensor | None = None
    kwargs_cache: dict = field(default_factory=dict)


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


def merge_cfg_model_kwargs(*parts: dict[str, Any], cache: dict | None = None) -> dict[str, Any]:
    """Concatenate per-branch tensor kwargs along the batch dim; non-tensors come from the first part.

    ``cache`` (one per trajectory) reuses a concatenation while its inputs are the same tensor objects,
    so the text embeddings are not re-copied every step.
    """
    first = parts[0]
    merged: dict[str, Any] = {}
    for k, v0 in first.items():
        vals = [p.get(k, v0) for p in parts]
        if not all(isinstance(v, torch.Tensor) for v in vals):
            merged[k] = v0
            continue
        hit = cache.get((k, len(vals))) if cache is not None else None
        if hit is not None and all(a is b for a, b in zip(hit[0], vals)):
            merged[k] = hit[1]
            continue
        merged[k] = torch.cat(vals, dim=0)
        if cache is not None:
            cache[(k, len(vals))] = (vals, merged[k])
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
    kw = merge_cfg_model_kwargs(*parts, cache=state.kwargs_cache if state is not None else None)
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


@dataclass
class GuidanceProbe:
    """Accumulate early-step CFG gaps per batch row."""

    tau_steps: int = 4
    trackers: list[CFGRejectionTracker] = field(default_factory=list)

    def ensure_batch(self, batch_size: int) -> None:
        while len(self.trackers) < int(batch_size):
            self.trackers.append(CFGRejectionTracker(tau_steps=int(self.tau_steps)))

    def note(self, out_cond: torch.Tensor, out_uncond: torch.Tensor, *, step: int) -> None:
        if int(step) >= int(self.tau_steps):
            return
        b = int(out_cond.shape[0])
        self.ensure_batch(b)
        for i in range(b):
            self.trackers[i].note(out_cond[i : i + 1], out_uncond[i : i + 1])

    def accumulated_scores(self) -> list[float]:
        return [t.accumulated_early_score() for t in self.trackers]

    def rerank_indices(self) -> list[int]:
        scores = self.accumulated_scores()
        if not scores:
            return []
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        return order

    def best_index(self) -> int:
        scores = self.accumulated_scores()
        return int(pick_best_candidate_index(scores))


def _fft_highpass(x: torch.Tensor, cutoff_frac: float) -> torch.Tensor:
    """High band of ``(B, C, H, W)`` above a radial FFT cutoff (fraction of the max frequency radius)."""
    cf = float(min(0.45, max(0.03, cutoff_frac)))
    h, w = x.shape[-2:]
    spec = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(h, device=x.device).abs().view(h, 1)
    fx = torch.fft.rfftfreq(w, device=x.device).view(1, -1)
    rad = torch.sqrt(fy * fy + fx * fx)
    keep = (rad / (rad.max() + 1e-8) > cf).to(spec.dtype)
    return torch.fft.irfft2(spec * keep, s=(h, w), dim=(-2, -1), norm="ortho")


def qsilk_micrograin_stabilize(
    x: torch.Tensor,
    *,
    quantile: float = 0.995,
    detail_amount: float = 0.12,
    cutoff_frac: float = 0.2,
) -> torch.Tensor:
    """Clamp extreme latent values, then add a gated high-frequency detail residual (``detail_amount=0``: clamp only)."""
    if x.ndim != 4 or float(detail_amount) <= 0.0 and float(quantile) <= 0.0:
        return x
    q = float(max(0.9, min(0.9999, quantile)))
    flat = x.float().reshape(x.shape[0], -1)
    thr = torch.quantile(flat.abs(), q, dim=1, keepdim=True).view(-1, 1, 1, 1)
    y = x.clamp(-thr.to(dtype=x.dtype), thr.to(dtype=x.dtype))
    if float(detail_amount) <= 0.0:
        return y

    high = _fft_highpass(y.float(), cutoff_frac).to(dtype=x.dtype)
    edge = high.abs().mean(dim=(1, 2, 3), keepdim=True)
    gate = (edge / (edge.mean() + 1e-6)).clamp(0.25, 2.0)
    return y + float(detail_amount) * gate * high


def spectral_latent_lowfreq_blend(x: torch.Tensor, strength: float = 0.0, cutoff_frac: float = 0.15) -> torch.Tensor:
    """``(1 - s) * x + s * lowpass(x)``; ``strength`` 0 is identity (try 0.05-0.2)."""
    if strength <= 0:
        return x
    s = float(min(1.0, strength))
    return x - s * _fft_highpass(x.float(), cutoff_frac).to(dtype=x.dtype)


def highfreq_layout_prior(x: torch.Tensor, strength: float = 0.0, kernel_size: int = 3) -> torch.Tensor:
    """``x + strength * (x - box_blur(x))``; ``strength`` 0 is identity (try 0.02-0.08)."""
    if strength <= 0:
        return x
    k = max(3, int(kernel_size) | 1)
    return x + float(strength) * (x - F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2))


# Planner keys that GaussianDiffusion.sample_loop accepts.
_DIFFUSION_SAFE_KEYS = frozenset({"step_noise_scales", "attention_layout_plan"})


def apply_token_emphasis_to_args(args: Any, *, prompt: str | None = None) -> dict[str, Any]:
    """
    Soft-apply TokenEmphasisPlanner: SD-weight fragments + CFG bump for hard tokens.

    Safe to call from quality policy / frontier prompt phase. Skips if prompt
    already contains ``(word:1.`` weight tags from a prior pass.
    """
    from frontier.planning import TokenEmphasisPlanner

    text = prompt if prompt is not None else str(getattr(args, "prompt", "") or "")
    if not text.strip():
        return {"skipped": "empty"}
    if re.search(r"\([^)]+:1\.\d+", text):
        return {"skipped": "already_weighted"}

    planner = TokenEmphasisPlanner()
    plan = planner.plan(text)
    if not plan.weights:
        return {"skipped": "no_hard_tokens"}

    args.prompt = planner.augment_with_weights(text, plan)
    cfg = getattr(args, "cfg_scale", None)
    if cfg is not None and plan.cfg_multiplier > 1.0:
        args.cfg_scale = float(cfg) * float(plan.cfg_multiplier)
    return {
        "weights": plan.as_prompt_weights(),
        "cfg_multiplier": plan.cfg_multiplier,
        "prompt": args.prompt,
    }


def merge_step_emphasis_into_noise_scales(
    *,
    step_emphasis: list[float] | tuple[float, ...] | None,
    step_noise_scales: list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    """
    Map narrative/mood ``step_emphasis`` into ``step_noise_scales`` (multiplicative).

    Sampler only understands ``step_noise_scales``; emphasis alone was a dead field.
    """
    if not step_emphasis:
        return list(step_noise_scales) if step_noise_scales is not None else None
    emp = [float(x) for x in step_emphasis]
    if not step_noise_scales:
        # Center around 1.0 so mean emphasis ≈ identity noise scale
        mean = sum(emp) / max(len(emp), 1)
        mean = mean if mean > 1e-6 else 1.0
        return [max(0.05, e / mean) for e in emp]
    base = [float(x) for x in step_noise_scales]
    n = max(len(emp), len(base))
    out: list[float] = []
    mean_e = sum(emp) / max(len(emp), 1) or 1.0
    for i in range(n):
        e = emp[min(i, len(emp) - 1)] / mean_e
        b = base[min(i, len(base) - 1)]
        out.append(max(0.05, b * e))
    return out


def sanitize_diffusion_feature_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    """Keep only planner keys ``sample_loop`` accepts (``step_emphasis`` folds into ``step_noise_scales``)."""
    if not extra:
        return {}
    work = dict(extra)
    emphasis = work.pop("step_emphasis", None)
    if emphasis is not None:
        work["step_noise_scales"] = merge_step_emphasis_into_noise_scales(
            step_emphasis=emphasis,
            step_noise_scales=work.get("step_noise_scales"),
        )
    return {k: v for k, v in work.items() if k in _DIFFUSION_SAFE_KEYS and v is not None}


_clip_bundle: tuple[object, object, str] | None = None


def _get_clip_model(model_id: str, device: torch.device) -> tuple[object, object]:
    global _clip_bundle
    if _clip_bundle is not None and _clip_bundle[2] == model_id:
        return _clip_bundle[0], _clip_bundle[1]
    from transformers import CLIPModel, CLIPProcessor

    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    _clip_bundle = (model, proc, model_id)
    return model, proc


def _clip_feature_tensor(features) -> torch.Tensor:
    """Projected CLIP embeddings from ``get_{image,text}_features`` across transformers versions.

    transformers>=5 returns a ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds the
    projected features; older versions return the tensor directly.
    """
    if isinstance(features, torch.Tensor):
        return features
    pooled = getattr(features, "pooler_output", None)
    if pooled is None and isinstance(features, tuple) and len(features) > 1:
        pooled = features[1]
    if not isinstance(pooled, torch.Tensor):
        raise TypeError(f"cannot extract CLIP features from {type(features).__name__}")
    return pooled


def tensor_chw01_to_pil(t: torch.Tensor):
    """t: (3, H, W) float [0, 1]."""
    from PIL import Image

    a = (t.detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(a)


_clip_text_feats: dict[tuple[str, str, str], torch.Tensor] = {}


def _clip_text_feature(model, proc, prompt: str, model_id: str, device: torch.device) -> torch.Tensor:
    """Normalized CLIP text embedding, cached per prompt (the mid-loop monitor re-scores the same prompt)."""
    key = (model_id, prompt, str(device))
    feat = _clip_text_feats.get(key)
    if feat is None:
        tok = proc(text=[prompt], return_tensors="pt", padding=True)
        nbc = device.type == "cuda"
        with torch.inference_mode():
            feat = _clip_feature_tensor(
                model.get_text_features(
                    input_ids=tok["input_ids"].to(device, non_blocking=nbc),
                    attention_mask=tok["attention_mask"].to(device, non_blocking=nbc)
                    if "attention_mask" in tok
                    else None,
                )
            )
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        if len(_clip_text_feats) >= 16:
            _clip_text_feats.clear()
        _clip_text_feats[key] = feat
    return feat


def clip_image_text_cosine(
    image_chw_01: torch.Tensor,
    prompt: str,
    model_id: str,
    device: torch.device,
) -> float:
    """
    Normalized CLIP cosine similarity in roughly [-1, 1] (typical positives ~0.2–0.35 for arbitrary prompts).
    """
    model, proc = _get_clip_model(model_id, device)
    nbc = device.type == "cuda"
    pixels = proc(images=tensor_chw01_to_pil(image_chw_01), return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        img_f = _clip_feature_tensor(model.get_image_features(pixel_values=pixels.to(device, non_blocking=nbc)))
        img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (img_f * _clip_text_feature(model, proc, prompt, model_id, device)).sum(dim=-1).squeeze()
    return float(sim.item())


def decode_latent_preview_rgb(
    x0: torch.Tensor,
    *,
    vae,
    latent_scale: float,
    ae_type: str,
    rae_bridge,
    batch_index: int = 0,
) -> torch.Tensor:
    """Single image (3, H, W) float 0..1 from first batch element."""
    z = x0[batch_index : batch_index + 1]
    if ae_type == "kl":
        z = z / float(latent_scale)
    elif ae_type == "rae" and rae_bridge is not None:
        z = rae_bridge.dit_to_rae(z)
    with torch.inference_mode():
        im = vae.decode(z).sample
    im = (im * 0.5 + 0.5).clamp(0, 1)
    return im[0]


def latent_x0_clip_cosine(
    x0_latent: torch.Tensor,
    *,
    prompt: str,
    model_id: str,
    device: torch.device,
    vae,
    latent_scale: float,
    ae_type: str,
    rae_bridge,
    batch_index: int = 0,
) -> float:
    """CLIP image–text cosine on a **decoded** preview of ``x0_latent`` (first batch element)."""
    rgb = decode_latent_preview_rgb(
        x0_latent,
        vae=vae,
        latent_scale=latent_scale,
        ae_type=ae_type,
        rae_bridge=rae_bridge,
        batch_index=batch_index,
    )
    return clip_image_text_cosine(rgb, prompt, model_id, device)


def maybe_clip_refine_latent(
    x0: torch.Tensor,
    *,
    prompt: str,
    sim_threshold: float,
    model_id: str,
    device: torch.device,
    vae,
    latent_scale: float,
    ae_type: str,
    rae_bridge,
    num_timesteps: int,
    t_frac: float,
    refine_steps: int,
    refiner: Callable[[torch.Tensor, int, int], torch.Tensor],
) -> torch.Tensor:
    """
    If CLIP cosine < ``sim_threshold``, run ``refiner(x0_clean, t_start, refine_steps)`` (caller supplies
    q_sample + ``sample_loop``). Threshold should be tuned per model id (try 0.18–0.28).
    """
    if sim_threshold <= -2.0:  # sentinel "off" if mis-used
        return x0
    try:
        rgb = decode_latent_preview_rgb(
            x0,
            vae=vae,
            latent_scale=latent_scale,
            ae_type=ae_type,
            rae_bridge=rae_bridge,
            batch_index=0,
        )
        sim = clip_image_text_cosine(rgb, prompt, model_id, device)
    except Exception:
        return x0
    if sim >= sim_threshold:
        return x0
    t_start = int(float(t_frac) * (num_timesteps - 1))
    t_start = min(max(1, t_start), num_timesteps - 1)
    return refiner(x0, t_start, refine_steps)
