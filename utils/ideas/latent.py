"""Latent-space ideas: initial noise, CFG scale, guidance direction, x0 estimate and per-step updates.

Every function computes in float32 and returns the input's dtype; 4-D ``(B, C, H, W)`` latents only (others pass
through). Strength 1 is the suggested amount; the stack never calls an idea at strength 0.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F

from utils.ideas import Ctx, idea


def _fp32(fn):
    """Run ``fn`` on float32 copies of its tensor arguments; cast the result back to the first argument's dtype."""

    @functools.wraps(fn)
    def wrapper(x, *args):
        if not torch.is_tensor(x) or x.ndim != 4:
            return x
        args = tuple(a.float() if torch.is_tensor(a) else a for a in args)
        return fn(x.float(), *args).to(x.dtype)

    return wrapper


def _dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.ndim))


def _renorm(x: torch.Tensor) -> torch.Tensor:
    """Per-sample zero mean, unit std (what the sampler expects of initial noise)."""
    return (x - x.mean(_dims(x), keepdim=True)) / x.std(_dims(x), keepdim=True).clamp_min(1e-6)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(_dims(x), keepdim=True).sqrt()


def _low(x: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Low-frequency part: area-downsample by ``k`` and upsample back."""
    h, w = x.shape[-2:]
    small = F.adaptive_avg_pool2d(x, (max(1, h // k), max(1, w // k)))
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def _blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), k, stride=1)


def _grid(x: torch.Tensor, cy: float = 0.5, cx: float = 0.5) -> torch.Tensor:
    """Distance from (cy, cx) in unit image coordinates, shaped (1, 1, H, W)."""
    h, w = x.shape[-2:]
    ys = (torch.arange(h, device=x.device, dtype=torch.float32) + 0.5) / h - cy
    xs = (torch.arange(w, device=x.device, dtype=torch.float32) + 0.5) / w - cx
    return (ys[:, None] ** 2 + xs[None, :] ** 2).sqrt()[None, None]


def _mean_one(g: torch.Tensor) -> torch.Tensor:
    return g / g.mean((-2, -1), keepdim=True).clamp_min(1e-8)


def _radial_freq(h: int, w: int, device) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.rfftfreq(w, device=device)
    return (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()


# ============================================================================================ noise


@idea("noise", "Multi-scale (pyramid) noise: extra low-resolution noise for bolder large-scale structure.")
@_fp32
def noise_pyramid(x, ctx: Ctx):
    h, w = x.shape[-2:]
    acc = x.clone()
    for level, k in enumerate((2, 4, 8, 16), start=1):
        if h // k < 1 or w // k < 1:
            break
        z = ctx.randn(x[..., : h // k, : w // k], salt=level)
        acc = acc + 0.6 * ctx.strength * 0.75**level * F.interpolate(z, size=(h, w), mode="bilinear")
    return _renorm(acc)


@idea("noise", "Tilt the noise spectrum toward low (positive) or high (negative strength) frequencies.")
@_fp32
def noise_spectral_tilt(x, ctx: Ctx):
    h, w = x.shape[-2:]
    f = _radial_freq(h, w, x.device).clamp_min(1.0 / max(h, w))
    gain = (f / f.mean()) ** (-0.35 * ctx.strength)
    gain[0, 0] = 1.0  # leave each channel's mean alone (no colour cast)
    return _renorm(torch.fft.irfft2(torch.fft.rfft2(x) * gain, s=(h, w)))


@idea("noise", "Phase-only noise: a perfectly flat spectrum, no random low-frequency blobs or colour casts.")
@_fp32
def noise_whiten(x, ctx: Ctx):
    h, w = x.shape[-2:]
    spec = torch.fft.rfft2(x)
    white = _renorm(torch.fft.irfft2(spec / spec.abs().clamp_min(1e-8), s=(h, w)))
    s = min(1.0, max(0.0, ctx.strength))
    return _renorm((1.0 - s) * x + s * white)


@idea("noise", "Standardise each channel of each sample (removes a random global colour bias).")
@_fp32
def noise_channel_balance(x, ctx: Ctx):
    m = x.mean((-2, -1), keepdim=True)
    sd = x.std((-2, -1), keepdim=True).clamp_min(1e-6)
    s = min(1.0, max(0.0, ctx.strength))
    return _renorm(x + s * ((x - m) / sd - x))


@idea("noise", "Shrink outlier low-frequency energy toward white-noise levels (fewer giant colour fields).")
@_fp32
def noise_lowfreq_tame(x, ctx: Ctx):
    low = _low(x, 8)
    ref = _low(ctx.randn(x), 8).std((-2, -1), keepdim=True)
    ratio = (ref / low.std((-2, -1), keepdim=True).clamp_min(1e-6)).clamp(max=1.0) ** ctx.strength
    return _renorm(x - low + low * ratio)


@idea("noise", "Gram-Schmidt the batch's noises: candidates start maximally different (better pick-best).")
@_fp32
def noise_orthogonal_batch(x, ctx: Ctx):
    b = x.shape[0]
    if b < 2:
        return x
    flat = x.reshape(b, -1)
    q, _ = torch.linalg.qr(flat.T)
    q = q.T * math.sqrt(flat.shape[1])
    q = q * torch.sign((q * flat).sum(1, keepdim=True)).clamp_min(-1)  # keep each sample's orientation
    s = min(1.0, max(0.0, ctx.strength))
    return _renorm(((1.0 - s) * flat + s * q).reshape(x.shape))


@idea("noise", "Antithetic pairs: every second candidate starts from the negated noise of the one before it.")
@_fp32
def noise_antithetic(x, ctx: Ctx):
    if x.shape[0] < 2:
        return x
    y = x.clone()
    s = min(1.0, max(0.0, ctx.strength))
    y[1::2] = (1.0 - s) * x[1::2] + s * -x[0::2][: x[1::2].shape[0]]
    return _renorm(y)


@idea("noise", "Left-right mirrored noise: symmetric compositions (faces, architecture, emblems).")
@_fp32
def noise_mirror_h(x, ctx: Ctx):
    s = min(1.0, max(0.0, ctx.strength))
    return _renorm((1.0 - s) * x + s * (x + x.flip(-1)) / math.sqrt(2.0))


@idea("noise", "Top-bottom mirrored noise: reflection-like compositions (lakes, mirrors).")
@_fp32
def noise_mirror_v(x, ctx: Ctx):
    s = min(1.0, max(0.0, ctx.strength))
    return _renorm((1.0 - s) * x + s * (x + x.flip(-2)) / math.sqrt(2.0))


@idea("noise", "Every candidate shares candidate 0's low frequencies: one layout, different details.")
@_fp32
def noise_shared_layout(x, ctx: Ctx):
    if x.shape[0] < 2:
        return x
    low = _low(x, 8)
    return _renorm(x + min(1.0, ctx.strength) * (low[:1] - low))


@idea("noise", "Every candidate shares candidate 0's high frequencies: same textures, different layouts.")
@_fp32
def noise_shared_texture(x, ctx: Ctx):
    if x.shape[0] < 2:
        return x
    high = x - _low(x, 8)
    return _renorm(x + min(1.0, ctx.strength) * (high[:1] - high))


@idea("noise", "Variation seed: slerp toward a second seed's noise (strength 1 = a 30% step).")
@_fp32
def noise_variation(x, ctx: Ctx):
    from frontier.multiview import slerp

    z = ctx.randn(x, salt=1)
    t = 0.3 * min(1.0 / 0.3, max(0.0, ctx.strength))
    return torch.stack([slerp(a, b, t) for a, b in zip(x, z, strict=True)])


def _focus_gain(x, ctx: Ctx, cy: float, cx: float, sigma: float) -> torch.Tensor:
    g = torch.exp(-(_grid(x, cy, cx) ** 2) / (2 * sigma**2))
    return 1.0 + 0.8 * ctx.strength * (g / g.mean() - 1.0)


@idea("noise", "Concentrate low-frequency energy at the centre: a prior for centred subjects.")
@_fp32
def noise_center_prior(x, ctx: Ctx):
    low = _low(x, 8)
    return _renorm(x - low + low * _focus_gain(x, ctx, 0.5, 0.5, 0.25))


@idea("noise", "Concentrate low-frequency energy at a rule-of-thirds point (picked by the seed).")
@_fp32
def noise_thirds_prior(x, ctx: Ctx):
    cy, cx = ((1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (2 / 3, 2 / 3))[int(ctx.seed) % 4]
    low = _low(x, 8)
    return _renorm(x - low + low * _focus_gain(x, ctx, cy, cx, 0.2))


@idea("noise", "Quasi-Monte-Carlo (scrambled Sobol) noise across the batch: candidates cover the space evenly.")
@_fp32
def noise_qmc(x, ctx: Ctx):
    b = x.shape[0]
    if b < 2:
        return x
    n = x[0].numel()
    eng = torch.quasirandom.SobolEngine(dimension=b, scramble=True, seed=int(ctx.seed))
    u = eng.draw(n).clamp(1e-6, 1 - 1e-6)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(ctx.seed)))
    q = z[perm].T.reshape(x.shape).to(x.device)
    s = min(1.0, max(0.0, ctx.strength))
    return _renorm((1.0 - s) * x + s * q)


# ============================================================================================ CFG scale


@idea("scale", "Bell-shaped CFG: lower at the start and end, highest mid-sampling (structure, then detail).")
def cfg_bell(w: float, ctx: Ctx) -> float:
    return 1.0 + (w - 1.0) * (1.0 - 0.3 * ctx.strength * math.cos(2.0 * math.pi * ctx.p))


@idea("scale", "Lower CFG on canvases larger than the model's native size (more tokens over-amplify guidance).")
def cfg_resolution_aware(w: float, ctx: Ctx) -> float:
    if ctx.native <= 0 or len(ctx.shape) < 4:
        return w
    ratio = ctx.native**2 / max(1, ctx.shape[-2] * ctx.shape[-1])
    return w if ratio >= 1.0 else 1.0 + (w - 1.0) * ratio ** (0.25 * ctx.strength)


@idea("scale", "Scale CFG with prompt density: dense multi-clause prompts get more guidance, short ones less.")
def cfg_prompt_density(w: float, ctx: Ctx) -> float:
    text = ctx.prompt.lower()
    clauses = len([c for c in text.split(",") if c.strip()]) + text.count(" and ")
    mult = min(1.25, max(0.8, 1.0 + 0.04 * ctx.strength * (clauses - 6)))
    return 1.0 + (w - 1.0) * mult


# ============================================================================================ guidance delta


@idea("delta", "Cap the guidance step at 60% of the conditional estimate's norm (no blown-out CFG).")
@_fp32
def delta_norm_clip(delta, d_cond, w, ctx: Ctx):
    return delta * (0.6 * _norm(d_cond) / _norm(delta)).clamp(max=1.0) ** ctx.strength


def _disagreement(delta: torch.Tensor) -> torch.Tensor:
    m = _blur(delta.abs().mean(1, keepdim=True), 3)
    return (m / m.mean((-2, -1), keepdim=True).clamp_min(1e-8)).clamp_min(1e-6)


@idea("delta", "Spend guidance where conditional and unconditional disagree (subject), less on background.")
@_fp32
def delta_spatial_focus(delta, d_cond, w, ctx: Ctx):
    return delta * _mean_one(_disagreement(delta) ** (0.5 * ctx.strength))


@idea("delta", "Even out guidance across the frame (stops an over-cooked subject on a flat background).")
@_fp32
def delta_spatial_even(delta, d_cond, w, ctx: Ctx):
    return delta * _mean_one((_disagreement(delta) ** (-0.5 * ctx.strength)).clamp(0.25, 4.0))


@idea("delta", "Remove each channel's mean from the guidance: CFG stops shifting global colour/brightness.")
@_fp32
def delta_channel_center(delta, d_cond, w, ctx: Ctx):
    return delta - min(1.0, ctx.strength) * delta.mean((-2, -1), keepdim=True)


@idea("delta", "Damp guidance components that flip sign from the previous step (less oscillation).")
@_fp32
def delta_sign_consensus(delta, d_cond, w, ctx: Ctx):
    prev = ctx.state.get("prev")
    ctx.state["prev"] = delta.detach().clone()
    if prev is None or prev.shape != delta.shape:
        return delta
    agree = torch.sign(delta) == torch.sign(prev)
    return torch.where(agree, delta, delta * (1.0 - 0.5 * min(1.0, ctx.strength)))


@idea("delta", "Soften guidance where the guided estimate leaves the latent range (prevents clipped colour).")
@_fp32
def delta_saturation_guard(delta, d_cond, w, ctx: Ctx):
    over = ((d_cond + (w - 1.0) * delta).abs() - 2.5).clamp_min(0.0)
    return delta / (1.0 + ctx.strength * over)


@idea("delta", "Sparse guidance: keep only the strongest 40% of guidance values (edits, not global shifts).")
@_fp32
def delta_topk_sparse(delta, d_cond, w, ctx: Ctx):
    flat = delta.abs().flatten(1)
    keep = min(1.0, max(0.05, 1.0 - 0.6 * ctx.strength))
    k = max(1, min(flat.shape[1], int(round((1.0 - keep) * flat.shape[1]))))
    thr = flat.kthvalue(k, dim=1).values.view(-1, 1, 1, 1)
    return torch.where(delta.abs() >= thr, delta, torch.zeros_like(delta))


@idea("delta", "Per-channel CFG rescale: the guided estimate keeps the conditional one's channel mean and std.")
@_fp32
def delta_channel_rescale(delta, d_cond, w, ctx: Ctx):
    if w == 1.0:
        return delta
    g = d_cond + (w - 1.0) * delta
    mg, sg = g.mean((-2, -1), keepdim=True), g.std((-2, -1), keepdim=True).clamp_min(1e-6)
    mc, sc = d_cond.mean((-2, -1), keepdim=True), d_cond.std((-2, -1), keepdim=True)
    g2 = g + 0.7 * min(1.0, ctx.strength) * ((g - mg) / sg * sc + mc - g)
    return (g2 - d_cond) / (w - 1.0)


@idea("delta", "Spatially smooth the guidance (no grainy CFG noise at high scales).")
@_fp32
def delta_smooth(delta, d_cond, w, ctx: Ctx):
    a = 0.5 * min(2.0, ctx.strength)
    return (1.0 - a) * delta + a * _blur(delta, 3)


@idea("delta", "Clamp guidance outliers to 2.5 standard deviations (kills hot spots).")
@_fp32
def delta_outlier_clamp(delta, d_cond, w, ctx: Ctx):
    k = max(1.0, 4.0 - 1.5 * ctx.strength) * delta.std(_dims(delta), keepdim=True)
    return torch.maximum(torch.minimum(delta, k), -k)


@idea("delta", "Auto-CFG: steer the effective guidance to ~35% of the conditional estimate, per sample.")
@_fp32
def delta_auto_ratio(delta, d_cond, w, ctx: Ctx):
    if w <= 1.0:
        return delta
    r = (w - 1.0) * _norm(delta) / _norm(d_cond)
    return delta * ((0.35 / r.clamp_min(1e-6)) ** (0.5 * ctx.strength)).clamp(0.5, 2.0)


@idea("delta", "Guide low frequencies early and high frequencies late (layout first, then detail).")
@_fp32
def delta_freq_tilt(delta, d_cond, w, ctx: Ctx):
    low = _low(delta, 4)
    a = 0.5 * ctx.strength * (1.0 - 2.0 * ctx.p)
    return low * (1.0 + a) + (delta - low) * (1.0 - a)


@idea("delta", "Weight guidance by the conditional estimate's edges (structure gets the CFG, flats less).")
@_fp32
def delta_edge_focus(delta, d_cond, w, ctx: Ctx):
    d = d_cond.mean(1, keepdim=True)
    gx = F.pad(d[..., :, 1:] - d[..., :, :-1], (0, 1, 0, 0), mode="replicate")
    gy = F.pad(d[..., 1:, :] - d[..., :-1, :], (0, 0, 0, 1), mode="replicate")
    e = _mean_one((gx**2 + gy**2).sqrt())
    s = min(1.0, ctx.strength)
    return delta * _mean_one((1.0 - 0.5 * s) + 0.5 * s * e)


@idea("delta", "Stronger guidance in the centre, softer at the borders (less busy backgrounds).")
@_fp32
def delta_center_weight(delta, d_cond, w, ctx: Ctx):
    r = _grid(delta)
    return delta * _mean_one(1.0 + 0.3 * ctx.strength * (1.0 - 2.0 * (r / r.max()) ** 2))


@idea("delta", "Low-rank guidance: keep the guidance's top half of channel components (denoised direction).")
@_fp32
def delta_lowrank(delta, d_cond, w, ctx: Ctx):
    b, c, h, wd = delta.shape
    if c < 2:
        return delta
    u, sv, vh = torch.linalg.svd(delta.reshape(b, c, h * wd), full_matrices=False)
    r = max(1, c // 2)
    lowrank = (u[..., :r] * sv[:, None, :r]) @ vh[:, :r]
    s = min(1.0, ctx.strength)
    return (1.0 - s) * delta + s * lowrank.reshape(delta.shape)


# ============================================================================================ x0 estimate


@idea("x0", "Soft-clip the clean-latent estimate with tanh beyond +-3 (gentler than dynamic thresholding).")
@_fp32
def x0_soft_clip(x0, ctx: Ctx):
    return x0 + min(1.0, ctx.strength) * (3.0 * torch.tanh(x0 / 3.0) - x0)


def _capture(ctx: Ctx, key: str, value: torch.Tensor, at: float = 0.3) -> torch.Tensor | None:
    ref = ctx.state.get(key)
    if ref is None and ctx.p >= at:
        ctx.state[key] = value.detach().clone()
        return None
    return ref if ref is not None and ref.shape == value.shape else None


@idea("x0", "Hold each channel's contrast at its 30%-progress value (no late washed-out or crunchy drift).")
@_fp32
def x0_channel_renorm(x0, ctx: Ctx):
    sd = x0.std((-2, -1), keepdim=True).clamp_min(1e-6)
    ref = _capture(ctx, "std", sd)
    if ref is None:
        return x0
    m = x0.mean((-2, -1), keepdim=True)
    return m + (x0 - m) * (1.0 + 0.5 * min(1.0, ctx.strength) * (ref / sd - 1.0))


@idea("x0", "Anchor each channel's mean at its 30%-progress value (colour decided early stays put).")
@_fp32
def x0_mean_anchor(x0, ctx: Ctx):
    m = x0.mean((-2, -1), keepdim=True)
    ref = _capture(ctx, "mean", m)
    return x0 if ref is None else x0 + 0.8 * min(1.0, ctx.strength) * (ref - m)


@idea("x0", "Sharpen the clean estimate mid-sampling (30-80%), when the model is still forming detail.")
@_fp32
def x0_midstep_sharpen(x0, ctx: Ctx):
    if not 0.3 <= ctx.p <= 0.8:
        return x0
    amount = 0.3 * ctx.strength * math.sin(math.pi * (ctx.p - 0.3) / 0.5)
    return x0 + amount * (x0 - _blur(x0, 3))


@idea("x0", "Pull the clean estimate toward left-right symmetry.")
@_fp32
def x0_mirror_h(x0, ctx: Ctx):
    return x0 + 0.5 * min(1.0, ctx.strength) * (x0.flip(-1) - x0)


@idea("x0", "Pull the clean estimate toward top-bottom symmetry.")
@_fp32
def x0_mirror_v(x0, ctx: Ctx):
    return x0 + 0.5 * min(1.0, ctx.strength) * (x0.flip(-2) - x0)


@idea("x0", "Grow latent contrast slightly as sampling ends (counters the hazy 'AI wash').")
@_fp32
def x0_contrast_stretch(x0, ctx: Ctx):
    m = x0.mean((-2, -1), keepdim=True)
    return m + (x0 - m) * (1.0 + 0.08 * ctx.strength * ctx.p)


@idea("x0", "Exponential moving average of the clean estimate across steps (steadier trajectories).")
@_fp32
def x0_temporal_ema(x0, ctx: Ctx):
    prev = ctx.state.get("ema")
    beta = 0.25 * min(1.0, ctx.strength) * (1.0 - ctx.p)
    out = x0 if prev is None or prev.shape != x0.shape else (1.0 - beta) * x0 + beta * prev
    ctx.state["ema"] = out.detach().clone()
    return out


@idea("x0", "More fine detail at the centre than the edges (a subject-focus / lens-like falloff).")
@_fp32
def x0_center_focus(x0, ctx: Ctx):
    g = torch.exp(-(_grid(x0) ** 2) / (2 * 0.3**2))
    return x0 + 0.3 * ctx.strength * (g / g.mean() - 1.0) * (x0 - _blur(x0, 3))


@idea("x0", "Lock the composition after 40%: later steps may change detail but not the low frequencies.")
@_fp32
def x0_lowfreq_lock(x0, ctx: Ctx):
    low = _low(x0, 8)
    ref = _capture(ctx, "low", low, at=0.4)
    return x0 if ref is None else x0 - min(1.0, ctx.strength) * (low - ref)


@idea("x0", "Clamp extreme high-frequency latent energy (no 'deep-fried' VAE decode texture).")
@_fp32
def x0_highfreq_clamp(x0, ctx: Ctx):
    low = _blur(x0, 3)
    high = x0 - low
    k = max(1.5, 3.5 - 1.5 * ctx.strength) * high.std((-2, -1), keepdim=True)
    return low + torch.maximum(torch.minimum(high, k), -k)


# ============================================================================================ per step


@idea("step", "Particle guidance: candidates repel each other in layout space (diverse batch).")
@_fp32
def step_particle_repulsion(x_new, x_prev, x0, ctx: Ctx):
    b = x_new.shape[0]
    if b < 2:
        return x_new
    feats = F.adaptive_avg_pool2d(x_new, 8)
    f = feats.flatten(1)
    diff = f[:, None, :] - f[None, :, :]
    dist2 = diff.pow(2).sum(-1)
    off = dist2[~torch.eye(b, dtype=torch.bool, device=x_new.device)]
    h = (off.median() / math.log(b + 1.0)).clamp_min(1e-6)
    push = ((torch.exp(-dist2 / h)[..., None] * diff).sum(1) * (2.0 / h)).view_as(feats)
    push = F.interpolate(push, size=x_new.shape[-2:], mode="bilinear", align_corners=False)
    push = push / _rms(push).clamp_min(1e-8)
    return x_new + 0.05 * ctx.strength * (1.0 - ctx.p) * _rms(x_new - x_prev) * push


@idea("step", "Late churn: a little fresh noise at 70-95% progress for natural micro-texture.")
@_fp32
def step_late_churn(x_new, x_prev, x0, ctx: Ctx):
    if not 0.7 <= ctx.p <= 0.95:
        return x_new
    return x_new + 0.15 * ctx.strength * _rms(x_new - x_prev) * ctx.randn(x_new)


@idea("step", "Clamp the latent to +-4 in the second half (catches runaway values before decode).")
@_fp32
def step_latent_clamp(x_new, x_prev, x0, ctx: Ctx):
    if ctx.p < 0.5:
        return x_new
    k = max(2.0, 6.0 - 2.0 * ctx.strength)
    return x_new.clamp(-k, k)


@idea("step", "Heavy-ball momentum on the trajectory, fading out by the end (sharper results at few steps).")
@_fp32
def step_momentum(x_new, x_prev, x0, ctx: Ctx):
    v, prev = x_new - x_prev, ctx.state.get("v")
    ctx.state["v"] = v.detach().clone()
    if prev is None or prev.shape != v.shape:
        return x_new
    return x_new + 0.15 * min(1.0, ctx.strength) * (1.0 - ctx.p) * prev


@idea("step", "Pull the batch's layouts together early (consistent candidates for a series or comic).")
@_fp32
def step_batch_align(x_new, x_prev, x0, ctx: Ctx):
    if x_new.shape[0] < 2:
        return x_new
    low = _low(x_new, 8)
    return x_new + 0.15 * ctx.strength * (1.0 - ctx.p) * (low.mean(0, keepdim=True) - low)


@idea("step", "Undo sudden jumps in latent spread between steps (variance collapse / explosion guard).")
@_fp32
def step_std_guard(x_new, x_prev, x0, ctx: Ctx):
    r = x_new.std(_dims(x_new), keepdim=True) / x_prev.std(_dims(x_prev), keepdim=True).clamp_min(1e-6)
    f = (r.clamp(0.8, 1.25) / r.clamp_min(1e-6)) ** min(1.0, ctx.strength)
    m = x_new.mean(_dims(x_new), keepdim=True)
    return m + (x_new - m) * f


@idea("step", "Inject high-pass noise mid-sampling (40-80%): grows fine texture instead of plastic surfaces.")
@_fp32
def step_detail_noise(x_new, x_prev, x0, ctx: Ctx):
    if not 0.4 <= ctx.p <= 0.8:
        return x_new
    z = ctx.randn(x_new)
    return x_new + 0.1 * ctx.strength * _rms(x_new - x_prev) * (z - _blur(z, 3))


@idea("step", "Calm the borders early: less high-frequency clutter in a 12% frame band.")
@_fp32
def step_border_calm(x_new, x_prev, x0, ctx: Ctx):
    if ctx.p > 0.6:
        return x_new
    h, w = x_new.shape[-2:]
    ys = torch.minimum(torch.arange(h, device=x_new.device), torch.arange(h, device=x_new.device).flip(0)) / h
    xs = torch.minimum(torch.arange(w, device=x_new.device), torch.arange(w, device=x_new.device).flip(0)) / w
    band = (1.0 - (torch.minimum(ys[:, None], xs[None, :]) / 0.12).clamp(max=1.0)).float()[None, None]
    return x_new - 0.3 * min(1.0, ctx.strength) * band * (x_new - _blur(x_new, 3))


@idea("step", "Final 10%: smooth latent speckle in low-variance regions only (clean skies and skin).")
@_fp32
def step_final_speckle(x_new, x_prev, x0, ctx: Ctx):
    if ctx.p < 0.9:
        return x_new
    high = x_new - _blur(x_new, 3)
    var = _blur(high.pow(2), 3)
    calm = 1.0 - var / var.amax((-2, -1), keepdim=True).clamp_min(1e-8)
    return x_new - 0.2 * min(1.0, ctx.strength) * calm * high


@idea("step", "Damp a step that moves the latent 3x more than the running median (divergence guard).")
@_fp32
def step_trajectory_guard(x_new, x_prev, x0, ctx: Ctx):
    d = _norm(x_new - x_prev)
    hist = ctx.state.setdefault("hist", [])
    hist.append(float(d.mean()))
    if len(hist) < 4:
        return x_new
    limit = 3.0 * float(torch.tensor(hist[:-1]).median())
    if hist[-1] <= limit:
        return x_new
    return x_prev + (x_new - x_prev) * (limit / d).clamp(max=1.0) ** min(1.0, ctx.strength)
