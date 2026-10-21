"""VP Gaussian diffusion for the DiT: training losses (offset noise, min-SNR, eps/v/x0 prediction) and the
sampling loop (VP solvers and rectified flow, CFG and the guidance stack)."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from .cfg_schedulers import cfg_scale_cosine_ramp, cfg_scale_linear, cfg_scale_piecewise, cfg_scale_snr_aware_multiplier
from .inference_timesteps import build_inference_timesteps
from .sampling_utils import (
    dynamic_threshold,
    gaussian_blur_latent,
    norm_thresholding,
    spatial_norm_thresholding,
    unsharp_mask_latent,
)
from .schedules import alpha_cumprod_from_betas, get_beta_schedule, snr_from_alpha_cumprod, space_timesteps
from .spectral_sfp import spectral_sfp_per_sample_loss
from .timestep_sampling import get_timestep_loss_weight

# Canonical VP update backends of ``sample_loop``; ``INFERENCE_SOLVER_ALIASES`` maps CLI names onto them.
_SOLVER_DDIM = "ddim"
_SOLVER_HEUN = "heun"
_SOLVER_DPMPP_2M = "dpmpp_2m"
_SOLVER_DPMPP_2M_SDE = "dpmpp_2m_sde"
_SOLVER_DPMPP_3M = "dpmpp_3m"
_SOLVER_UNIPC = "unipc"
_SOLVER_APEX = "apex"
_SOLVER_APEX_ADAPTIVE = "apex_adaptive"


def _build_block_cache(
    threshold: float,
    recompute_every: int,
    *,
    taylor: bool = False,
    taylor_order: int = 1,
    taylor_interval: int = 4,
    dbc_separate_cfg: bool = True,
):
    if float(threshold) <= 0.0:
        return None
    if bool(taylor):
        from utils.superior.caches import TaylorBlockCache, TaylorCacheConfig

        return TaylorBlockCache(
            TaylorCacheConfig(
                rel_l1_threshold=float(threshold),
                recompute_every=max(1, int(recompute_every)),
                use_taylor=True,
                max_order=max(0, int(taylor_order)),
                cache_interval=max(1, int(taylor_interval)),
            )
        )
    from utils.superior.caches import BlockCacheConfig, BlockDiTCache

    return BlockDiTCache(
        BlockCacheConfig(
            rel_l1_threshold=float(threshold),
            max_reuse_streak=max(1, int(recompute_every)),
            recompute_every=max(1, int(recompute_every)),
            cfg_split_fingerprint=bool(dbc_separate_cfg),
        )
    )


INFERENCE_SOLVERS = (
    _SOLVER_DDIM,
    _SOLVER_HEUN,
    _SOLVER_DPMPP_2M,
    _SOLVER_DPMPP_2M_SDE,
    _SOLVER_DPMPP_3M,
    _SOLVER_UNIPC,
    _SOLVER_APEX,
    _SOLVER_APEX_ADAPTIVE,
)
# Solvers that inject noise when eta > 0; eta defaults to 1 for the SDE solvers and 0 otherwise.
_VP_STOCHASTIC = (_SOLVER_DDIM, _SOLVER_HEUN, _SOLVER_DPMPP_2M_SDE)
_FLOW_STOCHASTIC = (_SOLVER_DPMPP_2M_SDE,)
_SDE_SOLVERS = (_SOLVER_DPMPP_2M_SDE,)

INFERENCE_SOLVER_ALIASES: dict[str, str] = {
    "ddim": _SOLVER_DDIM,
    # These names have always meant DDIM on the VP path (not EDM Euler); kept for CLI compatibility.
    "euler": _SOLVER_DDIM,
    "euler_explicit": _SOLVER_DDIM,
    "euler_explicit_1": _SOLVER_DDIM,
    "edm_euler": _SOLVER_DDIM,
    "edm-euler": _SOLVER_DDIM,
    "edm_euler_a": _SOLVER_DDIM,
    "dpmpp_2m": _SOLVER_DPMPP_2M,
    "dpm++_2m": _SOLVER_DPMPP_2M,
    "dpm++2m": _SOLVER_DPMPP_2M,
    "dpmsolver++_2m": _SOLVER_DPMPP_2M,
    "dpmsolver_pp_2m": _SOLVER_DPMPP_2M,
    "dpmsolver_pp_1": _SOLVER_DPMPP_2M,
    "dpmpp_2m_sde": _SOLVER_DPMPP_2M_SDE,
    "dpm++_2m_sde": _SOLVER_DPMPP_2M_SDE,
    "dpm++2m_sde": _SOLVER_DPMPP_2M_SDE,
    "dpmsolver++_2m_sde": _SOLVER_DPMPP_2M_SDE,
    "dpmpp_3m": _SOLVER_DPMPP_3M,
    "dpm++_3m": _SOLVER_DPMPP_3M,
    "dpm++3m": _SOLVER_DPMPP_3M,
    "dpmsolver++_3m": _SOLVER_DPMPP_3M,
    "unipc": _SOLVER_UNIPC,
    "uni_pc": _SOLVER_UNIPC,
    "uni-pc": _SOLVER_UNIPC,
    "heun": _SOLVER_HEUN,
    "heun_2": _SOLVER_HEUN,
    "edm_heun": _SOLVER_HEUN,
    "edm-heun": _SOLVER_HEUN,
    "predictor_corrector": _SOLVER_HEUN,
    "predictor-corrector": _SOLVER_HEUN,
    "pc": _SOLVER_HEUN,
    "pc2": _SOLVER_HEUN,
    "rk2": _SOLVER_HEUN,
    "apex": _SOLVER_APEX,
    "apex_adaptive": _SOLVER_APEX_ADAPTIVE,
}

FLOW_SOLVER_ALIASES = {
    "euler": "euler",
    "euler_forward": "euler",
    "heun": "heun",
    "edm_heun": "heun",
    "edm-heun": "heun",
    "rk2": "heun",
    "predictor_corrector": "heun",
    "predictor-corrector": "heun",
    "midpoint": "midpoint",
    "midpoint_pc": "midpoint",
    "dpmpp_2m": "dpmpp_2m",
    "dpm++_2m": "dpmpp_2m",
    "dpmsolver++_2m": "dpmpp_2m",
    "dpmpp_2m_sde": "dpmpp_2m_sde",
    "dpm++_2m_sde": "dpmpp_2m_sde",
    "dpmsolver++_2m_sde": "dpmpp_2m_sde",
    "apex": "apex",
}


def _canonicalize(name: str, aliases: dict[str, str], kind: str) -> str:
    key = str(name).lower().strip()
    if key not in aliases:
        raise ValueError(f"Unknown {kind} {name!r}. Known aliases: {', '.join(sorted(aliases))}")
    return aliases[key]


def canonicalize_vp_solver(name: str) -> str:
    return _canonicalize(name, INFERENCE_SOLVER_ALIASES, "VP inference solver")


def canonicalize_flow_solver(name: str) -> str:
    return _canonicalize(name, FLOW_SOLVER_ALIASES, "flow solver")


def _autocast_prediction(fn: Callable, device, enabled: bool, dtype: torch.dtype | None) -> Callable:
    """Run a model-prediction closure under CUDA autocast and hand back the input dtype.

    Off-GPU this is a no-op: CPU autocast is bf16-only and usually slower than fp32 here.
    """
    if not enabled or torch.device(device).type != "cuda":
        return fn
    amp_dtype = dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    def run(x_in: torch.Tensor, *args):
        with torch.autocast("cuda", dtype=amp_dtype):
            return fn(x_in, *args).to(x_in.dtype)

    return run


def _control_guidance_scale_for_step(
    base_scale: float,
    step_index: int,
    total_steps: int,
    *,
    start: float = 0.0,
    end: float = 1.0,
    decay_power: float = 1.0,
) -> float:
    """ControlNet scale at a step: ``base_scale`` decaying as ``(1-u)^decay_power`` across the progress window
    ``[start, end]`` (``u`` is the position inside it) and 0 outside; a zero-width window applies only to step 0."""
    b = float(base_scale)
    if b <= 0.0 or total_steps <= 0:
        return 0.0
    s, e = sorted((max(0.0, min(1.0, float(start))), max(0.0, min(1.0, float(end)))))
    if e <= s:
        return b if step_index <= 0 else 0.0
    p = 0.0 if total_steps <= 1 else step_index / (total_steps - 1)
    if p < s or p > e:
        return 0.0
    u = (p - s) / (e - s)
    return b * max(0.0, min(1.0, (1.0 - u) ** max(1e-6, float(decay_power))))


def _scale_control_value(control_scale, factor: float):
    """Scale a control / adapter strength that may be a tensor, a per-control list or a scalar."""
    f = float(factor)
    if torch.is_tensor(control_scale):
        return control_scale * f
    if isinstance(control_scale, (list, tuple)):
        return [float(x) * f for x in control_scale]
    try:
        return float(control_scale) * f
    except (TypeError, ValueError):
        return control_scale


def offset_noise(noise: torch.Tensor, noise_offset: float) -> torch.Tensor:
    """Noise offset (Guttenberg 2023): add ``noise_offset * N(0, 1)`` constant per sample and channel, so the model
    learns to move each channel's mean (overall brightness / colour cast) instead of keeping it at the data mean."""
    if noise_offset <= 0:
        return noise
    shape = noise.shape[:2] + (1,) * (noise.ndim - 2)
    return noise + float(noise_offset) * torch.randn(shape, device=noise.device, dtype=noise.dtype)


def create_diffusion(
    timestep_respacing: str = "",
    num_timesteps: int = 1000,
    beta_schedule: str = "linear",
    loss_type: str = "mse",
    prediction_type: str = "epsilon",
):
    """``GaussianDiffusion`` with ``prediction_type`` ``epsilon`` | ``v`` | ``x0`` (use the same at train and sample time).

    ``timestep_respacing``: "" = all steps; "50" = that many evenly spaced; "ddim50" = DDIM striding;
    "10,15,20" = per-section step counts. Only the ``ddim``/section forms change sampling (``set_timesteps``).
    """
    respacing_str = None
    if not timestep_respacing:
        use_timesteps = np.arange(num_timesteps)
    else:
        s = str(timestep_respacing).strip()
        if s.startswith("ddim") or "," in s:
            respacing_str = timestep_respacing
            use_timesteps = space_timesteps(num_timesteps, s)
        else:
            steps = int(s)
            use_timesteps = np.linspace(0, num_timesteps - 1, steps).astype(int)
    return GaussianDiffusion(
        num_timesteps=num_timesteps,
        use_timesteps=use_timesteps,
        beta_schedule=beta_schedule,
        loss_type=loss_type,
        prediction_type=prediction_type,
        timestep_respacing_str=respacing_str,
    )


class GaussianDiffusion:
    def __init__(
        self,
        num_timesteps=1000,
        use_timesteps=None,
        beta_schedule="linear",
        loss_type="mse",
        prediction_type="epsilon",
        timestep_respacing_str=None,
    ):
        if use_timesteps is None:
            use_timesteps = np.arange(num_timesteps)
        self.num_timesteps = int(num_timesteps)
        self.use_timesteps = np.asarray(use_timesteps)
        self.timestep_respacing_str = timestep_respacing_str  # "ddim50" or "10,15,20" for set_timesteps
        self.loss_type = loss_type
        self.prediction_type = prediction_type  # "epsilon" | "v" | "x0"
        beta = get_beta_schedule(beta_schedule, num_timesteps)
        alpha_cumprod = alpha_cumprod_from_betas(beta)
        snr = snr_from_alpha_cumprod(alpha_cumprod)
        alpha_cumprod_prev = np.concatenate([[1.0], alpha_cumprod[:-1]])
        self.beta = torch.from_numpy(beta).float()
        self.alpha_cumprod = torch.from_numpy(alpha_cumprod).float()
        self.alpha_cumprod_prev = torch.from_numpy(alpha_cumprod_prev).float()
        self.sqrt_alpha_cumprod = torch.from_numpy(np.sqrt(alpha_cumprod)).float()
        self.sqrt_one_minus_alpha_cumprod = torch.from_numpy(np.sqrt(1.0 - alpha_cumprod)).float()
        self.snr = torch.from_numpy(snr).float()

    _BUFFERS = (
        "beta",
        "alpha_cumprod",
        "alpha_cumprod_prev",
        "sqrt_alpha_cumprod",
        "sqrt_one_minus_alpha_cumprod",
        "snr",
    )

    def _to_device(self, device):
        for name in self._BUFFERS:
            setattr(self, name, getattr(self, name).to(device))

    def _coef(self, name: str, t, x: torch.Tensor) -> torch.Tensor:
        """Schedule buffer ``name`` at ``t`` on ``x``'s device, shaped to broadcast against ``x``."""
        self._to_device(x.device)
        return getattr(self, name)[t][(...,) + (None,) * (x.ndim - 1)]

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion ``x_t = sqrt(abar) x_0 + sqrt(1 - abar) noise`` (pass the exact training noise)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._coef("sqrt_alpha_cumprod", t, x_start) * x_start
            + self._coef("sqrt_one_minus_alpha_cumprod", t, x_start) * noise
        )

    def training_target(self, x_start, noise, t):
        """Regression target for ``prediction_type``: ``noise`` (eps), ``sqrt(abar) noise - sqrt(1-abar) x_0`` (v) or ``x_0``."""
        if self.prediction_type == "x0":
            return x_start
        if self.prediction_type != "v":
            return noise
        return (
            self._coef("sqrt_alpha_cumprod", t, x_start) * noise
            - self._coef("sqrt_one_minus_alpha_cumprod", t, x_start) * x_start
        )

    def timestep_loss_weight(self, t, loss_weighting="min_snr", min_snr_gamma=5.0, loss_weighting_sigma_data=0.5):
        """``(B,)`` weight for the per-sample MSE at ``t`` (see ``diffusion.timestep_sampling``)."""
        self._to_device(t.device)
        return get_timestep_loss_weight(
            loss_weighting,
            snr=self.snr[t],
            alpha_cumprod=self.alpha_cumprod[t],
            min_snr_gamma=min_snr_gamma,
            loss_weighting_sigma_data=loss_weighting_sigma_data,
            prediction_type=self.prediction_type,
        )

    def per_sample_training_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        refinement_prob=0.0,
        refinement_max_t=150,
        noise_offset=0.0,
        min_snr_gamma=5.0,
        loss_weighting="min_snr",
        loss_weighting_sigma_data=0.5,
        use_spectral_sfp_loss=False,
        spectral_sfp_low_sigma=0.22,
        spectral_sfp_high_sigma=0.22,
        spectral_sfp_tau_power=1.0,
    ):
        """
        Same forward as ``training_losses`` but returns per-row loss ``(B,)`` before batch mean.
        Used for Diffusion-DPO and other per-example objectives. ``noise_offset`` is applied once to ``noise``
        (see :func:`offset_noise`), so ``x_t`` and the target see the same noise.
        """
        device = x_start.device
        if refinement_prob > 0 and refinement_max_t > 0 and torch.rand(1, device=device).item() < refinement_prob:
            t = torch.randint(
                0, min(refinement_max_t, self.num_timesteps), (x_start.shape[0],), device=device, dtype=t.dtype
            )
        if noise is None:
            noise = torch.randn_like(x_start)
        noise = offset_noise(noise, noise_offset)
        x_t = self.q_sample(x_start, t, noise=noise)
        model_out = model(x_t, t, **(model_kwargs or {}))
        if model_out.shape != x_start.shape and model_out.shape[1] > x_start.shape[1]:
            model_out = model_out[:, : x_start.shape[1]]
        target = self.training_target(x_start, noise, t)
        if use_spectral_sfp_loss:
            loss = spectral_sfp_per_sample_loss(
                model_out,
                target,
                t,
                self.num_timesteps,
                low_sigma=float(spectral_sfp_low_sigma),
                high_sigma=float(spectral_sfp_high_sigma),
                tau_power=float(spectral_sfp_tau_power),
            )
        else:
            loss = nn.functional.mse_loss(model_out, target, reduction="none")
            loss = loss.mean(dim=tuple(range(1, loss.ndim)))
        return loss * self.timestep_loss_weight(t, loss_weighting, min_snr_gamma, loss_weighting_sigma_data)

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, *, sample_weights=None, **loss_kwargs):
        """``{"loss": scalar}``: the batch mean of :meth:`per_sample_training_losses` (``loss_kwargs``: offset noise,
        refinement, ``loss_weighting`` min_snr | min_snr_soft | unit | edm | v | eps, spectral SFP), weighted by the
        optional ``(B,)`` ``sample_weights`` (e.g. aesthetic score)."""
        loss = self.per_sample_training_losses(model, x_start, t, model_kwargs=model_kwargs, noise=noise, **loss_kwargs)
        if sample_weights is not None and sample_weights.shape[0] == loss.shape[0]:
            sample_weights = sample_weights.to(loss.device, dtype=loss.dtype)
            return {"loss": (loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)}
        return {"loss": loss.mean()}

    def _guidance_space(self, t: torch.Tensor, ndim: int, *, flow: bool, need_sigma: bool):
        """``((a, b), sigma)`` for the guidance stack: ``x0 = a*x_t + b*model_out`` for this parameterisation,
        and the step's EDM noise level (VP ``sqrt((1-abar)/abar)``, flow ``s/(1-s)``) when ``need_sigma``."""
        shape = (-1,) + (1,) * (ndim - 1)
        if flow:
            s = t.to(torch.float32).view(shape) / float(max(self.num_timesteps - 1, 1))
            affine = (1.0, -s)
            noise = s / (1.0 - s).clamp_min(1e-12)
        else:
            self._to_device(t.device)
            abar = self.alpha_cumprod[t].view(shape)
            sa, so = abar.sqrt(), (1.0 - abar).sqrt()
            affine = {"v": (sa, -so), "x0": (0.0, 1.0)}.get(self.prediction_type, (1.0 / sa, -so / sa))
            noise = so / sa
        return affine, (float(noise.flatten()[0]) if need_sigma else None)

    def _predict_x0_and_noise(self, model_out, x_t, t):
        """Convert model output (epsilon, v, or x0) to x_0 and implied noise."""
        sigma = self._coef("sqrt_one_minus_alpha_cumprod", t, x_t)
        sqrt_alpha = self._coef("sqrt_alpha_cumprod", t, x_t)
        if self.prediction_type == "v":
            pred_noise = (sqrt_alpha * model_out + sigma * x_t) / (sigma + 1e-8)
            x_0_pred = sqrt_alpha * x_t - sigma * model_out
        elif self.prediction_type == "x0":
            x_0_pred = model_out
            pred_noise = (x_t - sqrt_alpha * x_0_pred) / (sigma + 1e-8)
        else:
            pred_noise = model_out
            x_0_pred = (x_t - sigma * pred_noise) / (sqrt_alpha + 1e-8)
        return x_0_pred, pred_noise

    def p_step(self, model, x_t, t, model_kwargs=None):
        """Deterministic (DDIM, eta=0) step ``x_t -> x_{t-1}``; returns ``(x_0_pred, x_prev)``."""
        out = model(x_t, t, **(model_kwargs or {}))
        if out.shape != x_t.shape and out.shape[1] > x_t.shape[1]:
            out = out[:, : x_t.shape[1]]
        x_0_pred, _ = self._predict_x0_and_noise(out, x_t, t)
        noise_impl = (x_t - self._coef("sqrt_alpha_cumprod", t, x_t) * x_0_pred) / (
            self._coef("sqrt_one_minus_alpha_cumprod", t, x_t) + 1e-8
        )
        t_prev = (t - 1).clamp(min=0)
        x_prev = (
            self._coef("sqrt_alpha_cumprod", t_prev, x_t) * x_0_pred
            + self._coef("sqrt_one_minus_alpha_cumprod", t_prev, x_t) * noise_impl
        )
        return x_0_pred, x_prev

    def set_timesteps(
        self,
        num_inference_steps: int,
        timestep_schedule: str = "ddim",
        *,
        scheduler: str | None = None,
        karras_rho: float = 7.0,
    ):
        """Set and return ``self.timesteps`` (noisy -> clean) for a schedule in
        ``diffusion.inference_timesteps.list_timestep_schedules()`` or ``indices:...``; ``scheduler`` is a legacy
        alias for ``timestep_schedule``. ``ddim`` uses the construction-time respacing (e.g. ``ddim50``) when one
        was given; every other schedule ignores it.
        """
        raw = str(scheduler if scheduler is not None else timestep_schedule).strip()
        if self.timestep_respacing_str and raw.lower() == "ddim":
            steps = space_timesteps(self.num_timesteps, self.timestep_respacing_str)[::-1]
        else:
            ac = self.alpha_cumprod.detach().cpu().numpy()
            steps = build_inference_timesteps(
                raw, self.num_timesteps, num_inference_steps, ac, karras_rho=float(karras_rho)
            )
        self.timesteps = torch.from_numpy(np.array(steps, dtype=np.int64))
        return self.timesteps

    def set_timesteps_from(self, num_inference_steps: int, t_start: int):
        """Img2img / from-z: timesteps from t_start down to 0 (FLUX/SD-style)."""
        t_start = min(max(0, int(t_start)), self.num_timesteps - 1)
        # High noise -> low, strictly decreasing: repeated indices would be zero-length solver steps.
        steps = np.unique(np.round(np.linspace(t_start, 0, num_inference_steps)).astype(np.int64))[::-1]
        self.timesteps = torch.from_numpy(steps.copy())
        return self.timesteps

    def _dynamic_threshold(
        self,
        x: torch.Tensor,
        percentile: float = 0.0,
        threshold_type: str = "percentile",
        threshold_value: float = 0.0,
    ) -> torch.Tensor:
        """Threshold an x0 prediction. ``percentile``: Imagen dynamic thresholding with data range
        ``threshold_value`` (default 1, the pixel range; use the latent range for VAE latents).
        ``norm`` / ``spatial_norm``: cap the per-sample / per-location RMS at ``threshold_value``."""
        if threshold_type == "norm" and threshold_value > 0:
            return norm_thresholding(x, threshold_value)
        if threshold_type == "spatial_norm" and threshold_value > 0:
            return spatial_norm_thresholding(x, threshold_value)
        if threshold_type == "percentile" and 0 < percentile < 100:
            return dynamic_threshold(x, percentile / 100.0, threshold_value if threshold_value > 0 else 1.0)
        return x

    def _abar(self, t) -> float:
        return float(self.alpha_cumprod[int(t.reshape(-1)[0]) if torch.is_tensor(t) else int(t)])

    def _ddim_step(self, x: torch.Tensor, t, t_next, x0: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """DDIM (Song et al. 2021) from ``t`` to ``t_next`` given the data prediction; ``eta`` > 0 adds DDPM-style noise."""
        a, a_n = self._abar(t), self._abar(t_next)
        eps = (x - a**0.5 * x0) / (1.0 - a) ** 0.5
        sig = float(eta) * max((1.0 - a_n) / (1.0 - a) * (1.0 - a / a_n), 0.0) ** 0.5
        x_next = a_n**0.5 * x0 + max(1.0 - a_n - sig * sig, 0.0) ** 0.5 * eps
        return x_next + sig * torch.randn_like(x) if sig > 0.0 else x_next

    def restart(
        self,
        x0: torch.Tensor,
        solve: Callable[[torch.Tensor, int], torch.Tensor],
        *,
        t_max: int,
        cycles: int = 2,
        flow: bool = False,
    ) -> torch.Tensor:
        """Restart sampling (Xu et al., NeurIPS 2023) from the end of the trajectory.

        Each cycle forward-noises the clean estimate to ``t_max`` with fresh noise (VP ``q_sample``;
        flow ``(1-s)*x0 + s*z``) and runs ``solve(x_t, t_max)``, a backward ODE solve back to t=0. The
        added noise contracts the error accumulated by the main solve.
        """
        t_max = max(1, min(int(t_max), self.num_timesteps - 1))
        for _ in range(max(1, int(cycles))):
            z = torch.randn_like(x0)
            if flow:
                s = t_max / max(self.num_timesteps - 1, 1)
                x_t = (1.0 - s) * x0 + s * z
            else:
                x_t = self.q_sample(x0, torch.full((x0.shape[0],), t_max, device=x0.device, dtype=torch.long), noise=z)
            x0 = solve(x_t, t_max)
        return x0

    def sample_loop(
        self,
        model,
        shape,
        model_kwargs_cond=None,
        model_kwargs_uncond=None,
        cfg_scale=7.5,
        num_inference_steps=50,
        eta: float | None = None,
        device="cuda",
        dtype=torch.float32,
        x_init=None,
        start_timestep=None,
        cfg_rescale=0.0,
        dynamic_threshold_percentile=0.0,
        dynamic_threshold_type="percentile",
        dynamic_threshold_value=0.0,
        scheduler="ddim",
        timestep_schedule=None,
        solver="ddim",
        karras_rho: float = 7.0,
        cfg_guidance_schedule: str | None = None,
        cfg_guidance_linear_start_multiplier: float = 0.7,
        cfg_guidance_linear_end_multiplier: float = 1.0,
        cfg_guidance_cosine_min_multiplier: float = 0.65,
        cfg_guidance_cosine_max_multiplier: float = 1.0,
        inpaint_mask=None,
        inpaint_x0=None,
        inpaint_noise=None,
        inpaint_freeze_known=False,
        ada_early_exit_delta_threshold: float = 0.0,
        ada_early_exit_patience: int = 0,
        ada_early_exit_min_steps: int = 0,
        sag_blur_sigma: float = 0.0,
        sag_scale: float = 0.0,
        volatile_cfg_boost: float = 0.0,
        volatile_cfg_quantile: float = 0.72,
        volatile_cfg_window: int = 6,
        periodic_alignment_interval: int = 0,
        periodic_alignment_threshold: float = 0.0,
        periodic_alignment_cfg_boost: float = 0.0,
        periodic_alignment_fn: Callable[[int, torch.Tensor], float] | None = None,
        periodic_alignment_rewind_strength: float = 0.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        control_guidance_decay: float = 1.0,
        holy_grail_enable: bool = False,
        holy_grail_cfg_early_ratio: float = 0.72,
        holy_grail_cfg_late_ratio: float = 1.0,
        holy_grail_control_mult: float = 1.0,
        holy_grail_adapter_mult: float = 1.0,
        holy_grail_frontload_control: bool = True,
        holy_grail_late_adapter_boost: float = 1.15,
        holy_grail_cads_strength: float = 0.0,
        holy_grail_cads_min_strength: float = 0.0,
        holy_grail_cads_power: float = 1.0,
        holy_grail_unsharp_sigma: float = 0.0,
        holy_grail_unsharp_amount: float = 0.0,
        holy_grail_clamp_quantile: float = 0.0,
        holy_grail_clamp_floor: float = 1.0,
        flow_matching_sample: bool = False,
        flow_solver: str = "euler",
        flow_schedule: str = "linear",
        flow_karras_rho: float = 7.0,
        flow_shift: float = 1.0,
        flow_shift_ref_hw: tuple[int, int] | None = None,
        flow_init_noise: torch.Tensor | None = None,
        return_intermediate_state: bool = False,
        feature_cache_delta_threshold: float = 0.0,
        feature_cache_max_reuse: int = 2,
        block_cache_threshold: float = 0.0,
        block_cache_recompute_every: int = 4,
        taylor_cache: bool = False,
        taylor_cache_order: int = 1,
        taylor_cache_interval: int = 4,
        apg_parallel_eta: float = -1.0,
        apg_momentum_beta: float = 0.0,
        apg_norm_threshold: float = 0.0,
        fdg_low: float = 1.0,
        fdg_high: float = 1.0,
        cfg_zero_star: bool = False,
        cfg_zero_init_frac: float = 0.04,
        cfg_interval=None,
        tcfg_damping: float = 0.0,
        pag_scale: float = 0.0,
        pag_blocks: str = "auto",
        slg_scale: float = 0.0,
        slg_skip_blocks: str = "",
        slg_start_frac: float = 0.01,
        slg_stop_frac: float = 0.30,
        qsilk_micrograin: float = 0.0,
        dynamic_dit_width: bool = False,
        dynamic_dit_early: float = 0.88,
        dynamic_sdt: bool = False,
        cfg_rejection_rerank: bool = False,
        dbc_separate_cfg: bool = True,
        regional_cfg_plan=None,
        step_noise_scales=None,
        sde_band=None,
        eps_scale: float = 1.0,
        eps_scale_k: float = 0.0,
        attention_layout_plan=None,
        inference_amp: bool = False,
        inference_amp_dtype: torch.dtype | None = None,
    ):
        """
        Sampling loop with CFG for VP diffusion (returns the final x0 prediction) or, with
        ``flow_matching_sample``, rectified flow (integrates dx/ds = v from s=1 to s=0, matching
        ``diffusion.flow_matching`` training; returns the latent at s=0).

        **Timesteps** (VP; ``timestep_schedule`` overrides ``scheduler``): see ``diffusion.inference_timesteps``
        (``karras_rho`` is EDM rho-spacing; ``indices:999,...,0`` is explicit). Flow grids: ``flow_schedule``
        in ``diffusion.solvers.list_flow_schedules``, SD3-shifted by ``flow_shift``; with ``flow_shift_ref_hw``
        (the latent grid ``flow_shift`` was tuned for) the shift scales by ``sqrt(H*W / ref_area)`` for the
        sampled ``shape`` (``diffusion.schedules.sd3_resolution_shift``).

        **Solvers**: VP ``ddim``, ``heun`` (EDM Heun on the eps-ODE), ``dpmpp_2m``, ``dpmpp_2m_sde``,
        ``dpmpp_3m``, ``unipc`` (UniPC-2, one forward per step), ``apex``, ``apex_adaptive``; flow ``euler``,
        ``heun``, ``midpoint``, ``dpmpp_2m``, ``dpmpp_2m_sde``, ``apex``.

        **Stochasticity**: ``eta`` (default 1 for ``dpmpp_2m_sde``, else 0) drives ``ddim``/``heun`` (VP)
        and the ``dpmpp_2m_sde`` solvers. ``sde_band="lo,hi"`` limits noise injection to EDM noise levels
        in [lo, hi] (VP ``sqrt((1-abar)/abar)``, flow ``s/(1-s)``); ``step_noise_scales`` multiplies eta per
        step. Asking for noise from an ODE solver raises ``ValueError``.

        **x0 post-processing** on every model evaluation: epsilon scaling against exposure bias (Ning et al.,
        ICLR 2024; ``eps / (eps_scale + eps_scale_k * t)``, off at 1.0), then dynamic thresholding
        (``dynamic_threshold_*``, see ``_dynamic_threshold``).

        **Guidance** (``utils.generation.guidance_stack``; every enabled term composes): cfg_rescale, apg_*,
        fdg_low/fdg_high, cfg_zero_star, cfg_interval, tcfg_damping, pag_*, slg_*. ``cfg_guidance_schedule``
        (linear, cosine, piecewise, snr) or the holy-grail ramp scales CFG per step. ``sag_*``: blur
        self-attention guidance. ``attention_layout_plan``: Dense Diffusion layout bias on the cross-attention
        of the plan's steps.

        **Loop policies** (VP and flow): volatile CFG; periodic alignment (``periodic_alignment_fn(step, x0)``
        below the threshold boosts CFG and optionally rewinds); AdaGen-style early exit. Inpaint freezing is VP only.

        ``x_init`` with ``start_timestep`` is the already-noised latent at that timestep (img2img).
        ``return_intermediate_state`` (VP): return ``(x_t, x0_pred, t_next)`` to resume with ``x_init=x_t``.
        """
        model_kwargs_cond = model_kwargs_cond or {}
        model_kwargs_uncond = model_kwargs_uncond or {}
        flow = bool(flow_matching_sample)
        do_inpaint = bool(
            inpaint_freeze_known and inpaint_mask is not None and inpaint_x0 is not None and inpaint_noise is not None
        )
        hg_enabled = bool(holy_grail_enable)
        cg_sched = str(cfg_guidance_schedule).strip().lower() if cfg_guidance_schedule else ""
        if hg_enabled and cg_sched:
            raise ValueError(
                "cfg_guidance_schedule cannot be combined with holy_grail_enable; disable one or the other."
            )
        if flow and return_intermediate_state:
            raise ValueError("return_intermediate_state is not supported for flow_matching_sample")
        if flow and do_inpaint:
            raise ValueError(
                "flow_matching_sample is not compatible with inpaint_freeze_known (VP q_sample trajectory). "
                "Use VP sampling or disable structured inpaint."
            )
        sol = canonicalize_flow_solver(flow_solver) if flow else canonicalize_vp_solver(solver)
        stochastic = _FLOW_STOCHASTIC if flow else _VP_STOCHASTIC
        eta_v = (1.0 if sol in _SDE_SOLVERS else 0.0) if eta is None else float(eta)
        sns = [float(v) for v in step_noise_scales] if step_noise_scales else []
        if eta_v > 0.0 and sol not in stochastic:
            raise ValueError(
                f"eta={eta_v} needs a stochastic solver ({', '.join(stochastic)}); {sol!r} is an ODE solver."
            )
        if sns and eta_v <= 0.0:
            raise ValueError(
                f"step_noise_scales (e.g. frontier serendipity curves) scale injected noise: "
                f"use eta > 0 with one of {', '.join(stochastic)}."
            )
        from utils.generation.guidance_stack import GuidanceConfig, GuidanceState, parse_sigma_interval

        band = parse_sigma_interval(sde_band)
        g = GuidanceConfig.build(
            depth=len(getattr(model, "blocks", [])) or 28,
            interval=cfg_interval,
            pag_blocks=pag_blocks if float(pag_scale) else "",
            slg_blocks=slg_skip_blocks if float(slg_scale) else "",
            rescale=float(cfg_rescale),
            apg_eta=float(apg_parallel_eta),
            apg_momentum=float(apg_momentum_beta),
            apg_norm_threshold=float(apg_norm_threshold),
            fdg_low=float(fdg_low),
            fdg_high=float(fdg_high),
            cfg_zero_star=bool(cfg_zero_star),
            cfg_zero_init_frac=float(cfg_zero_init_frac),
            tcfg_damping=float(tcfg_damping),
            pag_scale=float(pag_scale),
            slg_scale=float(slg_scale),
            slg_window=(float(slg_start_frac), float(slg_stop_frac)),
        )
        denom = float(max(self.num_timesteps - 1, 1))
        resume = x_init is not None and start_timestep is not None
        if flow:
            s0 = max(0, min(self.num_timesteps - 1, int(start_timestep))) / denom if resume else 1.0
            init = x_init if resume else flow_init_noise
            x = torch.randn(shape, device=device, dtype=dtype) if init is None else init.to(device=device, dtype=dtype)
            if tuple(x.shape) != tuple(shape):
                raise ValueError(f"initial flow latent shape {tuple(x.shape)} != sample shape {tuple(shape)}")
            from diffusion.schedules import sd3_resolution_shift
            from diffusion.solvers import build_flow_s_grid

            shift = float(flow_shift)
            if flow_shift_ref_hw is not None:
                ref_area = int(flow_shift_ref_hw[0]) * int(flow_shift_ref_hw[1])
                shift = sd3_resolution_shift(int(shape[-2]) * int(shape[-1]), ref_area, shift)
            grid = build_flow_s_grid(
                str(flow_schedule or "linear"),
                max(1, int(num_inference_steps)),
                s_start=s0,
                s_end=0.0,
                karras_rho=float(flow_karras_rho),
                shift=shift,
            )
            n_steps = len(grid) - 1
        else:
            if resume:
                x = x_init.to(device=device, dtype=dtype)
                timesteps = self.set_timesteps_from(num_inference_steps, start_timestep).to(device)
            else:
                x = torch.randn(shape, device=device, dtype=dtype)
                ts_name = str(timestep_schedule if timestep_schedule is not None else scheduler).strip()
                timesteps = self.set_timesteps(
                    num_inference_steps, timestep_schedule=ts_name, karras_rho=float(karras_rho)
                ).to(device)
            n_steps = len(timesteps)
        self._to_device(device)

        feat_cache = None
        if float(feature_cache_delta_threshold) > 0.0:
            from utils.superior.caches import FeatureCacheConfig, FeatureCachePolicy

            feat_cache = FeatureCachePolicy(
                FeatureCacheConfig(
                    delta_threshold=float(feature_cache_delta_threshold),
                    max_reuse_streak=max(1, int(feature_cache_max_reuse)),
                )
            )
        block_cache = _build_block_cache(
            block_cache_threshold,
            block_cache_recompute_every,
            taylor=taylor_cache,
            taylor_order=taylor_cache_order,
            taylor_interval=taylor_cache_interval,
            dbc_separate_cfg=bool(dbc_separate_cfg),
        )
        gstate = GuidanceState()
        probe = None
        if bool(cfg_rejection_rerank):
            from utils.generation.guidance_probe import GuidanceProbe

            probe = GuidanceProbe()
            self._last_guidance_probe = probe
        layout = None
        if attention_layout_plan is not None:
            from frontier.attention import region_masks_from_boxes

            masks = region_masks_from_boxes(attention_layout_plan.boxes, shape[-2], shape[-1], device=device)
            model = layout = _LayoutModel(model, attention_layout_plan, masks, int(shape[0]))
        has_control = "control_image" in model_kwargs_cond
        base_control_scale = model_kwargs_cond.get("control_scale", 1.0)
        if hg_enabled:
            from diffusion.sampling import (
                HolyGrailRecipe,
                apply_condition_noise,
                build_holy_grail_step_plan,
                cads_noise_std,
            )

            hg_recipe = HolyGrailRecipe(
                base_cfg=float(cfg_scale),
                cfg_early_ratio=float(holy_grail_cfg_early_ratio),
                cfg_late_ratio=float(holy_grail_cfg_late_ratio),
                control_base_scale=float(holy_grail_control_mult),
                adapter_base_scale=float(holy_grail_adapter_mult),
                frontload_control=bool(holy_grail_frontload_control),
                late_adapter_boost=float(holy_grail_late_adapter_boost),
            )
        cfg_box = [float(cfg_scale)]

        def scheduled_cfg(i: int, t_batch: torch.Tensor) -> float:
            if not cg_sched:
                return float(cfg_scale)
            if cg_sched == "linear":
                return cfg_scale_linear(
                    float(cfg_scale),
                    i,
                    n_steps,
                    start_multiplier=float(cfg_guidance_linear_start_multiplier),
                    end_multiplier=float(cfg_guidance_linear_end_multiplier),
                )
            if cg_sched in ("cosine", "cosine_ramp"):
                return cfg_scale_cosine_ramp(
                    float(cfg_scale),
                    i,
                    n_steps,
                    min_multiplier=float(cfg_guidance_cosine_min_multiplier),
                    max_multiplier=float(cfg_guidance_cosine_max_multiplier),
                )
            if cg_sched == "piecewise":
                return cfg_scale_piecewise(float(cfg_scale), i, n_steps)
            if cg_sched in ("snr", "snr_aware"):
                if flow:
                    # Flow SNR is ((1-s)/s)^2; pass the abar with the same SNR.
                    s = float(t_batch.reshape(-1)[0]) / denom
                    abar = (1.0 - s) ** 2 / ((1.0 - s) ** 2 + s**2)
                else:
                    abar = self._abar(t_batch)
                return float(cfg_scale) * cfg_scale_snr_aware_multiplier(abar)
            raise ValueError(
                f"Unknown cfg_guidance_schedule {cfg_guidance_schedule!r}; use linear, cosine, piecewise, snr (or unset)."
            )

        def guided(x_in: torch.Tensor, t_batch: torch.Tensor, i: int) -> torch.Tensor:
            if feat_cache is not None and feat_cache.should_skip_forward(x_in):
                return feat_cache.cached_prediction()
            if layout is not None:
                layout.step = i
            mk_c, mk_u = model_kwargs_cond, model_kwargs_uncond
            boost = 1.0 if float(cfg_scale) == 0.0 else cfg_box[0] / float(cfg_scale)
            plan = (
                build_holy_grail_step_plan(recipe=hg_recipe, step_index=i, total_steps=n_steps) if hg_enabled else None
            )
            cs = (float(plan.cfg_scale) if hg_enabled else scheduled_cfg(i, t_batch)) * boost
            if has_control:
                sf = _control_guidance_scale_for_step(
                    1.0,
                    i,
                    n_steps,
                    start=float(control_guidance_start),
                    end=float(control_guidance_end),
                    decay_power=float(control_guidance_decay),
                )
                if hg_enabled:
                    sf *= float(plan.control_scale)
                mk_c = dict(mk_c)
                mk_c["control_scale"] = _scale_control_value(base_control_scale, sf)
                if hg_enabled and "adapter_scale" in mk_c:
                    mk_c["adapter_scale"] = _scale_control_value(mk_c["adapter_scale"], plan.adapter_scale)
                if mk_u and ("control_image" in mk_u or "control_scale" in mk_u):
                    mk_u = dict(mk_u)
                    mk_u["control_scale"] = _scale_control_value(mk_u.get("control_scale", base_control_scale), sf)
                    if hg_enabled and "adapter_scale" in mk_u:
                        mk_u["adapter_scale"] = _scale_control_value(mk_u["adapter_scale"], plan.adapter_scale)
            emb = mk_c.get("encoder_hidden_states")
            if hg_enabled and float(holy_grail_cads_strength) > 0.0 and torch.is_tensor(emb):
                sig = cads_noise_std(
                    progress=1.0 if n_steps <= 1 else i / (n_steps - 1),
                    base_strength=float(holy_grail_cads_strength),
                    min_strength=float(holy_grail_cads_min_strength),
                    power=float(holy_grail_cads_power),
                )
                mk_c = {**mk_c, "encoder_hidden_states": apply_condition_noise(emb, std=sig)}
            use_cfg = cs != 1.0 and bool(mk_u)
            if use_cfg or g.pag or g.slg_active(i, n_steps) or g.zero_init_active(i, n_steps, flow=flow):
                affine, sigma = self._guidance_space(t_batch, x_in.ndim, flow=flow, need_sigma=g.interval is not None)
                if regional_cfg_plan is not None and use_cfg:
                    from utils.generation.regional_box_prompting import regional_cfg_forward

                    out = regional_cfg_forward(
                        model,
                        x_in,
                        t_batch,
                        model_kwargs_cond=mk_c,
                        model_kwargs_uncond=mk_u,
                        plan=regional_cfg_plan,
                        cfg_scale=float(cs),
                        block_cache=block_cache,
                        sample_step=int(i),
                        guidance=g,
                        x0_affine=affine,
                        sigma=sigma,
                    )
                else:
                    from utils.generation.cfg_batched import batched_cfg_forward

                    out = batched_cfg_forward(
                        model,
                        x_in,
                        t_batch,
                        model_kwargs_cond=mk_c,
                        model_kwargs_uncond=mk_u,
                        cfg_scale=float(cs),
                        guidance=g,
                        state=gstate,
                        x0_affine=affine,
                        sigma=sigma,
                        step=int(i),
                        total_steps=n_steps,
                        flow=flow,
                        block_cache=block_cache,
                        guidance_probe=probe,
                        dbc_separate_cfg=bool(dbc_separate_cfg),
                    )
            else:
                # Only pass block_cache when caching is on: class-conditioned DiTs do not accept it.
                kw = mk_c if block_cache is None else {**mk_c, "block_cache": block_cache}
                out = model(x_in, t_batch, **kw)[:, : x_in.shape[1]]
            if bool(dynamic_sdt) or bool(dynamic_dit_width):
                from utils.superior.caches import apply_dynamic_width, apply_sdt_latent_blend

                if bool(dynamic_sdt):
                    out = apply_sdt_latent_blend(x_in, out)
                if bool(dynamic_dit_width):
                    prog = 1.0 if n_steps <= 1 else i / (n_steps - 1)
                    out = apply_dynamic_width(out, prog, early=float(dynamic_dit_early))
            if feat_cache is not None:
                feat_cache.note_fresh(x_in, out)
            return out

        guided = _autocast_prediction(guided, device, inference_amp, inference_amp_dtype)
        sag_on = float(sag_scale) > 0.0 and float(sag_blur_sigma) > 0.0

        def predict(x_in: torch.Tensor, t_batch: torch.Tensor, i: int) -> torch.Tensor:
            out = guided(x_in, t_batch, i)
            if sag_on:
                out_blur = guided(gaussian_blur_latent(x_in, float(sag_blur_sigma)), t_batch, i)
                out = out + float(sag_scale) * (out - out_blur)
            return out

        threshold_on = dynamic_threshold_percentile > 0 or (
            dynamic_threshold_type != "percentile" and dynamic_threshold_value > 0
        )

        def refine_x0(x0: torch.Tensor, x_in: torch.Tensor, alpha: float, t_batch: torch.Tensor) -> torch.Tensor:
            """Epsilon scaling then thresholding of the data prediction of ``x_in = alpha*x0 + sigma*eps``."""
            lam = float(eps_scale) + float(eps_scale_k) * float(t_batch.reshape(-1)[0])
            if lam != 1.0 and alpha > 0.0:
                # x0 re-derived from eps/lam with x fixed: (x - sigma*eps/lam)/alpha.
                x0 = x0 / lam + x_in * ((1.0 - 1.0 / lam) / alpha)
            if threshold_on:
                x0 = self._dynamic_threshold(
                    x0,
                    percentile=dynamic_threshold_percentile,
                    threshold_type=dynamic_threshold_type,
                    threshold_value=dynamic_threshold_value,
                )
            return x0

        def eta_at(i: int, sigma: float) -> float:
            if eta_v <= 0.0 or (band is not None and not band[0] <= sigma <= band[1]):
                return 0.0
            return eta_v * (sns[min(i, len(sns) - 1)] if sns else 1.0)

        vol_deltas: list[float] = []
        exit_streak = [0]

        def after_step(i: int, x_prev: torch.Tensor, x_new: torch.Tensor, x0: torch.Tensor):
            if float(volatile_cfg_boost) > 0.0:
                vol_deltas.append(float((x_new - x_prev).abs().mean()))
                del vol_deltas[: -max(2, int(volatile_cfg_window))]
                if len(vol_deltas) >= 2:
                    thr = float(np.quantile(vol_deltas[:-1], float(np.clip(volatile_cfg_quantile, 0.05, 0.95))))
                    boosted = vol_deltas[-1] > thr
                    cfg_box[0] = float(cfg_scale) * (1.0 + float(volatile_cfg_boost)) if boosted else float(cfg_scale)
            p_int = int(periodic_alignment_interval)
            p_due = p_int > 0 and (i + 1) % p_int == 0 and float(periodic_alignment_cfg_boost) > 0.0
            if p_due and periodic_alignment_fn is not None:
                if float(periodic_alignment_fn(int(i), x0)) < float(periodic_alignment_threshold):
                    cfg_box[0] *= 1.0 + float(periodic_alignment_cfg_boost)
                    rw = min(1.0, max(0.0, float(periodic_alignment_rewind_strength)))
                    x_new = (1.0 - rw) * x_new + rw * x_prev
            stop = False
            if float(ada_early_exit_delta_threshold) > 0.0 and int(ada_early_exit_patience) > 0:
                if i >= int(ada_early_exit_min_steps):
                    small = float((x_new - x_prev).abs().mean()) < float(ada_early_exit_delta_threshold)
                    exit_streak[0] = exit_streak[0] + 1 if small else 0
                    stop = exit_streak[0] >= int(ada_early_exit_patience)
            return x_new, stop

        loop_kw = dict(predict=predict, refine_x0=refine_x0, solver=sol, eta_at=eta_at, after_step=after_step)
        if flow:
            out = self._flow_loop(x, grid, **loop_kw)
        else:
            inpaint = None
            if do_inpaint:
                inpaint = tuple(v.to(device=device, dtype=dtype) for v in (inpaint_mask, inpaint_x0, inpaint_noise))
            x_t, out, t_next = self._vp_loop(x, timesteps, inpaint=inpaint, karras_rho=float(karras_rho), **loop_kw)
        if hg_enabled and float(holy_grail_unsharp_amount) > 0.0 and float(holy_grail_unsharp_sigma) > 0.0:
            out = unsharp_mask_latent(
                out, sigma=float(holy_grail_unsharp_sigma), amount=float(holy_grail_unsharp_amount)
            )
        if hg_enabled and float(holy_grail_clamp_quantile) > 0.0:
            out = dynamic_threshold(out, float(holy_grail_clamp_quantile), float(holy_grail_clamp_floor), rescale=False)
        if float(qsilk_micrograin) > 0.0:
            from utils.generation.latent_filters import qsilk_micrograin_stabilize

            out = qsilk_micrograin_stabilize(out, detail_amount=float(qsilk_micrograin))
        if return_intermediate_state:
            return x_t, out, int(t_next)
        return out

    def _vp_loop(self, x, timesteps, *, predict, refine_x0, solver, eta_at, after_step, inpaint, karras_rho):
        """VP sampling over discrete ``timesteps``; returns ``(x_t, x0_pred, t_next)``."""
        from diffusion.solvers import (
            ApexController,
            SolverState,
            redistribute_remaining_vp_indices,
            vp_apex_update,
            vp_dpmpp_update,
            vp_unipc_correct,
        )

        B = x.shape[0]
        state = SolverState(max_order=3)
        apex_ctrl = ApexController()
        unipc_prev = None
        x0 = None
        t_next_val = 0

        def denoise(x_in: torch.Tensor, t: torch.Tensor, i: int) -> torch.Tensor:
            x0_raw, _ = self._predict_x0_and_noise(predict(x_in, t, i), x_in, t)
            return refine_x0(x0_raw, x_in, self._abar(t) ** 0.5, t)

        for i in range(len(timesteps)):
            t = timesteps[i].expand(B)
            last = i + 1 == len(timesteps)
            t_next = torch.zeros_like(t) if last else timesteps[i + 1].expand(B)
            t_next_val = int(t_next[0])
            x0 = denoise(x, t, i)
            if unipc_prev is not None:
                x = vp_unipc_correct(
                    x_prev=unipc_prev[0],
                    x0_new=x0,
                    alpha_bar_prev=self._abar(unipc_prev[1]),
                    alpha_bar_cur=self._abar(t),
                    state=state,
                    order=unipc_prev[2],
                )
            if last:
                break
            x_prev = x
            a, a_n = self._abar(t), self._abar(t_next)
            eta_i = eta_at(i, ((1.0 - a) / a) ** 0.5)
            if solver in (_SOLVER_DPMPP_2M, _SOLVER_DPMPP_2M_SDE, _SOLVER_DPMPP_3M, _SOLVER_UNIPC):
                order = 3 if solver == _SOLVER_DPMPP_3M else 2
                x, state = vp_dpmpp_update(
                    x=x, x0_pred=x0, alpha_bar_cur=a, alpha_bar_next=a_n, state=state, order=order, eta=eta_i
                )
                if solver == _SOLVER_UNIPC:
                    unipc_prev = (x_prev, t, min(2, state.order))
            elif solver in (_SOLVER_APEX, _SOLVER_APEX_ADAPTIVE):
                res = vp_apex_update(
                    x=x,
                    x0_pred=x0,
                    alpha_bar_cur=a,
                    alpha_bar_next=a_n,
                    state=state,
                    controller=apex_ctrl,
                    progress=i / max(len(timesteps) - 1, 1),
                )
                x, state = res.x_next, res.state
                rem = len(timesteps) - (i + 2)
                if solver == _SOLVER_APEX_ADAPTIVE and rem > 0:
                    # Same NFE budget; re-space the knots still ahead by measured difficulty.
                    tail = redistribute_remaining_vp_indices(
                        t_next_val,
                        rem,
                        self.num_timesteps,
                        self.alpha_cumprod.detach().cpu().numpy(),
                        difficulty=apex_ctrl.last_difficulty,
                        rho=karras_rho,
                    )
                    if len(tail) == rem:
                        tail_t = torch.as_tensor(tail, device=timesteps.device, dtype=timesteps.dtype)
                        timesteps = torch.cat([timesteps[: i + 2], tail_t])
            elif solver == _SOLVER_HEUN:
                x_e = self._ddim_step(x, t, t_next, x0)
                x0_e = denoise(x_e, t_next, i + 1)
                eps = 0.5 * ((x - a**0.5 * x0) / (1.0 - a) ** 0.5 + (x_e - a_n**0.5 * x0_e) / (1.0 - a_n) ** 0.5)
                x0 = (x - (1.0 - a) ** 0.5 * eps) / a**0.5
                x = self._ddim_step(x, t, t_next, x0, eta=eta_i)
            else:
                x = self._ddim_step(x, t, t_next, x0, eta=eta_i)
            x, stop = after_step(i, x_prev, x, x0)
            if inpaint is not None:
                mask, x0_known, noise_known = inpaint
                x = mask * x + (1.0 - mask) * self.q_sample(x0_known, t_next, noise=noise_known)
            if stop:
                break
        if inpaint is not None:
            x0 = inpaint[0] * x0 + (1.0 - inpaint[0]) * inpaint[1]
        return x, x0, t_next_val

    def _flow_loop(self, x, grid, *, predict, refine_x0, solver, eta_at, after_step):
        """Rectified-flow sampling over the s-grid (``grid[0]`` -> 0); returns the latent at s=0."""
        from diffusion.solvers import ApexController, SolverState, flow_apex_update, flow_dpmpp_2m_update

        B = x.shape[0]
        T = int(self.num_timesteps)
        n = len(grid) - 1
        state = SolverState(max_order=3)
        apex_ctrl = ApexController()

        def velocity(x_in: torch.Tensor, s: float, i: int) -> torch.Tensor:
            t_b = torch.full((B,), max(0, min(T - 1, round(s * (T - 1)))), device=x_in.device, dtype=torch.long)
            v = predict(x_in, t_b, i)
            x0 = x_in - s * v
            x0_ref = refine_x0(x0, x_in, 1.0 - s, t_b)
            return v if x0_ref is x0 or s <= 0.0 else (x_in - x0_ref) / s

        for i in range(n):
            s, s_n = float(grid[i]), float(grid[i + 1])
            ds = s_n - s
            x_prev = x
            v = velocity(x, s, i)
            x0 = x - s * v
            if solver == "heun" and s_n > 0.0:
                v2 = velocity(x + ds * v, s_n, min(i + 1, n - 1))
                x = x + 0.5 * ds * (v + v2)
            elif solver == "midpoint":
                s_m = 0.5 * (s + s_n)
                x = x + ds * velocity(x + (s_m - s) * v, s_m, i)
            elif solver in ("dpmpp_2m", "dpmpp_2m_sde"):
                eta_i = eta_at(i, s / max(1.0 - s, 1e-12))
                x, state = flow_dpmpp_2m_update(x=x, velocity=v, s_cur=s, s_next=s_n, state=state, eta=eta_i)
            elif solver == "apex":
                res = flow_apex_update(
                    x=x, velocity=v, s_cur=s, s_next=s_n, state=state, controller=apex_ctrl, progress=i / max(n - 1, 1)
                )
                x, state = res.x_next, res.state
            else:
                # Euler, and Heun's last step: EDM (Alg. 2) skips the correction when sigma_next = 0.
                x = x + ds * v
            x, stop = after_step(i, x_prev, x, x0)
            if stop:
                return x0
        return x


class _LayoutModel:
    """Model proxy that turns on the Dense Diffusion layout bias (``models.layout_attn_context``) per call.

    The sampler sets ``step``; cross-attention biases the first ``batch`` rows (the conditional branch).
    """

    def __init__(self, model, plan, region_masks: torch.Tensor, batch: int):
        self.model, self.plan, self.region_masks, self.batch, self.step = model, plan, region_masks, batch, 0

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    def __call__(self, x, t, **kwargs):
        from models.layout_attn_context import clear_layout_attn, set_layout_attn

        set_layout_attn(
            plan=self.plan,
            region_masks=self.region_masks,
            step_index=self.step,
            cond_batch_size=self.batch,
            num_patch_tokens=(x.shape[-2] * x.shape[-1]) // int(getattr(self.model, "patch_size", 1)) ** 2,
        )
        try:
            return self.model(x, t, **kwargs)
        finally:
            clear_layout_attn()
