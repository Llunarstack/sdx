# Gaussian diffusion for DiT: SD/SDXL-style features (offset noise, min-SNR, ε/v/x0-pred, DDIM, CFG).
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .inference_timesteps import build_inference_timesteps
from .losses.timestep_loss_weight import get_timestep_loss_weight
from .respace import space_timesteps
from .sampling_utils import norm_thresholding, spatial_norm_thresholding
from .schedules import get_beta_schedule
from .spectral_sfp import spectral_sfp_per_sample_loss

INFERENCE_SOLVERS = ("ddim", "heun")
FLOW_INFERENCE_SOLVERS = ("euler", "heun")


def _control_guidance_scale_for_step(
    base_scale: float,
    step_index: int,
    total_steps: int,
    *,
    start: float = 0.0,
    end: float = 1.0,
    decay_power: float = 1.0,
) -> float:
    """Compute step-wise control scale over denoising progress."""
    b = float(base_scale)
    if b <= 0.0 or total_steps <= 0:
        return 0.0
    s = float(max(0.0, min(1.0, start)))
    e = float(max(0.0, min(1.0, end)))
    if e < s:
        s, e = e, s
    if e <= s:
        return b if step_index <= 0 else 0.0
    p = 0.0 if total_steps <= 1 else float(step_index) / float(max(total_steps - 1, 1))
    if p < s or p > e:
        return 0.0
    u = (p - s) / max(1e-8, e - s)
    pow_v = max(1e-6, float(decay_power))
    w = (1.0 - u) ** pow_v
    return b * max(0.0, min(1.0, w))


def _scale_control_value(control_scale, factor: float):
    f = float(factor)
    if torch.is_tensor(control_scale):
        return control_scale * f
    if isinstance(control_scale, (list, tuple)):
        return [float(x) * f for x in control_scale]
    try:
        return float(control_scale) * f
    except Exception:
        return control_scale


def create_diffusion(
    timestep_respacing: str = "",
    num_timesteps: int = 1000,
    beta_schedule: str = "linear",
    loss_type: str = "mse",
    prediction_type: str = "epsilon",
):
    """Create GaussianDiffusion. prediction_type: 'epsilon' (default), 'v' (velocity, SD2-style),
    or 'x0' (direct clean-latent prediction; use same type at train and sample time).
    timestep_respacing: "" = use all steps; int string (e.g. "50") = that many evenly spaced;
    "ddim50" = DDIM striding for 50 steps; "10,15,20" = section-based step counts."""
    use_timesteps = None
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
        alpha = 1.0 - beta
        alpha_cumprod = np.cumprod(alpha)
        alpha_cumprod_prev = np.concatenate([[1.0], alpha_cumprod[:-1]])
        self.beta = torch.from_numpy(beta).float()
        self.alpha_cumprod = torch.from_numpy(alpha_cumprod).float()
        self.alpha_cumprod_prev = torch.from_numpy(alpha_cumprod_prev).float()
        self.sqrt_alpha_cumprod = torch.from_numpy(np.sqrt(alpha_cumprod)).float()
        self.sqrt_one_minus_alpha_cumprod = torch.from_numpy(np.sqrt(1.0 - alpha_cumprod)).float()
        # For min-SNR weighting: SNR(t) = alpha_cumprod / (1 - alpha_cumprod)
        snr = alpha_cumprod / (1.0 - alpha_cumprod + 1e-8)
        self.snr = torch.from_numpy(snr).float()

    def _to_device(self, device):
        self.beta = self.beta.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        if hasattr(self, "snr"):
            self.snr = self.snr.to(device)

    def q_sample(self, x_start, t, noise=None, noise_offset=0.0):
        """Forward diffusion: x_0 -> x_t. noise_offset (SD-style): shift noise for better light/dark balance."""
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device, dtype=x_start.dtype)
        if noise_offset > 0:
            noise = noise + noise_offset * torch.randn(
                x_start.shape[0], 1, 1, 1, device=noise.device, dtype=noise.dtype
            ).expand_as(noise)
        self._to_device(x_start.device)
        sqrt_alpha = self.sqrt_alpha_cumprod.to(x_start.device)[t][(...,) + (None,) * (x_start.ndim - 1)]
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod.to(x_start.device)[t][(...,) + (None,) * (x_start.ndim - 1)]
        return sqrt_alpha * x_start + sqrt_one_minus * noise

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
        Used for Diffusion-DPO and other per-example objectives.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B = x_start.shape[0]
        device = x_start.device
        if refinement_prob > 0 and refinement_max_t > 0 and torch.rand(1, device=device).item() < refinement_prob:
            t = torch.randint(0, min(refinement_max_t, self.num_timesteps), (B,), device=device, dtype=t.dtype)
        if noise is None:
            noise = torch.randn_like(x_start, device=device, dtype=x_start.dtype)
        x_t = self.q_sample(x_start, t, noise=noise, noise_offset=noise_offset)
        model_out = model(x_t, t, **model_kwargs)
        if model_out.shape != x_start.shape and model_out.shape[1] > x_start.shape[1]:
            model_out = model_out[:, : x_start.shape[1]]
        self._to_device(device)
        sqrt_alpha = self.sqrt_alpha_cumprod.to(device)[t][(...,) + (None,) * (x_start.ndim - 1)]
        sqrt_one = self.sqrt_one_minus_alpha_cumprod.to(device)[t][(...,) + (None,) * (x_start.ndim - 1)]
        if self.prediction_type == "v":
            target = sqrt_alpha * noise - sqrt_one * x_start
        elif self.prediction_type == "x0":
            target = x_start
        else:
            target = noise
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
        snr_t = self.snr.to(device)[t] if hasattr(self, "snr") else None
        alpha = self.alpha_cumprod.to(device)[t]
        weight = get_timestep_loss_weight(
            loss_weighting,
            snr=snr_t,
            alpha_cumprod=alpha,
            min_snr_gamma=min_snr_gamma,
            loss_weighting_sigma_data=loss_weighting_sigma_data,
        )
        return loss * weight

    def training_losses(
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
        sample_weights=None,
        loss_weighting="min_snr",
        loss_weighting_sigma_data=0.5,
        use_spectral_sfp_loss=False,
        spectral_sfp_low_sigma=0.22,
        spectral_sfp_high_sigma=0.22,
        spectral_sfp_tau_power=1.0,
    ):
        """
        Compute training loss. SD/SDXL-style: offset noise; epsilon, v-, or x0-prediction.
        loss_weighting: "min_snr" (default) | "min_snr_soft" | "unit" | "edm" | "v" | "eps".
        sample_weights: (B,) optional; per-sample loss weight (e.g. aesthetic score).
        use_spectral_sfp_loss: if True, replace spatial MSE with frequency-weighted FFT loss
        (see diffusion/spectral_sfp.py); **not** used with MDM masked loss in train.py.
        """
        if model_kwargs is None:
            model_kwargs = {}
        device = x_start.device
        loss = self.per_sample_training_losses(
            model,
            x_start,
            t,
            model_kwargs=model_kwargs,
            noise=noise,
            refinement_prob=refinement_prob,
            refinement_max_t=refinement_max_t,
            noise_offset=noise_offset,
            min_snr_gamma=min_snr_gamma,
            loss_weighting=loss_weighting,
            loss_weighting_sigma_data=loss_weighting_sigma_data,
            use_spectral_sfp_loss=use_spectral_sfp_loss,
            spectral_sfp_low_sigma=spectral_sfp_low_sigma,
            spectral_sfp_high_sigma=spectral_sfp_high_sigma,
            spectral_sfp_tau_power=spectral_sfp_tau_power,
        )
        if sample_weights is not None and sample_weights.shape[0] == loss.shape[0]:
            sample_weights = sample_weights.to(device, dtype=loss.dtype)
            loss = (loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
        else:
            loss = loss.mean()
        return {"loss": loss}

    def _predict_x0_and_noise(self, model_out, x_t, t):
        """Convert model output (epsilon, v, or x0) to x_0 and implied noise."""
        self._to_device(x_t.device)
        alpha = self.alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        sigma = self.sqrt_one_minus_alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        sqrt_alpha = self.sqrt_alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        if self.prediction_type == "v":
            pred_noise = (sqrt_alpha * model_out + sigma * x_t) / (sigma + 1e-8)
            x_0_pred = sqrt_alpha * x_t - sigma * model_out
        elif self.prediction_type == "x0":
            x_0_pred = model_out
            pred_noise = (x_t - sqrt_alpha * x_0_pred) / (sigma + 1e-8)
        else:
            pred_noise = model_out
            x_0_pred = (x_t - sigma * pred_noise) / (alpha.sqrt() + 1e-8)
        return x_0_pred, pred_noise

    def p_step(self, model, x_t, t, model_kwargs=None):
        """Single DDPM denoising step: x_t -> x_{t-1}. Returns (x_0_pred, x_prev)."""
        if model_kwargs is None:
            model_kwargs = {}
        out = model(x_t, t, **model_kwargs)
        if out.shape != x_t.shape and out.shape[1] > x_t.shape[1]:
            out = out[:, : x_t.shape[1]]
        x_0_pred, pred_noise = self._predict_x0_and_noise(out, x_t, t)
        self._to_device(x_t.device)
        sqrt_alpha = self.sqrt_alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        sigma = self.sqrt_one_minus_alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        noise_impl = (x_t - sqrt_alpha * x_0_pred) / (sigma + 1e-8)
        t_prev = (t - 1).clamp(min=0)
        alpha_prev = self.alpha_cumprod.to(x_t.device)[t_prev][(...,) + (None,) * (x_t.ndim - 1)]
        sigma_prev = self.sqrt_one_minus_alpha_cumprod.to(x_t.device)[t_prev][(...,) + (None,) * (x_t.ndim - 1)]
        x_prev = alpha_prev.sqrt() * x_0_pred + sigma_prev * noise_impl
        return x_0_pred, x_prev

    def set_timesteps(
        self,
        num_inference_steps: int,
        timestep_schedule: str = "ddim",
        *,
        scheduler: Optional[str] = None,
        karras_rho: float = 7.0,
    ):
        """
        Set ``self.timesteps`` (high noise → low). ``timestep_schedule`` selects the index path;
        ``scheduler`` is a legacy alias for ``timestep_schedule`` when provided.

        Registered schedules: see ``diffusion.inference_timesteps.list_timestep_schedules()`` —
        includes ``ddim``, ``euler``, ``karras_rho``, ``snr_uniform``, ``quad_cosine``.
        If ``timestep_respacing_str`` was set at construction (e.g. ``ddim50``) and the
        requested name is ``ddim``, respacing from ``space_timesteps`` is used (same as
        legacy behavior). Other schedule names ignore respacing (e.g. ``euler`` with respacing
        still uses the euler index path).
        """
        name = str(scheduler if scheduler is not None else timestep_schedule).lower().strip()
        ts_resp = getattr(self, "timestep_respacing_str", None)
        if ts_resp and name == "ddim":
            steps = space_timesteps(self.num_timesteps, ts_resp)
            self.timesteps = torch.from_numpy(steps[::-1].copy().astype(np.int64))
            return self.timesteps
        ac = self.alpha_cumprod.detach().cpu().numpy()
        steps = build_inference_timesteps(
            name,
            self.num_timesteps,
            num_inference_steps,
            ac,
            karras_rho=float(karras_rho),
        )
        self.timesteps = torch.from_numpy(np.asarray(steps, dtype=np.int64).copy())
        return self.timesteps

    def set_timesteps_from(self, num_inference_steps: int, t_start: int):
        """Img2img / from-z: timesteps from t_start down to 0 (FLUX/SD-style)."""
        t_start = min(max(0, int(t_start)), self.num_timesteps - 1)
        # High noise → low: first index must be near t_start (matches img2img / hires second pass).
        steps = np.linspace(t_start, 0, num_inference_steps, endpoint=True)
        self.timesteps = torch.from_numpy(steps.astype(np.int64).copy())
        return self.timesteps

    def _dynamic_threshold(
        self,
        x: torch.Tensor,
        percentile: float = 0.0,
        threshold_type: str = "percentile",
        threshold_value: float = 0.0,
    ) -> torch.Tensor:
        """Apply dynamic thresholding to x0 to reduce oversaturation. Types: percentile (quantile clamp), norm, spatial_norm (ControlNet-style)."""
        if threshold_type == "norm" and threshold_value > 0:
            return norm_thresholding(x, threshold_value)
        if threshold_type == "spatial_norm" and threshold_value > 0:
            return spatial_norm_thresholding(x, threshold_value)
        if threshold_type == "percentile" and percentile > 0 and percentile < 100:
            b = x.shape[0]
            x_flat = x.reshape(b, -1)
            q = torch.quantile(x_flat.float(), percentile / 100.0, dim=1, keepdim=True)
            q = q.reshape(b, *([1] * (x.ndim - 1))).to(x.dtype)
            return torch.where(x > q, q, torch.where(x < -q, -q, x))
        return x

    def step_with_pred(
        self,
        x_t,
        t,
        t_next,
        pred_out,
        eta=0.0,
        dynamic_threshold_percentile=0.0,
        dynamic_threshold_type="percentile",
        dynamic_threshold_value=0.0,
        # PBFM (physics/perceptual guidance) - lightweight edge/high-pass drift.
        pbfm_edge_boost: float = 0.0,
        pbfm_edge_kernel: int = 3,
    ):
        """DDIM-style step using pre-computed model output (for CFG). pred_out = ε, v, or x0 per prediction_type."""
        x_0_pred, _ = self._predict_x0_and_noise(pred_out, x_t, t)
        # PBFM: add a high-pass ("edge") drift to x_0_pred before DDIM update.
        # This is a heuristic implementation intended to reduce oversmoothing.
        if pbfm_edge_boost and float(pbfm_edge_boost) != 0.0:
            k = int(pbfm_edge_kernel)
            k = max(3, k | 1)  # odd >=3
            hp = x_0_pred - torch.nn.functional.avg_pool2d(x_0_pred, kernel_size=k, stride=1, padding=k // 2)
            x_0_pred = x_0_pred + float(pbfm_edge_boost) * hp
        if dynamic_threshold_percentile > 0 or (dynamic_threshold_type != "percentile" and dynamic_threshold_value > 0):
            x_0_pred = self._dynamic_threshold(
                x_0_pred,
                percentile=dynamic_threshold_percentile,
                threshold_type=dynamic_threshold_type,
                threshold_value=dynamic_threshold_value,
            )
        self._to_device(x_t.device)
        alpha = self.alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        alpha_next = self.alpha_cumprod.to(x_t.device)[t_next][(...,) + (None,) * (x_t.ndim - 1)]
        sigma = eta * ((1 - alpha_next) / (1 - alpha + 1e-8) * (1 - alpha / (alpha_next + 1e-8))).clamp(0).sqrt()
        dir_xt = (
            (1 - alpha_next - sigma**2).clamp(0).sqrt() * (x_t - alpha.sqrt() * x_0_pred) / (1 - alpha + 1e-8).sqrt()
        )
        x_next = alpha_next.sqrt() * x_0_pred + dir_xt
        if eta > 0:
            x_next = x_next + sigma * torch.randn_like(x_t, device=x_t.device, dtype=x_t.dtype)
        return x_next, x_0_pred

    def _sample_loop_flow_matching(
        self,
        model,
        shape,
        model_kwargs_cond=None,
        model_kwargs_uncond=None,
        *,
        cfg_scale: float = 7.5,
        cfg_rescale: float = 0.0,
        num_inference_steps: int = 50,
        flow_solver: str = "euler",
        device="cuda",
        dtype=torch.float32,
        x_init=None,
        start_timestep=None,
        speculative_draft_cfg_scale: float = 0.0,
        speculative_close_thresh: float = 0.0,
        speculative_blend: float = 0.35,
        dynamic_threshold_percentile: float = 0.0,
        dynamic_threshold_type: str = "percentile",
        dynamic_threshold_value: float = 0.0,
        flow_init_noise: Optional[torch.Tensor] = None,
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
    ) -> torch.Tensor:
        """
        Sample to match ``diffusion.flow_matching.flow_matching_per_sample_losses`` training:

        - Interpolation ``x = (1-s) x_0 + s \\epsilon`` with ``s \\in [0,1]``, ``t = \\mathrm{round}(s(T-1))``.
        - Model predicts velocity ``v \\approx \\epsilon - x_0``; integrate ``\\mathrm{d}x/\\mathrm{d}s = v``
          from ``s=1`` (noise) to ``s=0`` (clean).

        Returns the final latent (estimated ``x_0``). CFG and optional speculative CFG match ``sample_loop``.
        """
        model_kwargs_cond = model_kwargs_cond or {}
        model_kwargs_uncond = model_kwargs_uncond or {}
        T = int(self.num_timesteps)
        denom = max(T - 1, 1)
        n = max(1, int(num_inference_steps))
        B = int(shape[0])
        fs = str(flow_solver).lower().strip()

        if x_init is not None and start_timestep is not None:
            t0 = int(start_timestep)
            t0 = max(0, min(T - 1, t0))
            s0 = float(t0) / float(denom)
            x0_hint = x_init.to(device=device, dtype=dtype)
            z = torch.randn(shape, device=device, dtype=dtype)
            if x0_hint.shape != z.shape:
                raise ValueError("flow_matching_sample: x_init must match sample shape for img2img-style start.")
            x = (1.0 - s0) * x0_hint + s0 * z
            s_vals = torch.linspace(s0, 0.0, n + 1, device=device, dtype=torch.float64)
        else:
            if flow_init_noise is not None:
                x = flow_init_noise.to(device=device, dtype=dtype)
                if tuple(x.shape) != tuple(shape):
                    raise ValueError(f"flow_init_noise shape {tuple(x.shape)} != sample shape {tuple(shape)}")
            else:
                x = torch.randn(shape, device=device, dtype=dtype)
            s_vals = torch.linspace(1.0, 0.0, n + 1, device=device, dtype=torch.float64)

        cfg_box = [float(cfg_scale)]
        spec_draft = float(speculative_draft_cfg_scale)
        spec_thr = float(speculative_close_thresh)
        spec_blend = float(speculative_blend)
        use_spec_cfg = spec_draft > 0.0 and model_kwargs_uncond is not None
        has_control = "control_image" in model_kwargs_cond
        base_control_scale = model_kwargs_cond.get("control_scale", 1.0)
        hg_enabled = bool(holy_grail_enable)
        if hg_enabled:
            from diffusion.holy_grail import HolyGrailRecipe, apply_condition_noise, build_holy_grail_step_plan, cads_noise_std

            hg_recipe = HolyGrailRecipe(
                base_cfg=float(cfg_scale),
                cfg_early_ratio=float(holy_grail_cfg_early_ratio),
                cfg_late_ratio=float(holy_grail_cfg_late_ratio),
                control_base_scale=float(holy_grail_control_mult),
                adapter_base_scale=float(holy_grail_adapter_mult),
                frontload_control=bool(holy_grail_frontload_control),
                late_adapter_boost=float(holy_grail_late_adapter_boost),
            )

        def _model_prediction(x_in: torch.Tensor, t_batch: torch.Tensor, step_idx: int) -> torch.Tensor:
            cs = cfg_box[0]
            mk_c = model_kwargs_cond
            mk_u = model_kwargs_uncond
            p = 1.0 if n <= 1 else float(step_idx) / float(max(n - 1, 1))
            if hg_enabled:
                plan = build_holy_grail_step_plan(recipe=hg_recipe, step_index=step_idx, total_steps=n)
                dyn = 1.0 if float(cfg_scale) == 0.0 else float(cfg_box[0]) / float(cfg_scale)
                cs = float(plan.cfg_scale) * dyn
            if has_control:
                sf = _control_guidance_scale_for_step(
                    1.0,
                    step_idx,
                    n,
                    start=float(control_guidance_start),
                    end=float(control_guidance_end),
                    decay_power=float(control_guidance_decay),
                )
                mk_c = dict(model_kwargs_cond)
                if hg_enabled:
                    sf = sf * float(plan.control_scale)
                mk_c["control_scale"] = _scale_control_value(base_control_scale, sf)
                if hg_enabled and "adapter_scale" in mk_c:
                    mk_c["adapter_scale"] = _scale_control_value(mk_c.get("adapter_scale", 1.0), plan.adapter_scale)
                if mk_u and ("control_image" in mk_u or "control_scale" in mk_u):
                    mk_u = dict(mk_u)
                    mk_u["control_scale"] = _scale_control_value(mk_u.get("control_scale", base_control_scale), sf)
                    if hg_enabled and "adapter_scale" in mk_u:
                        mk_u["adapter_scale"] = _scale_control_value(mk_u.get("adapter_scale", 1.0), plan.adapter_scale)
            if hg_enabled and float(holy_grail_cads_strength) > 0.0 and "encoder_hidden_states" in mk_c:
                emb = mk_c.get("encoder_hidden_states", None)
                if torch.is_tensor(emb):
                    mk_c = dict(mk_c)
                    sig = cads_noise_std(
                        progress=p,
                        base_strength=float(holy_grail_cads_strength),
                        min_strength=float(holy_grail_cads_min_strength),
                        power=float(holy_grail_cads_power),
                    )
                    mk_c["encoder_hidden_states"] = apply_condition_noise(emb, std=sig)
            if use_spec_cfg and cs != 1.0 and cs > 0 and mk_u:
                from utils.generation.speculative_denoise import speculative_cfg_prediction

                return speculative_cfg_prediction(
                    model,
                    x_in,
                    t_batch,
                    model_kwargs_cond=mk_c,
                    model_kwargs_uncond=mk_u,
                    cfg_scale=cs,
                    draft_cfg_scale=spec_draft,
                    cfg_rescale=float(cfg_rescale),
                    close_thresh=spec_thr,
                    blend_on_close=spec_blend,
                )
            if cs != 1.0 and cs > 0 and mk_u:
                out_cond = model(x_in, t_batch, **mk_c)
                out_uncond = model(x_in, t_batch, **mk_u)
                if out_cond.shape != x_in.shape and out_cond.shape[1] > x_in.shape[1]:
                    out_cond, out_uncond = out_cond[:, : x_in.shape[1]], out_uncond[:, : x_in.shape[1]]
                delta = out_cond - out_uncond
                if cfg_rescale > 0:
                    sig = delta.std() + 1e-8
                    scale = max(sig / cfg_rescale, 1.0)
                    delta = delta / scale
                return out_uncond + cs * delta
            o = model(x_in, t_batch, **mk_c)
            if o.shape != x_in.shape and o.shape[1] > x_in.shape[1]:
                o = o[:, : x_in.shape[1]]
            return o

        def _t_batch_from_s(s: float) -> torch.Tensor:
            ti = int(round(float(s) * float(denom)))
            ti = max(0, min(T - 1, ti))
            return torch.full((B,), ti, device=device, dtype=torch.long)

        for i in range(n):
            s_cur = float(s_vals[i].item())
            s_next = float(s_vals[i + 1].item())
            ds = s_next - s_cur
            t_b = _t_batch_from_s(s_cur)
            v1 = _model_prediction(x, t_b, i)
            if fs == "heun":
                x_pred = x + v1 * ds
                t_b2 = _t_batch_from_s(s_next)
                v2 = _model_prediction(x_pred, t_b2, min(i + 1, n - 1))
                x = x + 0.5 * (v1 + v2) * ds
            else:
                x = x + v1 * ds

        if dynamic_threshold_percentile > 0 or (
            dynamic_threshold_type != "percentile" and dynamic_threshold_value > 0
        ):
            x = self._dynamic_threshold(
                x,
                percentile=dynamic_threshold_percentile,
                threshold_type=dynamic_threshold_type,
                threshold_value=dynamic_threshold_value,
            )
        if hg_enabled:
            if float(holy_grail_unsharp_amount) > 0.0 and float(holy_grail_unsharp_sigma) > 0.0:
                from diffusion.holy_grail import unsharp_mask_latent

                x = unsharp_mask_latent(
                    x,
                    sigma=float(holy_grail_unsharp_sigma),
                    amount=float(holy_grail_unsharp_amount),
                )
            if float(holy_grail_clamp_quantile) > 0.0:
                from diffusion.holy_grail import dynamic_percentile_clamp

                x = dynamic_percentile_clamp(
                    x,
                    quantile=float(holy_grail_clamp_quantile),
                    floor=float(holy_grail_clamp_floor),
                )
        return x

    def sample_loop(
        self,
        model,
        shape,
        model_kwargs_cond=None,
        model_kwargs_uncond=None,
        cfg_scale=7.5,
        num_inference_steps=50,
        eta=0.0,
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
        # Optional: masked/inpainting-style freezing of known regions.
        # This approximates MDM-style "fill the blanks" behavior at inference:
        # after each denoise step, unmasked latents are reset to q_sample(x0_known, t).
        inpaint_mask=None,
        inpaint_x0=None,
        inpaint_noise=None,
        inpaint_freeze_known=False,
        # AdaGen-style early exit: stop when latent updates get small.
        ada_early_exit_delta_threshold: float = 0.0,
        ada_early_exit_patience: int = 0,
        ada_early_exit_min_steps: int = 0,
        # PBFM guidance
        pbfm_edge_boost: float = 0.0,
        pbfm_edge_kernel: int = 3,
        # Blur-based self-attention guidance (heuristic): extra model forward on Gaussian-blurred x;
        # amplifies high-frequency structure vs a smoothed baseline. Costs ~2× forwards when enabled.
        sag_blur_sigma: float = 0.0,
        sag_scale: float = 0.0,
        volatile_cfg_boost: float = 0.0,
        volatile_cfg_quantile: float = 0.72,
        volatile_cfg_window: int = 6,
        # Optional: decode x0_pred periodically (caller supplies CLIP/VLM fn) and boost CFG when score is low.
        periodic_alignment_interval: int = 0,
        periodic_alignment_threshold: float = 0.0,
        periodic_alignment_cfg_boost: float = 0.0,
        periodic_alignment_fn: Optional[Callable[[int, torch.Tensor], float]] = None,
        # Speculative CFG: second forward at draft_cfg_scale; blend if |full-draft| mean < close_thresh.
        speculative_draft_cfg_scale: float = 0.0,
        speculative_close_thresh: float = 0.0,
        speculative_blend: float = 0.35,
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
        # Rectified-flow path (matches diffusion.flow_matching training); mutually exclusive with VP DDIM updates below.
        flow_matching_sample: bool = False,
        flow_solver: str = "euler",
        flow_init_noise: Optional[torch.Tensor] = None,
    ):
        """
        Full sampling loop with CFG (SD/SDXL-style). Returns x_0 (denoised latent).

        When ``flow_matching_sample=True``, runs Euler/Heun integration in ``s`` (see
        ``_sample_loop_flow_matching``); VP ``scheduler`` / ``solver`` (ddim/heun) are not used.

        **Timestep placement** (``scheduler`` or ``timestep_schedule``): chooses discrete
        training indices — ``ddim``, ``euler``, ``karras_rho``, ``snr_uniform``, ``quad_cosine``
        (see ``diffusion.inference_timesteps``). ``timestep_schedule`` overrides ``scheduler``
        when set.

        **Solver** (``solver``): ``ddim`` = one DDIM-style update per step (default);
        ``heun`` = predictor–corrector with two model evaluations per step (often sharper,
        ~2× forward cost when SAG is off).

        cfg_rescale: ComfyUI-style; if > 0, rescale CFG delta to reduce over-saturation (e.g. 0.7).
        dynamic_threshold_percentile: if > 0, clamp x_0_pred to this percentile (e.g. 99.5).
        dynamic_threshold_type: "percentile" | "norm" | "spatial_norm" (ControlNet-style).
        dynamic_threshold_value: min norm for norm/spatial_norm (e.g. 1.0).

        volatile_cfg_boost: if > 0, when the latest latent update exceeds a quantile of recent
        updates, multiply CFG on subsequent steps by (1 + volatile_cfg_boost).

        periodic_alignment_interval: if > 0 and ``periodic_alignment_fn`` is set, every N completed
        steps call ``fn(step_index, x_0_pred)`` expecting a CLIP-like cosine in roughly [-1, 1].
        If the value is **below** ``periodic_alignment_threshold``, multiply CFG by
        (1 + ``periodic_alignment_cfg_boost``) for subsequent steps (multiplicative on current CFG).

        speculative_draft_cfg_scale: if >0 and uncond kwargs exist, run draft CFG forward then full CFG;
        when predictions are close (mean abs diff < speculative_close_thresh), blend toward draft.

        flow_matching_sample: if True, use rectified-flow sampler (``flow_solver`` = euler | heun).

        flow_init_noise: optional latent at ``s=1`` (same as training ``epsilon``); default is standard Gaussian noise.
        """
        model_kwargs_cond = model_kwargs_cond or {}
        model_kwargs_uncond = model_kwargs_uncond or {}
        do_inpaint = bool(
            inpaint_freeze_known and inpaint_mask is not None and inpaint_x0 is not None and inpaint_noise is not None
        )
        if flow_matching_sample:
            if do_inpaint:
                raise ValueError(
                    "flow_matching_sample is not compatible with inpaint_freeze_known (VP q_sample trajectory). "
                    "Use VP sampling or disable structured inpaint."
                )
            fs_flow = str(flow_solver).lower().strip()
            if fs_flow not in FLOW_INFERENCE_SOLVERS:
                raise ValueError(f"flow_solver must be one of {FLOW_INFERENCE_SOLVERS}, got {flow_solver!r}")
            return self._sample_loop_flow_matching(
                model,
                shape,
                model_kwargs_cond=model_kwargs_cond,
                model_kwargs_uncond=model_kwargs_uncond,
                cfg_scale=float(cfg_scale),
                cfg_rescale=float(cfg_rescale),
                num_inference_steps=int(num_inference_steps),
                flow_solver=fs_flow,
                device=device,
                dtype=dtype,
                x_init=x_init,
                start_timestep=start_timestep,
                speculative_draft_cfg_scale=float(speculative_draft_cfg_scale),
                speculative_close_thresh=float(speculative_close_thresh),
                speculative_blend=float(speculative_blend),
                dynamic_threshold_percentile=float(dynamic_threshold_percentile),
                dynamic_threshold_type=str(dynamic_threshold_type),
                dynamic_threshold_value=float(dynamic_threshold_value),
                flow_init_noise=flow_init_noise,
                control_guidance_start=float(control_guidance_start),
                control_guidance_end=float(control_guidance_end),
                control_guidance_decay=float(control_guidance_decay),
                holy_grail_enable=bool(holy_grail_enable),
                holy_grail_cfg_early_ratio=float(holy_grail_cfg_early_ratio),
                holy_grail_cfg_late_ratio=float(holy_grail_cfg_late_ratio),
                holy_grail_control_mult=float(holy_grail_control_mult),
                holy_grail_adapter_mult=float(holy_grail_adapter_mult),
                holy_grail_frontload_control=bool(holy_grail_frontload_control),
                holy_grail_late_adapter_boost=float(holy_grail_late_adapter_boost),
                holy_grail_cads_strength=float(holy_grail_cads_strength),
                holy_grail_cads_min_strength=float(holy_grail_cads_min_strength),
                holy_grail_cads_power=float(holy_grail_cads_power),
                holy_grail_unsharp_sigma=float(holy_grail_unsharp_sigma),
                holy_grail_unsharp_amount=float(holy_grail_unsharp_amount),
                holy_grail_clamp_quantile=float(holy_grail_clamp_quantile),
                holy_grail_clamp_floor=float(holy_grail_clamp_floor),
            )
        ts_name = str(timestep_schedule if timestep_schedule is not None else scheduler).lower().strip()
        sol = str(solver).lower().strip()
        if sol not in INFERENCE_SOLVERS:
            raise ValueError(f"Unknown solver {solver!r}. Choose one of: {INFERENCE_SOLVERS}")
        if do_inpaint:
            inpaint_mask = inpaint_mask.to(device=device, dtype=dtype)
            inpaint_x0 = inpaint_x0.to(device=device, dtype=dtype)
            inpaint_noise = inpaint_noise.to(device=device, dtype=dtype)
        if x_init is not None and start_timestep is not None:
            x = x_init.to(device=device, dtype=dtype)
            timesteps = self.set_timesteps_from(num_inference_steps, start_timestep).to(device)
        else:
            x = torch.randn(shape, device=device, dtype=dtype)
            timesteps = self.set_timesteps(
                num_inference_steps,
                timestep_schedule=ts_name,
                karras_rho=float(karras_rho),
            ).to(device)
        x_0_pred = None
        early_exit_enabled = float(ada_early_exit_delta_threshold) > 0.0 and int(ada_early_exit_patience) > 0
        early_exit_patience = int(ada_early_exit_patience)
        early_exit_min_steps = int(ada_early_exit_min_steps)
        early_exit_counter = 0
        cfg_box = [float(cfg_scale)]
        vol_deltas: list = []
        v_boost = float(volatile_cfg_boost)
        v_q = float(volatile_cfg_quantile)
        v_win = max(2, int(volatile_cfg_window))
        p_int = int(periodic_alignment_interval)
        p_thr = float(periodic_alignment_threshold)
        p_boost = float(periodic_alignment_cfg_boost)
        p_fn = periodic_alignment_fn
        spec_draft = float(speculative_draft_cfg_scale)
        spec_thr = float(speculative_close_thresh)
        spec_blend = float(speculative_blend)
        use_spec_cfg = spec_draft > 0.0 and model_kwargs_uncond is not None
        has_control = "control_image" in model_kwargs_cond
        base_control_scale = model_kwargs_cond.get("control_scale", 1.0)
        hg_enabled = bool(holy_grail_enable)
        if hg_enabled:
            from diffusion.holy_grail import HolyGrailRecipe, apply_condition_noise, build_holy_grail_step_plan, cads_noise_std

            hg_recipe = HolyGrailRecipe(
                base_cfg=float(cfg_scale),
                cfg_early_ratio=float(holy_grail_cfg_early_ratio),
                cfg_late_ratio=float(holy_grail_cfg_late_ratio),
                control_base_scale=float(holy_grail_control_mult),
                adapter_base_scale=float(holy_grail_adapter_mult),
                frontload_control=bool(holy_grail_frontload_control),
                late_adapter_boost=float(holy_grail_late_adapter_boost),
            )
        for i in range(len(timesteps)):
            t = timesteps[i].expand(shape[0])
            t_next = (
                timesteps[i + 1].expand(shape[0])
                if i + 1 < len(timesteps)
                else torch.zeros(shape[0], device=device, dtype=torch.long)
            )
            x_prev = x

            def _model_prediction(x_in: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
                cs = cfg_box[0]
                mk_c = model_kwargs_cond
                mk_u = model_kwargs_uncond
                p = 1.0 if len(timesteps) <= 1 else float(i) / float(max(len(timesteps) - 1, 1))
                if hg_enabled:
                    plan = build_holy_grail_step_plan(recipe=hg_recipe, step_index=i, total_steps=len(timesteps))
                    dyn = 1.0 if float(cfg_scale) == 0.0 else float(cfg_box[0]) / float(cfg_scale)
                    cs = float(plan.cfg_scale) * dyn
                if has_control:
                    sf = _control_guidance_scale_for_step(
                        1.0,
                        i,
                        len(timesteps),
                        start=float(control_guidance_start),
                        end=float(control_guidance_end),
                        decay_power=float(control_guidance_decay),
                    )
                    mk_c = dict(model_kwargs_cond)
                    if hg_enabled:
                        sf = sf * float(plan.control_scale)
                    mk_c["control_scale"] = _scale_control_value(base_control_scale, sf)
                    if hg_enabled and "adapter_scale" in mk_c:
                        mk_c["adapter_scale"] = _scale_control_value(mk_c.get("adapter_scale", 1.0), plan.adapter_scale)
                    if mk_u and ("control_image" in mk_u or "control_scale" in mk_u):
                        mk_u = dict(mk_u)
                        mk_u["control_scale"] = _scale_control_value(mk_u.get("control_scale", base_control_scale), sf)
                        if hg_enabled and "adapter_scale" in mk_u:
                            mk_u["adapter_scale"] = _scale_control_value(
                                mk_u.get("adapter_scale", 1.0), plan.adapter_scale
                            )
                if hg_enabled and float(holy_grail_cads_strength) > 0.0 and "encoder_hidden_states" in mk_c:
                    emb = mk_c.get("encoder_hidden_states", None)
                    if torch.is_tensor(emb):
                        mk_c = dict(mk_c)
                        sig = cads_noise_std(
                            progress=p,
                            base_strength=float(holy_grail_cads_strength),
                            min_strength=float(holy_grail_cads_min_strength),
                            power=float(holy_grail_cads_power),
                        )
                        mk_c["encoder_hidden_states"] = apply_condition_noise(emb, std=sig)
                if use_spec_cfg and cs != 1.0 and cs > 0 and mk_u:
                    from utils.generation.speculative_denoise import speculative_cfg_prediction

                    return speculative_cfg_prediction(
                        model,
                        x_in,
                        t_batch,
                        model_kwargs_cond=mk_c,
                        model_kwargs_uncond=mk_u,
                        cfg_scale=cs,
                        draft_cfg_scale=spec_draft,
                        cfg_rescale=float(cfg_rescale),
                        close_thresh=spec_thr,
                        blend_on_close=spec_blend,
                    )
                if cs != 1.0 and cs > 0 and mk_u:
                    out_cond = model(x_in, t_batch, **mk_c)
                    out_uncond = model(x_in, t_batch, **mk_u)
                    if out_cond.shape != x_in.shape and out_cond.shape[1] > x_in.shape[1]:
                        out_cond, out_uncond = out_cond[:, : x_in.shape[1]], out_uncond[:, : x_in.shape[1]]
                    delta = out_cond - out_uncond
                    if cfg_rescale > 0:
                        sig = delta.std() + 1e-8
                        scale = max(sig / cfg_rescale, 1.0)
                        delta = delta / scale
                    return out_uncond + cs * delta
                o = model(x_in, t_batch, **mk_c)
                if o.shape != x_in.shape and o.shape[1] > x_in.shape[1]:
                    o = o[:, : x_in.shape[1]]
                return o

            def _apply_sag(x_in: torch.Tensor, t_batch: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
                if float(sag_scale) > 0.0 and float(sag_blur_sigma) > 0.0:
                    from .sampling_utils import gaussian_blur_latent

                    x_b = gaussian_blur_latent(x_in, float(sag_blur_sigma))
                    out_b = _model_prediction(x_b, t_batch)
                    return base_out + float(sag_scale) * (base_out - out_b)
                return base_out

            out1 = _apply_sag(x, t, _model_prediction(x, t))
            if i + 1 < len(timesteps):
                if sol == "heun":
                    x_euler, _ = self.step_with_pred(
                        x,
                        t,
                        t_next,
                        out1,
                        eta=eta,
                        dynamic_threshold_percentile=dynamic_threshold_percentile,
                        dynamic_threshold_type=dynamic_threshold_type,
                        dynamic_threshold_value=dynamic_threshold_value,
                        pbfm_edge_boost=pbfm_edge_boost,
                        pbfm_edge_kernel=pbfm_edge_kernel,
                    )
                    out2 = _apply_sag(x_euler, t_next, _model_prediction(x_euler, t_next))
                    out = 0.5 * (out1 + out2)
                else:
                    out = out1
                x, x_0_pred = self.step_with_pred(
                    x,
                    t,
                    t_next,
                    out,
                    eta=eta,
                    dynamic_threshold_percentile=dynamic_threshold_percentile,
                    dynamic_threshold_type=dynamic_threshold_type,
                    dynamic_threshold_value=dynamic_threshold_value,
                    pbfm_edge_boost=pbfm_edge_boost,
                    pbfm_edge_kernel=pbfm_edge_kernel,
                )
                if v_boost > 0.0:
                    delta_lat_v = (x - x_prev).abs().mean()
                    try:
                        dv = float(delta_lat_v.detach().cpu().item())
                    except Exception:
                        dv = float(delta_lat_v.detach().mean().item())
                    vol_deltas.append(dv)
                    while len(vol_deltas) > v_win:
                        vol_deltas.pop(0)
                    if len(vol_deltas) >= 2:
                        qclip = float(np.clip(v_q, 0.05, 0.95))
                        thr = float(np.quantile(np.asarray(vol_deltas[:-1], dtype=np.float64), qclip))
                        if vol_deltas[-1] > thr:
                            cfg_box[0] = float(cfg_scale) * (1.0 + v_boost)
                        else:
                            cfg_box[0] = float(cfg_scale)
                if (
                    p_int > 0
                    and p_fn is not None
                    and p_boost > 0.0
                    and x_0_pred is not None
                    and (i + 1) % p_int == 0
                ):
                    try:
                        sim = float(p_fn(int(i), x_0_pred))
                        if sim < p_thr:
                            cfg_box[0] = float(cfg_box[0]) * (1.0 + p_boost)
                    except Exception:
                        pass
                if do_inpaint:
                    # Keep known/unmasked regions consistent with the forward noising trajectory.
                    x_known_next = self.q_sample(inpaint_x0, t_next, noise=inpaint_noise, noise_offset=0.0)
                    x = inpaint_mask * x + (1.0 - inpaint_mask) * x_known_next

                    # Ensure returned x0_pred also respects context for early-exit/return consistency.
                    if early_exit_enabled and i >= early_exit_min_steps and (i + 1) < len(timesteps):
                        x_0_pred = inpaint_mask * x_0_pred + (1.0 - inpaint_mask) * inpaint_x0

                if early_exit_enabled and i >= early_exit_min_steps:
                    # Heuristic: stop when average absolute latent change is below threshold for a few steps.
                    delta_lat = (x - x_prev).abs().mean()
                    try:
                        delta_val = float(delta_lat.detach().cpu().item())
                    except Exception:
                        delta_val = float(delta_lat.detach().mean().item())
                    if delta_val < float(ada_early_exit_delta_threshold):
                        early_exit_counter += 1
                    else:
                        early_exit_counter = 0
                    if early_exit_counter >= early_exit_patience:
                        break
            else:
                x_0_pred, _ = self._predict_x0_and_noise(out1, x, t)
                if dynamic_threshold_percentile > 0 or (
                    dynamic_threshold_type != "percentile" and dynamic_threshold_value > 0
                ):
                    x_0_pred = self._dynamic_threshold(
                        x_0_pred,
                        percentile=dynamic_threshold_percentile,
                        threshold_type=dynamic_threshold_type,
                        threshold_value=dynamic_threshold_value,
                    )
                if do_inpaint:
                    # Force final x0 for known/unmasked regions to match the provided context.
                    x_0_pred = inpaint_mask * x_0_pred + (1.0 - inpaint_mask) * inpaint_x0
        if hg_enabled and x_0_pred is not None:
            if float(holy_grail_unsharp_amount) > 0.0 and float(holy_grail_unsharp_sigma) > 0.0:
                from diffusion.holy_grail import unsharp_mask_latent

                x_0_pred = unsharp_mask_latent(
                    x_0_pred,
                    sigma=float(holy_grail_unsharp_sigma),
                    amount=float(holy_grail_unsharp_amount),
                )
            if float(holy_grail_clamp_quantile) > 0.0:
                from diffusion.holy_grail import dynamic_percentile_clamp

                x_0_pred = dynamic_percentile_clamp(
                    x_0_pred,
                    quantile=float(holy_grail_clamp_quantile),
                    floor=float(holy_grail_clamp_floor),
                )
        return x_0_pred
