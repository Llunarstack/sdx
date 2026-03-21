# Gaussian diffusion for DiT: SD/SDXL-style features (offset noise, min-SNR, v-pred, DDIM, CFG).
import numpy as np
import torch
import torch.nn as nn

from .respace import space_timesteps
from .sampling_utils import norm_thresholding, spatial_norm_thresholding
from .schedules import get_beta_schedule
from .timestep_loss_weight import get_timestep_loss_weight


def create_diffusion(
    timestep_respacing: str = "",
    num_timesteps: int = 1000,
    beta_schedule: str = "linear",
    loss_type: str = "mse",
    prediction_type: str = "epsilon",
):
    """Create GaussianDiffusion. prediction_type: 'epsilon' (default) or 'v' (velocity, SD2-style).
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
        self.prediction_type = prediction_type  # "epsilon" or "v"
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
    ):
        """
        Compute training loss. SD/SDXL-style: offset noise, epsilon or v-prediction.
        loss_weighting: "min_snr" (default) | "min_snr_soft" | "unit" | "edm" | "v" | "eps".
        sample_weights: (B,) optional; per-sample loss weight (e.g. aesthetic score).
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
        # Target: epsilon or v (velocity). v = sqrt(alpha_bar)*noise - sqrt(1-alpha_bar)*x0
        self._to_device(device)
        sqrt_alpha = self.sqrt_alpha_cumprod.to(device)[t][(...,) + (None,) * (x_start.ndim - 1)]
        sqrt_one = self.sqrt_one_minus_alpha_cumprod.to(device)[t][(...,) + (None,) * (x_start.ndim - 1)]
        if self.prediction_type == "v":
            target = sqrt_alpha * noise - sqrt_one * x_start
        else:
            target = noise
        loss = nn.functional.mse_loss(model_out, target, reduction="none")
        # Per-sample mean (over spatial/channels)
        loss = loss.mean(dim=tuple(range(1, loss.ndim)))
        # Timestep loss weighting: min_snr / min_snr_soft / unit / edm / v / eps
        snr_t = self.snr.to(device)[t] if hasattr(self, "snr") else None
        alpha = self.alpha_cumprod.to(device)[t]
        weight = get_timestep_loss_weight(
            loss_weighting,
            snr=snr_t,
            alpha_cumprod=alpha,
            min_snr_gamma=min_snr_gamma,
            loss_weighting_sigma_data=loss_weighting_sigma_data,
        )
        loss = loss * weight
        # Aesthetic / sample weighting
        if sample_weights is not None and sample_weights.shape[0] == loss.shape[0]:
            sample_weights = sample_weights.to(device, dtype=loss.dtype)
            loss = (loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
        else:
            loss = loss.mean()
        return {"loss": loss}

    def _predict_x0_and_noise(self, model_out, x_t, t):
        """Convert model output (epsilon or v) to x_0 and noise."""
        self._to_device(x_t.device)
        alpha = self.alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        sigma = self.sqrt_one_minus_alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        sqrt_alpha = self.sqrt_alpha_cumprod.to(x_t.device)[t][(...,) + (None,) * (x_t.ndim - 1)]
        if self.prediction_type == "v":
            pred_noise = (sqrt_alpha * model_out + sigma * x_t) / (sigma + 1e-8)
            x_0_pred = sqrt_alpha * x_t - sigma * model_out
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

    def set_timesteps(self, num_inference_steps: int, scheduler: str = "ddim"):
        """Set inference timesteps. scheduler: 'ddim' (default, strided) or 'euler' (linear spacing)."""
        if scheduler == "euler":
            steps = np.linspace(0, self.num_timesteps - 1, num_inference_steps, dtype=np.int64)
            self.timesteps = torch.from_numpy(steps[::-1].copy())
        elif getattr(self, "timestep_respacing_str", None):
            steps = space_timesteps(self.num_timesteps, self.timestep_respacing_str)
            self.timesteps = torch.from_numpy(steps[::-1].copy().astype(np.int64))
        else:
            step = max(1, self.num_timesteps // num_inference_steps)
            self.timesteps = torch.from_numpy(np.arange(0, self.num_timesteps, step)[::-1].copy().astype(np.int64))
        return self.timesteps

    def set_timesteps_from(self, num_inference_steps: int, t_start: int):
        """Img2img / from-z: timesteps from t_start down to 0 (FLUX/SD-style)."""
        t_start = min(max(0, int(t_start)), self.num_timesteps - 1)
        # Linear spacing from t_start to 0 (inclusive)
        steps = np.linspace(t_start, 0, num_inference_steps, endpoint=True)
        self.timesteps = torch.from_numpy(steps.astype(np.int64)[::-1].copy())
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
        """DDIM-style step using pre-computed model output (for CFG). pred_out = combined prediction (epsilon or v)."""
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
    ):
        """
        Full sampling loop with CFG (SD/SDXL-style). Returns x_0 (denoised latent).
        scheduler: "ddim" (default) or "euler" (linear timestep spacing).
        cfg_rescale: ComfyUI-style; if > 0, rescale CFG delta to reduce over-saturation (e.g. 0.7).
        dynamic_threshold_percentile: if > 0, clamp x_0_pred to this percentile (e.g. 99.5).
        dynamic_threshold_type: "percentile" | "norm" | "spatial_norm" (ControlNet-style).
        dynamic_threshold_value: min norm for norm/spatial_norm (e.g. 1.0).
        """
        model_kwargs_cond = model_kwargs_cond or {}
        model_kwargs_uncond = model_kwargs_uncond or {}
        do_inpaint = bool(
            inpaint_freeze_known and inpaint_mask is not None and inpaint_x0 is not None and inpaint_noise is not None
        )
        if do_inpaint:
            inpaint_mask = inpaint_mask.to(device=device, dtype=dtype)
            inpaint_x0 = inpaint_x0.to(device=device, dtype=dtype)
            inpaint_noise = inpaint_noise.to(device=device, dtype=dtype)
        if x_init is not None and start_timestep is not None:
            x = x_init.to(device=device, dtype=dtype)
            timesteps = self.set_timesteps_from(num_inference_steps, start_timestep).to(device)
        else:
            x = torch.randn(shape, device=device, dtype=dtype)
            timesteps = self.set_timesteps(num_inference_steps, scheduler=scheduler).to(device)
        x_0_pred = None
        early_exit_enabled = float(ada_early_exit_delta_threshold) > 0.0 and int(ada_early_exit_patience) > 0
        early_exit_patience = int(ada_early_exit_patience)
        early_exit_min_steps = int(ada_early_exit_min_steps)
        early_exit_counter = 0
        for i in range(len(timesteps)):
            t = timesteps[i].expand(shape[0])
            t_next = (
                timesteps[i + 1].expand(shape[0])
                if i + 1 < len(timesteps)
                else torch.zeros(shape[0], device=device, dtype=torch.long)
            )
            x_prev = x
            if cfg_scale != 1.0 and cfg_scale > 0 and model_kwargs_uncond:
                out_cond = model(x, t, **model_kwargs_cond)
                out_uncond = model(x, t, **model_kwargs_uncond)
                if out_cond.shape != x.shape and out_cond.shape[1] > x.shape[1]:
                    out_cond, out_uncond = out_cond[:, : x.shape[1]], out_uncond[:, : x.shape[1]]
                delta = out_cond - out_uncond
                if cfg_rescale > 0:
                    sigma = delta.std() + 1e-8
                    scale = max(sigma / cfg_rescale, 1.0)
                    delta = delta / scale
                out = out_uncond + cfg_scale * delta
            else:
                out = model(x, t, **model_kwargs_cond)
                if out.shape != x.shape and out.shape[1] > x.shape[1]:
                    out = out[:, : x.shape[1]]
            if i + 1 < len(timesteps):
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
                x_0_pred, _ = self._predict_x0_and_noise(out, x, t)
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
        return x_0_pred
