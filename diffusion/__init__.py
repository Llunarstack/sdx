"""
sdx.diffusion — Diffusion schedules, Gaussian diffusion wrapper, and sampling helpers.

Submodule summary
-----------------
gaussian_diffusion  Core VP-DDPM and flow-matching sampling loop.
flow_matching       Rectified-flow training losses.
schedules           VP beta schedules (linear, cosine, squaredcos_cap_v2, sigmoid).
snr_utils           SNR computation helpers.
sampling_utils      Blur, thresholding, and norm helpers for sampling.
cfg_schedulers      Classifier-free guidance scale schedules.
consistency_utils   Consistency model utilities (EMA target, one-step refine).
timestep_sampling   Training timestep samplers (uniform, logit-normal, high-noise).
self_conditioning   Self-conditioning helpers.
spectral_sfp        Frequency-weighted spectral loss.
pixel_perfect       Pixel-perfect canvas and AR grid utilities.
bridge_training     Latent bridge / Schrödinger-bridge auxiliary losses.
latent_bridge       Linear latent interpolation utilities.
attention_steering  Attention steering (AST) for inference-time control.
inference_timesteps Inference timestep schedule builders.
respace             Legacy DDIM respacing utilities.
"""

import warnings as _warnings
from importlib.util import find_spec as _find_spec

_TORCH_AVAILABLE = _find_spec("torch") is not None

# ---------------------------------------------------------------------------
# Torch-free exports (used by lightweight tooling + prompt helpers).
# ---------------------------------------------------------------------------
from .pixel_perfect import (  # noqa: E402
    LATENT_TO_PIXEL,
    PixelPerfectCanvas,
    ar_block_grid_side,
    dit_rgb_stride_px,
    latent_hw_from_pixels,
    pixel_stride_for_pipeline,
    pixels_from_latent_hw,
    resolve_pixel_perfect_hw,
    snap_to_multiple,
    tag_manifest_pixel_perfect,
    validate_latent_matches_ar_grid,
    validate_pixels_against_dit,
)
from .schedules import get_beta_schedule  # noqa: E402
from .snr_utils import alpha_cumprod_from_betas, snr_from_alpha_cumprod, snr_from_betas  # noqa: E402

__all__ = [
    "LATENT_TO_PIXEL",
    "PixelPerfectCanvas",
    "ar_block_grid_side",
    "dit_rgb_stride_px",
    "latent_hw_from_pixels",
    "pixels_from_latent_hw",
    "pixel_stride_for_pipeline",
    "resolve_pixel_perfect_hw",
    "snap_to_multiple",
    "tag_manifest_pixel_perfect",
    "validate_latent_matches_ar_grid",
    "validate_pixels_against_dit",
    "get_beta_schedule",
    "alpha_cumprod_from_betas",
    "snr_from_alpha_cumprod",
    "snr_from_betas",
]

if _TORCH_AVAILABLE:
    # -----------------------------------------------------------------------
    # Torch-dependent exports.
    # -----------------------------------------------------------------------
    from .attention_steering import ASTConfig, AttentionSteerer, steer_attention  # noqa: E402, F401
    from .cfg_schedulers import (  # noqa: E402, F401
        cfg_scale_cosine_ramp,
        cfg_scale_linear,
        cfg_scale_piecewise,
        cfg_scale_snr_aware,
    )
    from .consistency_utils import (  # noqa: E402, F401
        consistency_delta_loss,
        one_step_consistency_refine,
        temporal_ema_target,
    )
    from .gaussian_diffusion import (  # noqa: E402, F401
        INFERENCE_SOLVERS,
        GaussianDiffusion,
        canonicalize_flow_solver,
        canonicalize_vp_solver,
        create_diffusion,
        list_inference_solver_aliases,
    )
    from .inference_timesteps import build_inference_timesteps, list_timestep_schedules  # noqa: E402, F401
    from .losses import get_loss_weight, get_timestep_loss_weight  # noqa: E402, F401
    from .respace import space_timesteps  # noqa: E402, F401
    from .sampling_utils import norm_thresholding, spatial_norm_thresholding  # noqa: E402, F401
    from .self_conditioning import blend_self_cond, maybe_detached_self_cond  # noqa: E402, F401
    from .timestep_sampling import sample_training_timesteps  # noqa: E402, F401

    try:
        from .bridge_training import bridge_aux_vp_loss, shuffle_pair_latents  # noqa: E402
    except Exception as _e:
        _warnings.warn(f"sdx.diffusion: bridge_training unavailable: {_e}", ImportWarning, stacklevel=2)
        bridge_aux_vp_loss = None  # type: ignore[assignment]
        shuffle_pair_latents = None  # type: ignore[assignment]

    try:
        from .flow_matching import flow_matching_per_sample_losses  # noqa: E402
    except Exception as _e:
        _warnings.warn(f"sdx.diffusion: flow_matching unavailable: {_e}", ImportWarning, stacklevel=2)
        flow_matching_per_sample_losses = None  # type: ignore[assignment]

    try:
        from .latent_bridge import linear_latent_interp  # noqa: E402
    except Exception as _e:
        _warnings.warn(f"sdx.diffusion: latent_bridge unavailable: {_e}", ImportWarning, stacklevel=2)
        linear_latent_interp = None  # type: ignore[assignment]

    try:
        from .spectral_sfp import spectral_sfp_per_sample_loss, time_frequency_weights  # noqa: E402
    except Exception as _e:
        _warnings.warn(f"sdx.diffusion: spectral_sfp unavailable: {_e}", ImportWarning, stacklevel=2)
        spectral_sfp_per_sample_loss = None  # type: ignore[assignment]
        time_frequency_weights = None  # type: ignore[assignment]

    __all__.extend(
        [
            "INFERENCE_SOLVERS",
            "canonicalize_flow_solver",
            "canonicalize_vp_solver",
            "list_inference_solver_aliases",
            "build_inference_timesteps",
            "list_timestep_schedules",
            "create_diffusion",
            "GaussianDiffusion",
            "cfg_scale_linear",
            "cfg_scale_cosine_ramp",
            "cfg_scale_piecewise",
            "cfg_scale_snr_aware",
            "maybe_detached_self_cond",
            "blend_self_cond",
            "temporal_ema_target",
            "consistency_delta_loss",
            "one_step_consistency_refine",
            "get_loss_weight",
            "get_timestep_loss_weight",
            "space_timesteps",
            "norm_thresholding",
            "spatial_norm_thresholding",
            "sample_training_timesteps",
            "ASTConfig",
            "AttentionSteerer",
            "steer_attention",
            "bridge_aux_vp_loss",
            "shuffle_pair_latents",
            "flow_matching_per_sample_losses",
            "linear_latent_interp",
            "spectral_sfp_per_sample_loss",
            "time_frequency_weights",
        ]
    )
else:

    def __getattr__(name: str):  # type: ignore[misc]
        raise ModuleNotFoundError(
            f"torch is required for diffusion.{name}, but torch is not installed in this environment."
        )
