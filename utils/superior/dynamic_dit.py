"""
**DyDiT-style** dynamic compute helpers (ICLR 2025 / DyDiT++).

Training-free inference hooks:
- **Timestep-wise dynamic width (TDW)**: scale model output by denoise progress.
- **Spatial token importance (SDT-lite)**: heuristic mask for "easy" latent regions.

Does not modify checkpoint weights — scales activations during sampling.
"""

from __future__ import annotations

import torch


def timestep_dynamic_width(
    progress: float,
    *,
    early: float = 0.88,
    late: float = 1.0,
    power: float = 1.0,
) -> float:
    """
    Width multiplier in ``[early, late]`` vs denoise progress (0=noisy → 1=clean).

    Early timesteps run slightly narrower (less over-processing).
    """
    p = float(max(0.0, min(1.0, progress))) ** float(power)
    return float(early) + (float(late) - float(early)) * p


def spatial_token_importance(latent: torch.Tensor) -> torch.Tensor:
    """
    Per-token importance in ``[0,1]`` from local gradient magnitude (SDT-lite).

    ``latent``: (B, C, H, W) → returns (B, 1, H, W) importance map.
    """
    if latent.ndim != 4:
        return torch.ones(latent.shape[0], 1, 1, 1, device=latent.device, dtype=latent.dtype)
    x = latent.float()
    gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
    imp = (gx + gy) * 0.5
    imp = imp / (imp.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return imp.to(dtype=latent.dtype)


def apply_dynamic_width(
    out: torch.Tensor,
    progress: float,
    *,
    early: float = 0.88,
    late: float = 1.0,
) -> torch.Tensor:
    """Scale model prediction by TDW multiplier."""
    s = timestep_dynamic_width(progress, early=early, late=late)
    return out * s


def apply_sdt_latent_blend(
    latent: torch.Tensor,
    model_delta: torch.Tensor,
    *,
    keep_threshold: float = 0.35,
) -> torch.Tensor:
    """
    Blend model update toward zero on low-importance spatial regions.

    ``model_delta`` is the velocity/noise increment for this step.
    """
    imp = spatial_token_importance(latent)
    gate = (imp >= float(keep_threshold)).to(dtype=model_delta.dtype)
    return model_delta * gate


__all__ = [
    "apply_dynamic_width",
    "apply_sdt_latent_blend",
    "spatial_token_importance",
    "timestep_dynamic_width",
]
