"""
Latent refinement helpers for the Holy Grail adaptive sampling stack.

``unsharp_mask_latent`` and ``dynamic_percentile_clamp`` are called at the
end of every sampling run when enabled. Both delegate to native CUDA kernels
(``sdx_cuda_gaussian_blur``, ``sdx_cuda_percentile_clamp``) when built,
falling back to pure PyTorch otherwise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion.sampling_utils import gaussian_blur_latent


def unsharp_mask_latent(
    x: torch.Tensor,
    *,
    sigma: float = 0.6,
    amount: float = 0.25,
) -> torch.Tensor:
    """
    Mild high-frequency enhancement for late denoise stages.

    Computes ``x + amount * (x - blur(x))``. Uses the native CUDA Gaussian
    blur kernel when available.

    Args:
        x: Latent tensor ``(B, C, H, W)``.
        sigma: Blur sigma (latent pixels).
        amount: Sharpening strength (0 = no-op).

    Returns:
        Sharpened tensor with the same shape and dtype as ``x``.
    """
    blurred = gaussian_blur_latent(x, sigma=float(sigma))
    return x + float(amount) * (x - blurred)


def dynamic_percentile_clamp(
    x: torch.Tensor,
    *,
    quantile: float = 0.995,
    floor: float = 1.0,
) -> torch.Tensor:
    """
    Per-sample dynamic clipping: ``x = clamp(x, -bound, bound) / bound``
    where ``bound = max(quantile(|x|), floor)``.

    Uses the native CUDA kernel (``sdx_cuda_percentile_clamp``) when built,
    falling back to pure PyTorch otherwise.

    Args:
        x: Tensor of at least 2 dimensions; first dim is batch.
        quantile: Percentile for bound computation (0.5–0.999).
        floor: Minimum bound value (prevents division by near-zero).

    Returns:
        Clamped and normalised tensor with the same shape and dtype as ``x``.
    """
    if x.dim() < 2:
        return x

    q = max(0.5, min(0.999, float(quantile)))
    floor_val = float(floor)

    # Try native CUDA kernel (avoids torch.quantile overhead on small tensors).
    try:
        import numpy as np
        from sdx_native.percentile_clamp_native import maybe_percentile_clamp_cuda

        orig_shape = x.shape
        B = orig_shape[0]
        row_len = x.numel() // B
        x_np = x.detach().float().cpu().numpy().reshape(B, row_len)
        result = maybe_percentile_clamp_cuda(x_np, q, floor_val)
        if result is not None:
            return torch.from_numpy(result).reshape(orig_shape).to(device=x.device, dtype=x.dtype)
    except Exception:
        pass

    # Pure-PyTorch fallback.
    xf = x.flatten(1).abs()
    bound = torch.quantile(xf, q=q, dim=1, keepdim=True).clamp(min=floor_val)
    while bound.dim() < x.dim():
        bound = bound.unsqueeze(-1)
    return x.clamp(min=-bound, max=bound) / bound


def consistency_blend_latent(
    x: torch.Tensor,
    teacher: torch.Tensor | None,
    *,
    strength: float = 0.15,
) -> torch.Tensor:
    """
    Blend ``x`` toward a teacher latent estimate.

    Args:
        x: Current latent.
        teacher: Target latent (must match ``x.shape``). No-op if ``None``.
        strength: Blend weight toward teacher (0 = keep x, 1 = use teacher).

    Returns:
        Blended tensor.
    """
    if teacher is None:
        return x
    if teacher.shape != x.shape:
        raise ValueError(
            f"teacher shape {teacher.shape} must match x shape {x.shape}"
        )
    s = max(0.0, min(1.0, float(strength)))
    return (1.0 - s) * x + s * teacher.to(dtype=x.dtype)
