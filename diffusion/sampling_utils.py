"""
Sampling utilities used by GaussianDiffusion for dynamic thresholding and
latent post-processing.

Gaussian blur uses the native CUDA kernel (``sdx_cuda_gaussian_blur``) when
built, falling back to a pure-PyTorch depthwise conv2d otherwise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Append dimensions so ``x`` has ``target_dims`` total. From k-diffusion / ControlNet."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def norm_thresholding(x0: torch.Tensor, value: float) -> torch.Tensor:
    """Scale x0 so its per-sample norm is at least ``value``. Reduces oversaturation."""
    if value <= 0:
        return x0
    s = x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value)
    s = append_dims(s, x0.ndim)
    return x0 * (value / s)


def spatial_norm_thresholding(x0: torch.Tensor, value: float) -> torch.Tensor:
    """Per-spatial-location norm (over channels) clamped to min ``value``; scale x0."""
    if value <= 0:
        return x0
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)


def _gaussian_blur_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Pure-PyTorch depthwise Gaussian blur fallback."""
    s = float(sigma)
    _, c, _h, _w = x.shape
    device, dtype = x.device, x.dtype
    r = min(7, max(1, int(s * 2)))
    k = 2 * r + 1
    coords = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g1d = torch.exp(-(coords ** 2) / (2.0 * s * s + 1e-8))
    g1d = g1d / (g1d.sum() + 1e-8)
    k2d = torch.outer(g1d, g1d)
    k2d = k2d / (k2d.sum() + 1e-8)
    kernel = k2d.view(1, 1, k, k).expand(c, 1, k, k).to(dtype=dtype)
    return F.conv2d(x, kernel, padding=r, groups=c)


def gaussian_blur_latent(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Depthwise Gaussian blur on a latent tensor ``(B, C, H, W)``.

    Uses the native CUDA kernel (``sdx_cuda_gaussian_blur``) when built —
    avoids rebuilding the kernel tensor on every call. Falls back to a
    pure-PyTorch depthwise conv2d otherwise.

    Args:
        x: Float tensor of shape ``(B, C, H, W)``.
        sigma: Gaussian sigma in latent pixels (typical range 0.2–1.0).

    Returns:
        Blurred tensor with the same shape and dtype as ``x``.
    """
    s = float(sigma)
    if s <= 0.0 or x.dim() != 4:
        return x

    # Try native CUDA kernel first (avoids per-call kernel tensor allocation).
    try:
        from sdx_native.gaussian_blur_native import maybe_gaussian_blur_cuda

        x_np = x.detach().float().cpu().numpy()
        result = maybe_gaussian_blur_cuda(x_np, s)
        if result is not None:
            return torch.from_numpy(result).to(device=x.device, dtype=x.dtype)
    except Exception:
        pass

    return _gaussian_blur_torch(x, s)
