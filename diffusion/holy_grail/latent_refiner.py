from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0 or x.dim() != 4:
        return x
    b, c, _h, _w = x.shape
    del b
    r = max(1, min(7, int(round(float(sigma) * 2.0))))
    k = 2 * r + 1
    coords = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    g = torch.exp(-(coords**2) / (2.0 * float(sigma) * float(sigma) + 1e-8))
    g = g / (g.sum() + 1e-8)
    k2d = torch.outer(g, g)
    k2d = k2d / (k2d.sum() + 1e-8)
    kernel = k2d.view(1, 1, k, k).expand(c, 1, k, k).to(dtype=x.dtype)
    return F.conv2d(x, kernel, padding=r, groups=c)


def unsharp_mask_latent(
    x: torch.Tensor,
    *,
    sigma: float = 0.6,
    amount: float = 0.25,
) -> torch.Tensor:
    """
    Mild high-frequency enhancement for late denoise stages.
    """
    blurred = _gaussian_blur(x, sigma=float(sigma))
    return x + float(amount) * (x - blurred)


def dynamic_percentile_clamp(
    x: torch.Tensor,
    *,
    quantile: float = 0.995,
    floor: float = 1.0,
) -> torch.Tensor:
    """
    Per-sample dynamic clipping with percentile-based bound.
    """
    if x.dim() < 2:
        return x
    q = max(0.5, min(0.999, float(quantile)))
    xf = x.flatten(1).abs()
    bound = torch.quantile(xf, q=q, dim=1, keepdim=True).clamp(min=float(floor))
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
    Blend with teacher latent estimate.
    """
    if teacher is None:
        return x
    if teacher.shape != x.shape:
        raise ValueError("teacher shape must match x")
    s = max(0.0, min(1.0, float(strength)))
    return (1.0 - s) * x + s * teacher.to(dtype=x.dtype)

