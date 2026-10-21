"""
Sampling utilities used by GaussianDiffusion: dynamic thresholding and latent blur / unsharp masking.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Append trailing singleton dims so ``x`` has ``target_dims`` (k-diffusion)."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def norm_thresholding(x0: torch.Tensor, value: float) -> torch.Tensor:
    """Scale each sample down so its RMS is at most ``value`` (reduces oversaturation)."""
    if value <= 0:
        return x0
    s = append_dims(x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value), x0.ndim)
    return x0 * (value / s)


def spatial_norm_thresholding(x0: torch.Tensor, value: float) -> torch.Tensor:
    """Like ``norm_thresholding`` but per spatial location (RMS over channels)."""
    if value <= 0:
        return x0
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)


def dynamic_threshold(
    x: torch.Tensor, quantile: float, data_range: float = 1.0, *, rescale: bool = True
) -> torch.Tensor:
    """Dynamic thresholding (Saharia et al. 2022, Imagen Sec. 2.3), per sample.

    ``s = max(quantile(|x|), data_range)``; ``x`` is clipped to ``[-s, s]`` and, with ``rescale``, mapped back
    onto ``[-data_range, data_range]`` (Imagen: ``data_range = 1``, divide by ``s``). A no-op while ``x``
    stays inside the data range. ``rescale=False`` only clips outliers.
    """
    q = torch.quantile(x.detach().flatten(1).abs().float(), float(quantile), dim=1)
    s = append_dims(q.clamp_min(float(data_range)).to(x.dtype), x.ndim)
    x = torch.maximum(torch.minimum(x, s), -s)
    return x * (float(data_range) / s) if rescale else x


def _gaussian_blur_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian blur with a kernel radius of ``2 * sigma`` clamped to ``[1, 7]``."""
    c = x.shape[1]
    r = min(7, max(1, int(sigma * 2)))
    k = 2 * r + 1
    coords = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    g1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma + 1e-8))
    g1d = g1d / g1d.sum()
    kernel = torch.outer(g1d, g1d).view(1, 1, k, k).expand(c, 1, k, k).to(dtype=x.dtype)
    return F.conv2d(x, kernel, padding=r, groups=c)


def gaussian_blur_latent(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise Gaussian blur of a ``(B, C, H, W)`` latent; ``sigma`` in latent pixels (typically 0.2-1.0).

    CPU tensors use the native CUDA kernel (``sdx_cuda_gaussian_blur``) when it is built; GPU tensors stay on
    the device (a host round trip per sampler step would cost more than the conv).
    """
    s = float(sigma)
    if s <= 0.0 or x.dim() != 4:
        return x
    if x.device.type == "cpu":
        try:
            from sdx_native.gaussian_blur_native import maybe_gaussian_blur_cuda

            result = maybe_gaussian_blur_cuda(x.detach().float().numpy(), s)
        except (ImportError, OSError, RuntimeError):
            result = None
        if result is not None:
            return torch.from_numpy(result).to(dtype=x.dtype)
    return _gaussian_blur_torch(x, s)


def unsharp_mask_latent(x: torch.Tensor, *, sigma: float = 0.6, amount: float = 0.25) -> torch.Tensor:
    """``x + amount * (x - blur(x))``: mild high-frequency lift for the final latent (holy-grail recipe)."""
    return x + float(amount) * (x - gaussian_blur_latent(x, sigma=float(sigma)))
