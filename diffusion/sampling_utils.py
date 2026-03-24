# Sampling utilities ported from reference repos (ControlNet, etc.).
# Used by gaussian_diffusion for dynamic thresholding of x0 predictions.
import torch
import torch.nn.functional as F


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Append dimensions so x has target_dims. From k-diffusion / ControlNet."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def norm_thresholding(x0: torch.Tensor, value: float) -> torch.Tensor:
    """Scale x0 so that its norm (per sample, over all dims) is at least `value`. Reduces oversaturation. From ControlNet."""
    if value <= 0:
        return x0
    s = x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value)
    s = append_dims(s, x0.ndim)
    return x0 * (value / s)


def spatial_norm_thresholding(x0: torch.Tensor, value: float) -> torch.Tensor:
    """Per-spatial-location norm (over channels) clamped to min `value`; scale x0. From ControlNet."""
    if value <= 0:
        return x0
    # x0: (B, C, H, W) -> s: (B, 1, H, W)
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)


def gaussian_blur_latent(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Depthwise Gaussian blur on latent (B, C, H, W). sigma in latent pixels (typical 0.2–1.0).
    """
    s = float(sigma)
    if s <= 0.0:
        return x
    if x.dim() != 4:
        return x
    _b, c, _h, _w = x.shape
    device, dtype = x.device, x.dtype
    r = min(7, max(1, int(s * 2)))
    k = 2 * r + 1
    coords = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g1d = torch.exp(-(coords**2) / (2.0 * (s**2) + 1e-8))
    g1d = g1d / (g1d.sum() + 1e-8)
    k2d = torch.outer(g1d, g1d)
    k2d = k2d / (k2d.sum() + 1e-8)
    kernel = k2d.view(1, 1, k, k).expand(c, 1, k, k).to(dtype=dtype)
    pad = r
    return F.conv2d(x, kernel, padding=pad, groups=c)
