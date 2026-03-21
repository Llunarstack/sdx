# Sampling utilities ported from reference repos (ControlNet, etc.).
# Used by gaussian_diffusion for dynamic thresholding of x0 predictions.
import torch


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
