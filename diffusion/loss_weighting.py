# Optional loss weighting strategies (ported from generative-models).
# Used when loss_weighting != "min_snr" in config. sigma = sqrt(1 - alpha_cumprod) per timestep.
import torch


def sigma_from_alpha_cumprod(alpha_cumprod: torch.Tensor) -> torch.Tensor:
    """Noise level sigma from alpha_cumprod (variance of noise in x_t is 1 - alpha_cumprod)."""
    return (1.0 - alpha_cumprod).clamp(min=1e-8).sqrt()


def unit_weight(sigma: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(sigma, device=sigma.device, dtype=sigma.dtype)


def edm_weight(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
    """EDM weighting: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2."""
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data).clamp(min=1e-8) ** 2


def v_weight(sigma: torch.Tensor) -> torch.Tensor:
    return edm_weight(sigma, sigma_data=1.0)


def eps_weight(sigma: torch.Tensor) -> torch.Tensor:
    return sigma ** (-2.0)


def get_loss_weight(
    alpha_cumprod: torch.Tensor,
    weighting: str,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    Per-timestep loss weight from alpha_cumprod.
    weighting: "unit" | "edm" | "v" | "eps". (min_snr is applied separately in training_losses.)
    """
    sigma = sigma_from_alpha_cumprod(alpha_cumprod)
    if weighting == "unit":
        return unit_weight(sigma)
    if weighting == "edm":
        return edm_weight(sigma, sigma_data=sigma_data)
    if weighting == "v":
        return v_weight(sigma)
    if weighting == "eps":
        return eps_weight(sigma)
    return unit_weight(sigma)
