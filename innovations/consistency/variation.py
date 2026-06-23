"""Variation control — identical, similar, or varied reproduction."""

import torch
import torch.nn as nn


class VariationControl(nn.Module):
    """Control variation level: identical, similar, varied."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Variation amount (0-1)
        self.variationler = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, base_latent: torch.Tensor, variation_amount: float) -> torch.Tensor:
        """Add controlled variation to latent."""
        if variation_amount == 0.0:
            # Exact reproduction
            return base_latent

        # Add Gaussian noise scaled by variation amount
        noise = torch.randn_like(base_latent) * variation_amount
        return base_latent + noise
