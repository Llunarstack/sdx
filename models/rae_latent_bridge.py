"""
RAE → DiT latent bridge (§11.6): maps Representation Autoencoder latents to SD-style 4-channel
diffusion latents and back for decode, so existing DiT (in_channels=4) can train on RAE data.

Trains two 1×1 convs: to_dit (C_rae → 4), to_rae (4 → C_rae). Optional cycle loss keeps them
approximately inverse on the RAE latent distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RAELatentBridge(nn.Module):
    """Maps RAE latent (B, C_rae, h, w) ↔ DiT latent (B, 4, h, w). Spatial size unchanged."""

    def __init__(self, rae_channels: int, dit_channels: int = 4):
        super().__init__()
        if rae_channels <= 0 or dit_channels <= 0:
            raise ValueError("rae_channels and dit_channels must be positive")
        self.rae_channels = int(rae_channels)
        self.dit_channels = int(dit_channels)
        self.to_dit = nn.Conv2d(self.rae_channels, self.dit_channels, kernel_size=1, bias=True)
        self.to_rae = nn.Conv2d(self.dit_channels, self.rae_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.to_dit.weight)
        nn.init.zeros_(self.to_dit.bias)
        nn.init.xavier_uniform_(self.to_rae.weight)
        nn.init.zeros_(self.to_rae.bias)

    def rae_to_dit(self, z_rae: torch.Tensor) -> torch.Tensor:
        return self.to_dit(z_rae)

    def dit_to_rae(self, z_dit: torch.Tensor) -> torch.Tensor:
        return self.to_rae(z_dit)

    def cycle_loss(self, z_rae: torch.Tensor) -> torch.Tensor:
        """L1 reconstruction: z ≈ to_rae(to_dit(z))."""
        z_dit = self.rae_to_dit(z_rae)
        back = self.dit_to_rae(z_dit)
        return (back - z_rae).abs().mean()
