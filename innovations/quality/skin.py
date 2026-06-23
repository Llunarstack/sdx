"""§1.3 Skin texture — subsurface scattering, pores, veins."""

import torch
import torch.nn as nn


class SkinTextureAuthenticator(nn.Module):
    """Hyper-realistic skin rendering with subsurface scattering."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.subsurface_scattering = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64 * 8 * 8),
        )
        self.pore_generator = nn.Conv2d(64, 1, 3, padding=1)
        self.vein_network = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sss_features = self.subsurface_scattering(x)
        sss_features = sss_features.view(sss_features.shape[0], 64, 8, 8)
        pores = torch.sigmoid(self.pore_generator(sss_features))
        veins = torch.tanh(self.vein_network(sss_features))
        return pores * 0.1 + veins * 0.05
