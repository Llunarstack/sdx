"""§1.5 Liquid physics — refraction, caustics, surface tension."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidPhysicsRenderer(nn.Module):
    """Real-time liquid dynamics (water, liquid metal, etc.)."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.surface_tension = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )
        self.refraction_map = nn.Conv2d(64, 2, 3, padding=1)
        self.caustics_generator = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, base_color: torch.Tensor) -> torch.Tensor:
        surface = self.surface_tension(x)
        surface = surface.view(surface.shape[0], 64, 8, 8)
        refraction = self.refraction_map(surface)
        caustics = torch.sigmoid(self.caustics_generator(surface))
        refracted = F.grid_sample(
            base_color, refraction.permute(0, 2, 3, 1).unsqueeze(1), align_corners=False
        ).squeeze(2)
        return refracted * (1.0 + caustics * 0.3)
