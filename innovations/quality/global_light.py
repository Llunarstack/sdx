"""§1.6 Global illumination — ambient occlusion and indirect light."""

import torch
import torch.nn as nn


class GlobalIlluminationApproximator(nn.Module):
    """Approximate global illumination for more realistic lighting."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.ao_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.indirect_light = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
            nn.ReLU(),
        )
        self.env_probe = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.mean(dim=list(range(2, x.dim())))
        ao = self.ao_predictor(x)
        indirect = self.indirect_light(x)
        self.env_probe(x)
        return indirect * ao
