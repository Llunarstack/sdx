"""Detail intensity control — surface, pore, wrinkle, micro-detail."""

from typing import Dict

import torch
import torch.nn as nn


class DetailIntensityController(nn.Module):
    """Control texture detail level independently from base image."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Surface detail intensity
        self.surface_detail = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Pore/texture visibility
        self.pore_visibility = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Wrinkle depth
        self.wrinkle_depth = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Material micro-detail (roughness)
        self.micro_detail = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, detail_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Control detail levels at different scales."""
        return {
            "surface_detail": self.surface_detail(detail_spec),
            "pore_visibility": self.pore_visibility(detail_spec),
            "wrinkle_depth": self.wrinkle_depth(detail_spec),
            "micro_detail": self.micro_detail(detail_spec),
        }
