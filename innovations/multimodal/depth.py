"""Depth-map guided generation — geometry from depth inputs."""

from typing import Dict

import torch
import torch.nn as nn


class DepthMapGuided(nn.Module):
    """Use depth maps to guide image generation."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Depth interpreter
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
        )

        # Normal map generator (convert depth to normals)
        self.normal_from_depth = nn.Conv2d(hidden_dim, 3, 3, padding=1)

        # Occlusion map generator
        self.occlusion_from_depth = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, depth_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract geometry from depth map."""
        depth_features = self.depth_encoder(depth_map)
        normals = self.normal_from_depth(depth_features)
        occlusion = self.occlusion_from_depth(depth_features)

        return {
            "depth_features": depth_features,
            "normals": normals,
            "occlusion": occlusion,
        }
