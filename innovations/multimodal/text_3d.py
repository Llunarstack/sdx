"""Text + 3D fusion — combine text descriptions with geometry guidance."""

import torch
import torch.nn as nn


class Text3DFusion(nn.Module):
    """Fuse text descriptions with 3D model guidance."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # 3D geometry encoder
        self.geometry_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Material predictor from geometry
        self.material_predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

        # Lighting from geometry
        self.lighting_from_geometry = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 9),
        )

    def forward(self, text_embedding: torch.Tensor, geometry_embedding: torch.Tensor) -> torch.Tensor:
        """Fuse text and 3D geometry."""
        geometry = self.geometry_encoder(geometry_embedding)
        materials = self.material_predictor(geometry)
        lighting = self.lighting_from_geometry(geometry)

        fused = torch.cat([text_embedding, materials, lighting], dim=-1)
        return fused
