"""§1.2 Metallic material rendering — PBR roughness, normals, Fresnel."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetallicMaterialRenderer(nn.Module):
    """Physically-based metallic surface rendering."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.roughness_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )
        self.normal_map_generator = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 256, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(256, 3, 3, padding=1),
        )
        self.specular_highlighter = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, light_direction: torch.Tensor) -> torch.Tensor:
        roughness = self.roughness_predictor(x)
        normals = F.normalize(self.normal_map_generator(x.unsqueeze(-1).unsqueeze(-1)), dim=1)
        fresnel = torch.clamp(
            1.0 - torch.abs(torch.sum(normals * light_direction.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)),
            0,
            1,
        )
        specular = self.specular_highlighter(normals)
        return specular * fresnel * (1.0 - roughness.unsqueeze(-1).unsqueeze(-1))
