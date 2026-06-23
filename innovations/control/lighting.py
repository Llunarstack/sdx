"""Lighting control — per-light parameters and ambient."""

from typing import Dict

import torch
import torch.nn as nn


class LightingController(nn.Module):
    """Ultra-precise lighting control (position, intensity, color, shadow)."""

    def __init__(self, hidden_dim: int = 512, num_lights: int = 5):
        super().__init__()
        self.num_lights = num_lights

        # Per-light controllers
        self.light_controllers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 8),  # (x, y, z, intensity, r, g, b, shadow_softness)
            )
            for _ in range(num_lights)
        ])

        # Global ambient light
        self.ambient = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, lighting_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Define precise lighting setup."""
        lights = []
        for controller in self.light_controllers:
            light_params = controller(lighting_spec)
            lights.append(light_params)

        ambient = self.ambient(lighting_spec)

        return {
            "lights": torch.stack(lights),
            "ambient": ambient,
        }
