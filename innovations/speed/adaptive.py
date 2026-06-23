"""Adaptive quality levels — parallel low/medium/high generation."""

from typing import List

import torch
import torch.nn as nn


class AdaptiveQualityLevels(nn.Module):
    """Generate at multiple quality levels, progressively refine."""

    def __init__(self):
        super().__init__()
        self.levels = 3  # Low, medium, high quality

        # Level-specific generators
        self.generators = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding=1) for _ in range(self.levels)
        ])

    def forward(self, latent: torch.Tensor) -> List[torch.Tensor]:
        """Generate at all quality levels in parallel."""
        outputs = []
        for gen in self.generators:
            output = gen(latent)
            outputs.append(output)
        return outputs
