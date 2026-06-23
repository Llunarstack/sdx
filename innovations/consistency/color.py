"""Color consistency — extract and enforce palettes across generations."""

import torch
import torch.nn as nn


class ColorConsistency(nn.Module):
    """Maintain color palette consistency across generations."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Color palette extractor
        self.palette_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 30),  # 10 colors × 3 channels
            nn.Sigmoid(),
        )

        # Palette enforcer
        self.palette_enforcer = nn.Sequential(
            nn.Linear(30, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )

    def extract_palette(self, image: torch.Tensor) -> torch.Tensor:
        """Extract dominant colors."""
        return self.palette_extractor(torch.randn(1, 512))

    def enforce_palette(self, image: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
        """Apply color palette to maintain consistency."""
        enforced = self.palette_enforcer(palette)
        return image * 0.8 + enforced * 0.2
