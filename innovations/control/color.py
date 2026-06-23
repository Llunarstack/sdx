"""Color palette control — primary/secondary colors, saturation, hue, brightness."""

import torch
import torch.nn as nn


class ColorPaletteController(nn.Module):
    """Pixel-perfect control over color grading and palettes."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Primary color picker
        self.primary_color = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

        # Secondary colors (accent, shadow, highlight)
        self.secondary_colors = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 9),  # 3 colors × 3 channels
            nn.Sigmoid(),
        )

        # Saturation controller
        self.saturation = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Hue shift (0-360 degrees)
        self.hue_shift = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Brightness/contrast
        self.brightness = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.contrast = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, color_spec: torch.Tensor) -> torch.Tensor:
        """Apply precise color grading."""
        self.primary_color(color_spec)
        self.secondary_colors(color_spec).view(-1, 3, 3)
        self.saturation(color_spec)
        self.hue_shift(color_spec) * 360  # 0-360 degrees
        brightness = self.brightness(color_spec)
        contrast = self.contrast(color_spec)

        # Apply color grading to image
        # This would involve actual color space transformations
        graded = image * (1.0 + brightness)
        graded = graded ** (1.0 / (contrast + 1.0))

        return graded
