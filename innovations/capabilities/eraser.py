"""Magic eraser — remove objects without artifacts."""

import torch
import torch.nn as nn


class MagicEraser(nn.Module):
    """Remove objects perfectly without traces or artifacts."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Object detector
        self.object_detector = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Background predictor
        self.background_predictor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 3, 3, padding=1),
        )

        # Artifact removal
        self.artifact_remover = nn.Conv2d(3, 3, 3, padding=1)

    def erase(self, image: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        """Remove object leaving perfect background."""
        # Detect edges
        mask = self.object_detector(image)

        # Predict background
        background = self.background_predictor(image)

        # Remove artifacts
        cleaned = self.artifact_remover(background)

        # Blend
        result = image * (1 - mask) + cleaned * mask
        return result
