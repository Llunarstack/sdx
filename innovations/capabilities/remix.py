"""Object remixing — swap objects between images."""

from typing import List

import torch
import torch.nn as nn


class ObjectRemixing(nn.Module):
    """Swap objects between images while maintaining realism."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Object segmenter
        self.segmenter = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Object extractor
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
        )

        # Contextual blender
        self.blender = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def remix(self, image1: torch.Tensor, image2: torch.Tensor, swap_list: List[str]) -> torch.Tensor:
        """Swap objects between images."""
        # Segment both images
        self.segmenter(image1)
        seg2 = self.segmenter(image2)

        # Extract objects
        extract1 = self.extractor(image1)
        extract2 = self.extractor(image2)

        # Swap and blend
        blended = self.blender(extract1 * seg2 + extract2 * (1 - seg2))
        return blended
