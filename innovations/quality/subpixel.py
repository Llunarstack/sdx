"""§1.1 Subpixel refinement — progressive 2x upsampling for finer detail."""

import torch
import torch.nn as nn


class SubpixelRefinement(nn.Module):
    """Subpixel-level detail enhancement (4x quality boost)."""

    def __init__(self, channels: int, upscale_factor: int = 4):
        super().__init__()
        self.channels = channels
        self.upscale_factor = upscale_factor
        self.stage1 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.detail_fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        return self.detail_fusion(x2)
