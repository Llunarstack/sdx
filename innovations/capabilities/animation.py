"""Animation from image — interpolate frames from a static image."""

from typing import List

import torch
import torch.nn as nn


class AnimationFromImage(nn.Module):
    """Create smooth animations from single static image."""

    def __init__(self, hidden_dim: int = 512, num_frames: int = 30):
        super().__init__()
        self.num_frames = num_frames

        # Motion detector
        self.motion_detector = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 2, 3, padding=1),  # Optical flow
        )

        # Frame interpolator
        self.frame_interpolator = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 3, 3, padding=1),
        )

    def animate(self, image: torch.Tensor, motion_type: str = "subtle") -> List[torch.Tensor]:
        """Create smooth animation from static image."""
        frames = [image]

        # Detect potential motion
        self.motion_detector(image)

        # Interpolate frames
        for i in range(1, self.num_frames):
            # Generate intermediate frame
            i / self.num_frames
            frame = self.frame_interpolator(image)
            frames.append(frame)

        return frames
