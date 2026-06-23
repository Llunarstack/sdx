"""Camera control — focal length, aperture, DOF, position, motion blur."""

from typing import Dict

import torch
import torch.nn as nn


class CameraController(nn.Module):
    """Cinematic camera control (focal length, aperture, DOF, motion)."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Focal length (mm)
        self.focal_length = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Aperture (f-stop)
        self.aperture = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Focus distance
        self.focus_distance = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Camera position and rotation
        self.camera_position = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        self.camera_rotation = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

        # Motion blur amount
        self.motion_blur = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, camera_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Define cinematic camera parameters."""
        return {
            "focal_length": self.focal_length(camera_spec),
            "aperture": self.aperture(camera_spec),
            "focus_distance": self.focus_distance(camera_spec),
            "position": self.camera_position(camera_spec),
            "rotation": self.camera_rotation(camera_spec),
            "motion_blur": self.motion_blur(camera_spec),
        }
