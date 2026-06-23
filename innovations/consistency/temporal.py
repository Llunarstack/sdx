"""Temporal consistency — smooth motion across video frames."""

import torch
import torch.nn as nn


class TemporalConsistency(nn.Module):
    """Maintain consistency across video frames (smooth motion)."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Motion predictor: predict next frame's features
        self.motion_predictor = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Optical flow predictor
        self.flow_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 2),  # (dx, dy)
        )

    def forward(self, frame_sequence: torch.Tensor) -> torch.Tensor:
        """Predict next frame maintaining temporal consistency."""
        motion, _ = self.motion_predictor(frame_sequence)
        return motion[:, -1, :]  # Last predicted frame
