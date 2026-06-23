"""Video-to-image style — extract style from video frames."""

import torch
import torch.nn as nn


class VideoToImageStyle(nn.Module):
    """Extract style from video frames to apply to images."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Frame sequence processor
        self.frame_processor = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Motion extractor
        self.motion_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Style aggregator
        self.style_aggregator = nn.Sequential(
            nn.Linear(hidden_dim + 128, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Extract consistent style from video."""
        # Process frame sequence
        lstm_out, (h_n, c_n) = self.frame_processor(video_frames)

        # Extract motion between frames
        motion = self.motion_extractor(torch.cat([lstm_out[:, 0], lstm_out[:, -1]], dim=-1))

        # Aggregate style
        style = self.style_aggregator(torch.cat([h_n[-1], motion], dim=-1))
        return style
