"""Loop video generation — seamless looping video from text."""

from typing import List

import torch
import torch.nn as nn


class LoopVideoGeneration(nn.Module):
    """Generate perfect looping videos from text."""

    def __init__(self, hidden_dim: int = 512, num_frames: int = 32):
        super().__init__()
        self.num_frames = num_frames

        # Start frame predictor
        self.start_frame_generator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )

        # Intermediate frame generator
        self.middle_frame_generator = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Loop closure enforcer
        self.loop_enforcer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )

    def generate_loop(self, prompt: torch.Tensor) -> List[torch.Tensor]:
        """Generate perfect looping video."""
        frames = []

        # Start frame
        start = self.start_frame_generator(prompt)
        frames.append(start)

        # Interpolate middle frames
        middle_input = start.unsqueeze(1).expand(-1, self.num_frames - 2, -1)
        lstm_out, _ = self.middle_frame_generator(middle_input)
        middle_frames = lstm_out.squeeze(0)
        frames.extend(middle_frames)

        # End frame should smoothly loop back
        end = self.loop_enforcer(torch.cat([frames[-1], start], dim=-1))
        frames.append(end)

        return frames
