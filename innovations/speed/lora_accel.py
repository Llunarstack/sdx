"""LoRA acceleration — low-rank adaptation for fast generation."""

import torch
import torch.nn as nn


class LoRAAcceleration(nn.Module):
    """Ultra-fast generation using Low-Rank Adaptation (LoRA)."""

    def __init__(self, hidden_dim: int = 512, rank: int = 32):
        super().__init__()
        self.rank = rank

        # LoRA weights (much smaller than full weights)
        self.lora_down = nn.Linear(hidden_dim, rank)
        self.lora_up = nn.Linear(rank, hidden_dim)

        # LoRA scaling
        self.scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation for fast generation."""
        lora_out = self.lora_up(torch.relu(self.lora_down(x)))
        return base_output + lora_out * self.scale
