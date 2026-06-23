"""Layer skipping — skip layers based on input complexity."""

from typing import List

import torch
import torch.nn as nn


class LayerSkipping(nn.Module):
    """Skip unnecessary layers based on input complexity."""

    def __init__(self, num_layers: int = 12, hidden_dim: int = 512):
        super().__init__()
        self.num_layers = num_layers

        # Predict which layers to skip
        self.layer_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_layers),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor) -> List[bool]:
        """Determine which layers to skip."""
        skip_logits = self.layer_predictor(embedding)
        skip_mask = skip_logits > 0.5
        return skip_mask
