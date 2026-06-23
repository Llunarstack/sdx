"""Token pruning — dynamically drop low-importance tokens."""

from typing import Tuple

import torch
import torch.nn as nn


class TokenPruning(nn.Module):
    """Dynamically prune unimportant tokens during generation."""

    def __init__(self, hidden_dim: int = 512, prune_ratio: float = 0.3):
        super().__init__()
        self.prune_ratio = prune_ratio

        # Predict token importance
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prune low-importance tokens."""
        importance = self.importance_scorer(tokens)

        # Keep only top-k tokens
        k = int(tokens.shape[1] * (1.0 - self.prune_ratio))
        _, indices = torch.topk(importance.squeeze(-1), k, dim=1)

        pruned = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return pruned, indices
