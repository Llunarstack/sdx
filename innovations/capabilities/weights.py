"""Prompt weighting — per-token influence control."""

from typing import List

import torch
import torch.nn as nn


class PromptWeighting(nn.Module):
    """Ultra-fine control over which words influence the image."""

    def __init__(self, hidden_dim: int = 512, max_tokens: int = 77):
        super().__init__()
        self.max_tokens = max_tokens

        # Per-token weight predictor
        self.token_weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, max_tokens),
            nn.Softmax(dim=-1),
        )

        # Token importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, max_tokens),
            nn.Sigmoid(),
        )

    def compute_weights(self, tokens: torch.Tensor, weights: List[float] = None) -> torch.Tensor:
        """Compute influence weight for each token."""
        if weights is not None:
            # User-specified weights
            return torch.tensor(weights)

        # Automatic importance scoring
        importance = self.importance_scorer(tokens.mean(dim=1))
        return importance
