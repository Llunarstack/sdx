"""Semantic consistency — anchor and validate meaning across variations."""

import torch
import torch.nn as nn


class SemanticConsistency(nn.Module):
    """Ensure semantic meaning stays consistent across variations."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Semantic anchor: key concepts that must remain
        self.semantic_anchor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Semantic validator
        self.semantic_validator = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def anchor_semantics(self, prompt: torch.Tensor) -> torch.Tensor:
        """Extract and store semantic anchors from prompt."""
        return self.semantic_anchor(prompt)

    def validate_semantics(self, generated: torch.Tensor, anchor: torch.Tensor) -> float:
        """Check if generation preserves semantic meaning."""
        combined = torch.cat([generated, anchor], dim=-1)
        score = self.semantic_validator(combined)
        return score.detach().item()
