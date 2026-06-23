"""Consistent seeding — deterministic seed-to-latent encoding."""

import torch
import torch.nn as nn


class ConsistentSeeding(nn.Module):
    """Deterministic seeding for reproducible generation."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Map seed to deterministic latent
        self.seed_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )

    def encode_seed(self, seed: int) -> torch.Tensor:
        """Convert seed to deterministic embedding."""
        seed_tensor = torch.tensor([float(seed)], dtype=torch.float32)
        return self.seed_encoder(seed_tensor.unsqueeze(0))
