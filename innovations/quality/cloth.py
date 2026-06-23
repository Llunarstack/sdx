"""§1.4 Cloth fabric simulator — weave patterns per fabric type."""

import torch
import torch.nn as nn


class ClothFabricSimulator(nn.Module):
    """Physically accurate fabric rendering (silk, cotton, wool, etc.)."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.fabric_type_encoder = nn.Embedding(8, 64)
        self.weave_pattern = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.GELU(),
            nn.Linear(256, 64 * 8 * 8),
        )
        self.thread_renderer = nn.Conv2d(64, 3, 3, padding=1)
        self.light_interaction = nn.Conv2d(3, 3, 1)

    def forward(self, x: torch.Tensor, fabric_type: torch.Tensor) -> torch.Tensor:
        fabric_emb = self.fabric_type_encoder(fabric_type)
        combined = torch.cat([x, fabric_emb], dim=-1)
        weave = self.weave_pattern(combined)
        weave = weave.view(weave.shape[0], 64, 8, 8)
        threads = self.thread_renderer(weave)
        return self.light_interaction(threads)
