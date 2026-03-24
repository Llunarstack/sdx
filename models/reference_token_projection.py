"""Map CLIP-style image embeddings to extra cross-attention tokens (IP-Adapter-style, untrained by default)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ReferenceTokenProjector(nn.Module):
    """
    Project a pooled image embedding (B, clip_dim) to (B, num_tokens, hidden_size)
    for concatenation to text_emb inside DiT_Text.
    """

    def __init__(self, clip_dim: int, hidden_size: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        h = max(int(clip_dim), int(hidden_size))
        self.net = nn.Sequential(
            nn.Linear(int(clip_dim), h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, self.hidden_size * self.num_tokens),
        )
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[3].weight, std=0.02)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        b = image_embeds.shape[0]
        x = self.net(image_embeds)
        return x.view(b, self.num_tokens, self.hidden_size)
