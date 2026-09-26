"""
2D -> 3D conditioning bridge.

The whole philosophy of SDX-Spatial is that *an image is just one camera view of
a 3D thing*. This module is where that becomes literal: it takes the conditioning
SDX's 2D DiT already produces — a pooled text embedding, and optionally the patch
tokens of an encoded image — and lifts it into the single conditioning vector that
seeds a :class:`~frontier.multiview.triplane.SpatialLatentField`.

Two entry points fall out of one module:

  * **text -> 3D**  : pass only the text embedding.
  * **image -> 3D** : pass image patch tokens (from the VAE latent or a ViT). The
    tokens are attention-pooled so a single reference photo can seed the field.

Keeping this as a thin, separately trainable adapter means the expensive 2D
backbone stays frozen: you reuse everything SDX already learned about the visual
world and only train the lift into 3D.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(slots=True)
class BridgeConfig:
    text_dim: int = 1024  # dim of the pooled text embedding from SDX
    image_dim: int = 1024  # dim of each image patch token (0 disables the image path)
    cond_dim: int = 256  # must match TriPlaneConfig.cond_dim
    num_heads: int = 8  # heads for image-token attention pooling
    hidden: int = 512


class AttentionPool(nn.Module):
    """Pool a variable-length token sequence into one vector via a learned query."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.norm(pooled.squeeze(1))  # (B, dim)


class Lift2Dto3D(nn.Module):
    """Adapter: 2D SDX conditioning -> tri-plane conditioning vector."""

    def __init__(self, cfg: BridgeConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BridgeConfig()
        self.text_proj = nn.Linear(self.cfg.text_dim, self.cfg.hidden)
        self.use_image = self.cfg.image_dim > 0
        if self.use_image:
            self.image_pool = AttentionPool(self.cfg.image_dim, self.cfg.num_heads)
            self.image_proj = nn.Linear(self.cfg.image_dim, self.cfg.hidden)
        self.mix = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden, self.cfg.cond_dim),
        )

    def forward(
        self,
        text_embed: Tensor,
        image_tokens: Tensor | None = None,
        image_token_mask: Tensor | None = None,
    ) -> Tensor:
        """Return a conditioning vector ``(B, cond_dim)``.

        Args:
            text_embed: ``(B, text_dim)`` pooled text embedding.
            image_tokens: optional ``(B, L, image_dim)`` patch tokens for image->3D.
            image_token_mask: optional ``(B, L)`` bool mask, ``True`` where a token
                is padding and should be ignored by the pool.
        """
        h = self.text_proj(text_embed)
        if image_tokens is not None:
            if not self.use_image:
                raise ValueError("bridge built with image_dim=0 but image_tokens were passed")
            pooled = self.image_pool(image_tokens, key_padding_mask=image_token_mask)
            h = h + self.image_proj(pooled)
        return self.mix(h)
