from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class NativeMultimodalTransformer(nn.Module):
    """
    Lightweight native multimodal token fusion block for SDX experiments.

    Inputs:
      - vision_tokens: (B, Nv, Dv)
      - text_tokens: (B, Nt, Dt)
      - optional extra_tokens: (B, Ne, De)
    Output:
      - fused tokens: (B, Nv, D_model)
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        model_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        extra_dim: int = 0,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.vision_proj = nn.Linear(int(vision_dim), self.model_dim)
        self.text_proj = nn.Linear(int(text_dim), self.model_dim)
        self.extra_proj = nn.Linear(int(extra_dim), self.model_dim) if int(extra_dim) > 0 else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.model_dim * float(mlp_ratio)),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(self.model_dim)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        *,
        extra_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        v = self.vision_proj(vision_tokens)
        t = self.text_proj(text_tokens)
        seq = [v, t]
        if extra_tokens is not None and self.extra_proj is not None:
            seq.append(self.extra_proj(extra_tokens))
        x = torch.cat(seq, dim=1)
        y = self.out_norm(self.encoder(x))
        # Return only visual slice as primary fused stream.
        return {
            "fused_vision_tokens": y[:, : v.shape[1], :],
            "fused_all_tokens": y,
        }

