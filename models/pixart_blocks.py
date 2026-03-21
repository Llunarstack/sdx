# Building blocks ported from PixArt-alpha / PixArt-sigma (external/PixArt-*).
# Optional: SizeEmbedder for multi-resolution (h, w) conditioning; use when training with variable aspect/size.
# KV compression and full PixArt blocks can be wired in later (see IMPROVEMENTS.md).
import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    """AdaLN: x * (1 + scale) + shift. shift/scale can be (B, D) applied to (B, N, D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SizeEmbedder(nn.Module):
    """
    Embed (height, width) or other size scalars into a vector (PixArt-style).
    Input s: (B, num_dims) e.g. (B, 2) for (h, w). Output: (B, hidden_size * num_dims) or (B, hidden_size) if concat=False.
    Used for multi-resolution / variable-aspect conditioning.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, concat_dims: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self.concat_dims = concat_dims

    @staticmethod
    def sinusoidal_embed(x: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=x.device) / half)
        args = x.unsqueeze(-1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb

    def forward(self, s: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        s: (B, num_dims) or (num_dims,) for single size. Values are e.g. height, width (in pixels or latent size).
        batch_size: target batch size (if s is (1, 2), we expand to (batch_size, 2)).
        """
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if s.shape[0] != batch_size:
            s = s.repeat((batch_size + s.shape[0] - 1) // s.shape[0], 1)[:batch_size]
        b, num_dims = s.shape[0], s.shape[1]
        s_flat = s.reshape(-1)
        s_freq = self.sinusoidal_embed(s_flat, self.frequency_embedding_size)
        s_emb = self.mlp(s_freq)
        s_emb = s_emb.reshape(b, num_dims * self.hidden_size)
        if not self.concat_dims:
            s_emb = s_emb.mean(dim=1, keepdim=True).expand(-1, self.hidden_size)
        return s_emb


class ZeroInitPatchChannelGate(nn.Module):
    """
    Lightweight channel-wise gate on patch tokens (SE-style).
    Last layer is zero-initialized so forward starts as identity: x + x * tanh(0) = x.
    Helps the model learn resolution- and content-dependent channel scaling without
    disturbing a pretrained checkpoint at step 0.
    """

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        mid = max(dim // max(1, reduction), 32)
        self.norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.fc1 = nn.Linear(dim, mid)
        self.fc2 = nn.Linear(mid, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        g = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x.mean(dim=1)))))
        return x + x * torch.tanh(g).unsqueeze(1)
