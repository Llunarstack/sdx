"""
Anatomy-aware attention and training-free spatial priors for figure generation.

``AnatomyAwareAttention`` is the learned module (EnhancedDiT / optional DiT hook).
``apply_anatomy_spatial_prior`` is training-free and safe at random init for sample-time use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_anatomy_spatial_prior(
    x: torch.Tensor,
    h_patches: int,
    w_patches: int,
    *,
    strength: float = 0.2,
) -> torch.Tensor:
    """
    Training-free residual that:
      - boosts local contrast in typical hand/limb bands (lower third)
      - adds mild left–right asymmetry in the face band (upper half)
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0 or x.ndim != 3 or h_patches < 2 or w_patches < 2:
        return x
    b, n, d = x.shape
    target = int(h_patches) * int(w_patches)
    tok = x[:, : min(n, target), :]
    if tok.shape[1] < target:
        pad = tok.new_zeros(b, target - tok.shape[1], d)
        tok = torch.cat([tok, pad], dim=1)
    spat = tok.transpose(1, 2).reshape(b, d, h_patches, w_patches)

    blur = F.avg_pool2d(spat, kernel_size=3, stride=1, padding=1)
    detail = spat - blur

    yy = torch.linspace(0.0, 1.0, h_patches, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
    # Hands / lower limbs
    hand_band = ((yy > 0.55) & (yy < 0.95)).to(dtype=x.dtype)
    # Face / head band asymmetry
    face_band = ((yy > 0.05) & (yy < 0.45)).to(dtype=x.dtype)
    flipped = torch.flip(spat, dims=[-1])
    asym = (spat - flipped) * face_band * 0.35

    residual = detail * hand_band * 0.55 + asym
    flat = residual.reshape(b, d, -1).transpose(1, 2)
    if n >= target:
        out = x.clone()
        out[:, :target, :] = out[:, :target, :] + s * flat
        return out
    return x + s * flat[:, :n, :]


class AnatomyAwareAttention(nn.Module):
    """Anatomy-aware attention for better human figure generation."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = int(hidden_size)
        heads = max(1, min(int(num_heads), hidden_size // 16 or 1))
        while hidden_size % heads != 0 and heads > 1:
            heads -= 1
        self.num_heads = heads

        self.anatomy_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.body_part_embeddings = nn.Parameter(torch.randn(10, hidden_size) * 0.02)
        self.anatomy_constraint = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        hand_heads = max(1, min(4, heads))
        while hidden_size % hand_heads != 0 and hand_heads > 1:
            hand_heads -= 1
        self.hand_attention = nn.MultiheadAttention(hidden_size, num_heads=hand_heads, batch_first=True)
        self.finger_embeddings = nn.Parameter(torch.randn(5, hidden_size) * 0.02)
        # Keep residual mild until trained
        self.residual_scale = 0.25

    def forward(self, x: torch.Tensor, anatomy_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            anatomy_mask: optional [B, N] human-region mask in [0, 1]
        """
        b, n, d = x.shape
        x_anatomy, _ = self.anatomy_attention(x, x, x)
        x_constrained = self.anatomy_constraint(x_anatomy)
        out = x + float(self.residual_scale) * x_constrained

        if anatomy_mask is None:
            return out

        hand_regions = anatomy_mask.to(device=x.device, dtype=x.dtype)
        if hand_regions.dim() == 2 and hand_regions.shape[1] != n:
            side = int(round(hand_regions.shape[1] ** 0.5))
            target = int(round(n**0.5))
            if side * side == hand_regions.shape[1] and target * target == n:
                hr = hand_regions.view(b, 1, side, side)
                hr = F.interpolate(hr, size=(target, target), mode="nearest")
                hand_regions = hr.view(b, -1)

        x_hands, _ = self.hand_attention(out, out, out)
        finger_features = self.finger_embeddings.unsqueeze(0).expand(b, -1, -1)
        finger_attended, _ = self.hand_attention(x_hands, finger_features, finger_features)
        hand_mask = hand_regions.unsqueeze(-1).expand(-1, -1, d)
        if hand_mask.shape[1] != n:
            return out
        return torch.where(hand_mask > 0.5, out + 0.3 * finger_attended, out)


__all__ = ["AnatomyAwareAttention", "apply_anatomy_spatial_prior"]
