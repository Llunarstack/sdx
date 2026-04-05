"""
Dynamic Patch Scheduling for Diffusion Transformers (arxiv 2602.16968).

Key insight: early denoising timesteps only need coarse global structure,
late timesteps need fine local detail. Using large patches early and small
patches late reduces FLOPs by ~40% with no quality loss.

This module provides:
  - DynamicPatchEmbed: replaces PatchEmbed, selects patch size per timestep.
  - TimestepPatchScheduler: maps timestep -> patch_size.
  - merge_patches / unmerge_patches: token-level up/downsampling helpers.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepPatchScheduler:
    """
    Maps a normalized timestep t in [0, 1] (0=clean, 1=noisy) to a patch size.

    Strategy:
      - High noise (t > high_thresh): use coarse_size (e.g. 16)
      - Low noise  (t < low_thresh):  use fine_size   (e.g. 4)
      - Middle:                        use base_size   (e.g. 8)
    """

    def __init__(
        self,
        fine_size: int = 4,
        base_size: int = 8,
        coarse_size: int = 16,
        high_thresh: float = 0.7,
        low_thresh: float = 0.3,
    ):
        self.fine_size = int(fine_size)
        self.base_size = int(base_size)
        self.coarse_size = int(coarse_size)
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)

    def get_patch_size(self, t_normalized: float) -> int:
        """t_normalized: float in [0, 1], 1 = fully noisy."""
        if t_normalized >= self.high_thresh:
            return self.coarse_size
        if t_normalized <= self.low_thresh:
            return self.fine_size
        return self.base_size

    def get_patch_size_batch(self, timesteps: torch.Tensor, max_t: int = 1000) -> List[int]:
        """timesteps: (B,) integer timesteps -> list of patch sizes per sample."""
        t_norm = (timesteps.float() / max_t).clamp(0.0, 1.0)
        return [self.get_patch_size(float(t)) for t in t_norm]


class DynamicPatchEmbed(nn.Module):
    """
    Multi-resolution patch embedding that switches kernel size based on timestep.

    Maintains separate conv weights for each supported patch size so the model
    can be trained end-to-end. At inference, only the active size is used.

    Args:
        in_channels: Latent channels (e.g. 4 for SD latents).
        hidden_size: Transformer hidden dimension.
        patch_sizes: Tuple of supported patch sizes (fine -> coarse).
        img_size: Expected spatial size (used for position embedding init).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 1152,
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
        img_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_sizes = tuple(patch_sizes)
        self.img_size = img_size

        # One projection per patch size
        self.projs = nn.ModuleDict(
            {
                str(p): nn.Conv2d(in_channels, hidden_size, kernel_size=p, stride=p, bias=False)
                for p in patch_sizes
            }
        )

        # Learnable position embeddings for each resolution
        self.pos_embeds = nn.ParameterDict(
            {
                str(p): nn.Parameter(
                    torch.zeros(1, (img_size // p) ** 2, hidden_size)
                )
                for p in patch_sizes
            }
        )
        self._init_weights()

    def _init_weights(self):
        for p_str, proj in self.projs.items():
            nn.init.xavier_uniform_(proj.weight.view(proj.weight.size(0), -1))
        for p_str, pe in self.pos_embeds.items():
            nn.init.trunc_normal_(pe, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        patch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: (B, C, H, W) latent.
            patch_size: Which patch size to use. Defaults to base (middle) size.
        Returns:
            tokens: (B, N, D)
            patch_size: int (the one actually used)
        """
        if patch_size is None:
            patch_size = self.patch_sizes[len(self.patch_sizes) // 2]
        p_str = str(patch_size)
        if p_str not in self.projs:
            # Fallback to nearest available
            patch_size = min(self.patch_sizes, key=lambda p: abs(p - patch_size))
            p_str = str(patch_size)

        tokens = self.projs[p_str](x)  # (B, D, H/p, W/p)
        B, D, h, w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)

        # Interpolate position embedding if spatial size differs from init
        pe = self.pos_embeds[p_str]
        n_init = pe.shape[1]
        n_actual = h * w
        if n_actual != n_init:
            s_init = int(math.isqrt(n_init))
            pe = pe.reshape(1, s_init, s_init, D).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
            pe = pe.permute(0, 2, 3, 1).reshape(1, n_actual, D)

        tokens = tokens + pe
        return tokens, patch_size


def merge_to_coarse(tokens: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Average-pool adjacent tokens to reduce sequence length by `factor^2`.
    tokens: (B, N, D) where N = h*w (square assumed).
    Returns: (B, N // factor^2, D)
    """
    B, N, D = tokens.shape
    h = w = int(math.isqrt(N))
    assert h * w == N, "merge_to_coarse requires square token grid"
    x = tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, h, w)
    x = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    return x.permute(0, 2, 3, 1).reshape(B, -1, D)


def unmerge_to_fine(tokens: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Bilinear upsample tokens to increase sequence length by `factor^2`.
    tokens: (B, N, D) where N = h*w (square assumed).
    Returns: (B, N * factor^2, D)
    """
    B, N, D = tokens.shape
    h = w = int(math.isqrt(N))
    assert h * w == N, "unmerge_to_fine requires square token grid"
    x = tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)
    x = F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(B, -1, D)


__all__ = [
    "TimestepPatchScheduler",
    "DynamicPatchEmbed",
    "merge_to_coarse",
    "unmerge_to_fine",
]
