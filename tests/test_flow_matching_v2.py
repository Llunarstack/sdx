"""Tests for the SD3/Flux-style flow-matching upgrade (2D image gen)."""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusion.flow_matching import (
    flow_matching_per_sample_losses,
    flow_matching_per_sample_losses_v2,
    sample_flow_time,
)


class _TinyDenoiser(nn.Module):
    """Maps (x_t, t) -> same-shaped velocity. Rank-agnostic (2D or video)."""

    def __init__(self, ch: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(ch))

    def forward(self, x, t, **kw):
        shape = [1, -1] + [1] * (x.dim() - 2)
        return x * self.scale.view(*shape)


def test_logit_normal_concentrates_in_the_middle():
    torch.manual_seed(0)
    s = sample_flow_time(20000, logit_normal=True, ln_mean=0.0, ln_std=1.0)
    assert 0.45 < float(s.mean()) < 0.55  # centered on 0.5
    mid_frac = float(((s > 0.25) & (s < 0.75)).float().mean())
    # Uniform would give 0.5 in that band; logit-normal puts more mass there.
    assert mid_frac > 0.55


def test_uniform_mode_matches_uniform():
    torch.manual_seed(0)
    s = sample_flow_time(20000, logit_normal=False)
    assert 0.47 < float(s.mean()) < 0.53
    mid_frac = float(((s > 0.25) & (s < 0.75)).float().mean())
    assert 0.47 < mid_frac < 0.53  # ~0.5, i.e. flat


def test_resolution_shift_pushes_toward_noise():
    torch.manual_seed(0)
    base = sample_flow_time(20000, logit_normal=True, shift=1.0)
    shifted = sample_flow_time(20000, logit_normal=True, shift=3.0)
    assert float(shifted.mean()) > float(base.mean())  # shift>1 raises s


def test_flow_v2_loss_2d():
    torch.manual_seed(0)
    model = _TinyDenoiser(4)
    x0 = torch.randn(3, 4, 8, 8)
    eps = torch.randn(3, 4, 8, 8)
    loss = flow_matching_per_sample_losses_v2(model, x0, eps, 1000, {})
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.scale.grad is not None


def test_flow_v2_loss_video_shape():
    # Same objective must work for 5D video latents (B, C, T, H, W).
    model = _TinyDenoiser(4)
    x0 = torch.randn(2, 4, 5, 8, 8)
    eps = torch.randn(2, 4, 5, 8, 8)
    loss = flow_matching_per_sample_losses_v2(model, x0, eps, 1000, {}, shift=3.0)
    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()


def test_v1_still_works():
    model = _TinyDenoiser(4)
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn(2, 4, 8, 8)
    loss = flow_matching_per_sample_losses(model, x0, eps, 1000, {})
    assert loss.shape == (2,)
