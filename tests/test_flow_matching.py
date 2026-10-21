"""Rectified-flow training objective: time sampling, resolution shift, per-sample loss (image and video)."""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusion.flow_matching import (
    flow_matching_per_sample_losses,
    sample_flow_time,
    shift_timestep_indices,
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


def test_flow_loss_2d():
    torch.manual_seed(0)
    model = _TinyDenoiser(4)
    x0 = torch.randn(3, 4, 8, 8)
    eps = torch.randn(3, 4, 8, 8)
    loss = flow_matching_per_sample_losses(model, x0, eps, 1000, {}, logit_normal=True)
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.scale.grad is not None


def test_flow_loss_video_shape():
    # Same objective must work for 5D video latents (B, C, T, H, W).
    model = _TinyDenoiser(4)
    x0 = torch.randn(2, 4, 5, 8, 8)
    eps = torch.randn(2, 4, 5, 8, 8)
    loss = flow_matching_per_sample_losses(model, x0, eps, 1000, {}, logit_normal=True, shift=3.0)
    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()


def test_uniform_default_works():
    model = _TinyDenoiser(4)
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn(2, 4, 8, 8)
    loss = flow_matching_per_sample_losses(model, x0, eps, 1000, {})
    assert loss.shape == (2,)


def test_shift_timestep_indices_matches_sampled_shift():
    t = torch.tensor([0, 250, 500, 750, 999])
    assert torch.equal(shift_timestep_indices(t, 1000, 1.0), t)
    shifted = shift_timestep_indices(t, 1000, 3.0)
    assert shifted[0] == 0 and shifted[-1] == 999
    assert (shifted[1:-1] > t[1:-1]).all()  # shift > 1 moves toward noise


def test_given_t_idx_is_used():
    class _Probe(torch.nn.Module):
        def forward(self, x, t, **kw):
            self.seen = t
            return torch.zeros_like(x)

    model = _Probe()
    t = torch.tensor([3, 600])
    flow_matching_per_sample_losses(model, torch.zeros(2, 4, 4, 4), torch.ones(2, 4, 4, 4), 1000, {}, t_idx=t)
    assert torch.equal(model.seen, t)
