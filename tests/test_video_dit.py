"""Tests for the factorized spatiotemporal video DiT (STDiT)."""

from __future__ import annotations

import torch
from diffusion.flow_matching import flow_matching_per_sample_losses_v2
from models.video_dit import FactorizedSpatioTemporalBlock, VideoDiT


def _active_block(dim=32, heads=4, seed=0):
    """A block with non-zero adaLN so its attention actually modulates (adaLN-Zero
    starts as identity, which is correct for training but hides the mechanism).

    Perturbations in these tests use *pattern-changing* noise, not a uniform shift:
    the block's LayerNorm removes any constant offset, so `x += 1.0` would be a
    no-op and wouldn't test propagation.
    """
    torch.manual_seed(seed)
    blk = FactorizedSpatioTemporalBlock(dim, heads)
    torch.nn.init.normal_(blk.ada.weight, std=0.3)
    torch.nn.init.normal_(blk.ada.bias, std=0.3)
    return blk


def test_block_preserves_shape():
    blk = _active_block()
    x = torch.randn(2, 4, 6, 32)  # (B, T, N, D)
    cond = torch.randn(2, 32)
    assert blk(x, cond).shape == x.shape


def test_temporal_attention_mixes_across_frames():
    blk = _active_block()
    x = torch.randn(1, 4, 6, 32)
    cond = torch.randn(1, 32)
    out = blk(x, cond)
    x2 = x.clone()
    torch.manual_seed(1)
    x2[:, -1] += torch.randn(1, 6, 32)  # pattern-changing perturbation of the last frame
    out2 = blk(x2, cond)
    # A change in the last frame should reach earlier frames (non-causal temporal).
    assert not torch.allclose(out[:, 0], out2[:, 0], atol=1e-5)


def test_causal_temporal_hides_the_future():
    blk = _active_block()
    x = torch.randn(1, 4, 6, 32)
    cond = torch.randn(1, 32)
    out = blk(x, cond, causal_temporal=True)
    x2 = x.clone()
    torch.manual_seed(1)
    x2[:, -1] += torch.randn(1, 6, 32)  # perturb the LAST frame
    out2 = blk(x2, cond, causal_temporal=True)
    # Frame 0 must not see the future...
    assert torch.allclose(out[:, 0], out2[:, 0], atol=1e-5)
    # ...but the last frame itself changes.
    assert not torch.allclose(out[:, -1], out2[:, -1], atol=1e-5)


def test_spatial_attention_mixes_within_frame():
    blk = _active_block()
    x = torch.randn(1, 3, 6, 32)
    cond = torch.randn(1, 32)
    out = blk(x, cond, causal_temporal=True)  # isolate spatial by removing temporal future leak
    x2 = x.clone()
    torch.manual_seed(2)
    x2[:, 0, 0] += torch.randn(32)  # perturb one spatial token in frame 0
    out2 = blk(x2, cond, causal_temporal=True)
    # Other spatial tokens in frame 0 should be affected.
    assert not torch.allclose(out[:, 0, 1], out2[:, 0, 1], atol=1e-5)


def test_conditioning_changes_output():
    blk = _active_block()
    x = torch.randn(1, 3, 6, 32)
    a = blk(x, torch.randn(1, 32))
    b = blk(x, torch.randn(1, 32))
    assert not torch.allclose(a, b)


def test_video_dit_forward_shape_and_grad():
    torch.manual_seed(0)
    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2)
    x = torch.randn(2, 4, 5, 16, 16)  # (B, C, T, H, W)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == x.shape
    named = dict(model.named_parameters())
    assert "_spatial_pos" in named and "_temporal_pos" in named
    out.mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_video_dit_accepts_text_context():
    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, context_dim=64)
    x = torch.randn(1, 4, 4, 16, 16)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t, context=torch.randn(1, 64))
    assert out.shape == x.shape


def test_video_dit_trains_with_flow_matching_v2():
    """The video backbone plugs straight into the upgraded flow-matching loss."""
    torch.manual_seed(0)
    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2)
    x0 = torch.randn(2, 4, 4, 16, 16)
    eps = torch.randn(2, 4, 4, 16, 16)
    loss = flow_matching_per_sample_losses_v2(model, x0, eps, 1000, {}, shift=3.0)
    assert loss.shape == (2,)
    loss.mean().backward()
    assert any(p.grad is not None for p in model.parameters())
