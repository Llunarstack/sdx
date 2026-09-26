"""Tests for the 2D->3D conditioning bridge and end-to-end field wiring."""

from __future__ import annotations

import pytest
import torch
from frontier.multiview import BridgeConfig, Lift2Dto3D, SpatialLatentField, TriPlaneConfig

BCFG = BridgeConfig(text_dim=64, image_dim=48, cond_dim=32, num_heads=4, hidden=64)


@pytest.fixture
def bridge() -> Lift2Dto3D:
    torch.manual_seed(0)
    return Lift2Dto3D(BCFG)


def test_text_only_shape(bridge: Lift2Dto3D) -> None:
    cond = bridge(torch.randn(3, BCFG.text_dim))
    assert cond.shape == (3, BCFG.cond_dim)


def test_text_plus_image_shape(bridge: Lift2Dto3D) -> None:
    cond = bridge(torch.randn(2, BCFG.text_dim), torch.randn(2, 10, BCFG.image_dim))
    assert cond.shape == (2, BCFG.cond_dim)


def test_image_tokens_change_output(bridge: Lift2Dto3D) -> None:
    bridge.eval()
    text = torch.randn(1, BCFG.text_dim)
    a = bridge(text)
    b = bridge(text, torch.randn(1, 10, BCFG.image_dim))
    assert not torch.allclose(a, b)


def test_padding_mask_ignores_padded_tokens(bridge: Lift2Dto3D) -> None:
    bridge.eval()
    text = torch.randn(1, BCFG.text_dim)
    real = torch.randn(1, 4, BCFG.image_dim)
    padded = torch.cat([real, torch.randn(1, 3, BCFG.image_dim)], dim=1)
    mask = torch.tensor([[False, False, False, False, True, True, True]])
    out_real = bridge(text, real)
    out_masked = bridge(text, padded, image_token_mask=mask)
    assert torch.allclose(out_real, out_masked, atol=1e-5)


def test_image_disabled_rejects_tokens() -> None:
    b = Lift2Dto3D(BridgeConfig(text_dim=64, image_dim=0, cond_dim=32))
    with pytest.raises(ValueError):
        b(torch.randn(1, 64), torch.randn(1, 5, 48))


def test_end_to_end_text_to_field() -> None:
    """The bridge output must drive a real 3D field forward pass."""
    torch.manual_seed(1)
    bridge = Lift2Dto3D(BridgeConfig(text_dim=64, image_dim=0, cond_dim=32))
    field = SpatialLatentField(TriPlaneConfig(cond_dim=32, plane_channels=8, grid_res=16, feature_dim=16))

    cond = bridge(torch.randn(2, 64))
    planes = field.encode(cond)
    pts = torch.empty(2, 20, 3).uniform_(-1, 1)
    sdf = field.decode_sdf(planes, pts)
    assert sdf.shape == (2, 20, 1)

    # gradient flows from the 3D field all the way back into the 2D bridge
    sdf.mean().backward()
    assert any(p.grad is not None for p in bridge.parameters())


def test_field_to_mesh_export() -> None:
    """Full path: conditioning -> field -> watertight mesh with vertex colors."""
    torch.manual_seed(2)
    field = SpatialLatentField(TriPlaneConfig(cond_dim=16, plane_channels=8, grid_res=16, feature_dim=16))
    planes = field.encode(torch.randn(1, 16))
    verts, faces, colors = field.to_mesh(planes, resolution=20, with_color=True)
    # An untrained field may or may not cross zero; assert the contract holds either way.
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    if len(verts):
        assert colors.shape == verts.shape
        assert faces.max() < len(verts)
