"""Tests for reference token projection, latent blur, and DiT_Text reference_tokens path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_gaussian_blur_latent_shape() -> None:
    from diffusion.sampling_utils import gaussian_blur_latent

    x = torch.randn(2, 4, 8, 8)
    y = gaussian_blur_latent(x, 0.0)
    assert y.shape == x.shape
    z = gaussian_blur_latent(x, 0.5)
    assert z.shape == x.shape


def test_reference_token_projector() -> None:
    from models.reference_token_projection import ReferenceTokenProjector

    p = ReferenceTokenProjector(768, 1152, num_tokens=4)
    emb = torch.randn(3, 768)
    out = p(emb)
    assert out.shape == (3, 4, 1152)


def test_dit_text_accepts_reference_tokens() -> None:
    from models.dit_text import DiT_Text

    m = DiT_Text(
        input_size=8,
        patch_size=2,
        in_channels=4,
        depth=2,
        hidden_size=768,
        num_heads=12,
        text_dim=4096,
        learn_sigma=False,
    )
    b = 1
    x = torch.randn(b, 4, 8, 8)
    t = torch.zeros(b, dtype=torch.long)
    enc = torch.randn(b, 16, 4096)
    ref = torch.randn(b, 2, 768)
    with torch.no_grad():
        y0 = m(x, t, encoder_hidden_states=enc)
        y1 = m(x, t, encoder_hidden_states=enc, reference_tokens=ref, reference_scale=0.5)
    assert y0.shape == y1.shape == (b, 4, 8, 8)
