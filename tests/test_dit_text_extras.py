"""Smoke tests for DiT-Text size conditioning and patch SE."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dit_text import DiT_Text


def test_dit_text_size_embed_and_patch_se():
    m = DiT_Text(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        text_dim=64,
        learn_sigma=False,
        class_dropout_prob=0.0,
        size_embed_dim=128,
        patch_se=True,
        patch_se_reduction=4,
    )
    m.eval()
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 100, (2,))
    enc = torch.randn(2, 16, 64)
    with torch.no_grad():
        y = m(x, t, encoder_hidden_states=enc)
    assert y.shape == (2, 4, 8, 8)

    # Explicit size_embed row
    sz = torch.tensor([[8.0, 8.0], [8.0, 8.0]])
    with torch.no_grad():
        y2 = m(x, t, encoder_hidden_states=enc, size_embed=sz)
    assert y2.shape == y.shape


def test_dit_text_patch_se_only():
    m = DiT_Text(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=128,
        depth=1,
        num_heads=4,
        text_dim=64,
        learn_sigma=False,
        class_dropout_prob=0.0,
        patch_se=True,
    )
    m.eval()
    x = torch.randn(1, 4, 8, 8)
    t = torch.tensor([0])
    enc = torch.randn(1, 8, 64)
    with torch.no_grad():
        y = m(x, t, encoder_hidden_states=enc)
    assert y.shape == (1, 4, 8, 8)
