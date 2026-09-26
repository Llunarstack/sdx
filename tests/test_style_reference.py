"""Tests for InstantStyle / B-LoRA style reference helpers."""

from __future__ import annotations

import torch
from models.reference_inject import (
    concat_reference_tokens,
    pick_block_text_emb,
    resolve_reference_block_start,
)


def test_resolve_reference_block_start_modes():
    assert resolve_reference_block_start(n_blocks=28, style_mode="full") == 0
    assert resolve_reference_block_start(n_blocks=28, style_mode="ip-adapter") == 0
    assert resolve_reference_block_start(n_blocks=28, style_mode="instantstyle") == (28 * 2) // 3
    assert resolve_reference_block_start(n_blocks=28, style_mode="style") == (28 * 2) // 3
    assert resolve_reference_block_start(n_blocks=28, style_mode="style_mid") == 14
    assert resolve_reference_block_start(n_blocks=28, style_mode="full", block_start=10) == 10
    assert resolve_reference_block_start(n_blocks=28, style_mode="instantstyle", block_start=5) == 5


def test_concat_and_pick_block_text_emb():
    text = torch.randn(2, 8, 16)
    ref = torch.ones(2, 4, 16)
    plain, with_ref = concat_reference_tokens(text, {"reference_tokens": ref, "reference_scale": 0.5})
    assert plain is text
    assert with_ref is not None
    assert with_ref.shape == (2, 12, 16)
    assert torch.allclose(with_ref[:, 8:, :], ref * 0.5)

    start = resolve_reference_block_start(n_blocks=9, style_mode="instantstyle")
    early = pick_block_text_emb(plain, with_ref, block_index=0, ref_block_start=start)
    late = pick_block_text_emb(plain, with_ref, block_index=start, ref_block_start=start)
    assert early.shape[1] == 8
    assert late.shape[1] == 12


def test_concat_disabled_when_scale_zero():
    text = torch.randn(1, 4, 8)
    ref = torch.randn(1, 2, 8)
    plain, with_ref = concat_reference_tokens(text, {"reference_tokens": ref, "reference_scale": 0.0})
    assert with_ref is None
    assert plain is text


def test_lora_train_layer_group_filter():
    import torch.nn as nn
    from models.lora_train import inject_trainable_lora

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                [nn.ModuleDict({"attn": nn.Linear(8, 8), "mlp": nn.Linear(8, 8)}) for _ in range(9)]
            )

    m = Tiny()
    # Name paths as blocks.i.attn / blocks.i.mlp via ModuleList indexing
    # inject looks at named_modules paths like blocks.0.attn
    params, n = inject_trainable_lora(m, rank=4, alpha=4.0, targets=("attn", "mlp"), layer_group="last")
    assert n > 0
    # last group matches models/lora.py: indices 4..8 for depth 0..8 → 5 blocks × 2 linears
    assert n == 10
    assert all(p.requires_grad for p in params)
