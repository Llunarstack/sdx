"""Lightweight tests for stable inference hook helpers."""

from __future__ import annotations

import torch
from utils.generation.inference_research_hooks import TemporalHarmonizerStub


def test_temporal_harmonizer_stub_identity_without_prev() -> None:
    stub = TemporalHarmonizerStub()
    x = torch.randn(2, 4, 8, 8)
    out = stub.condition_latent(x, None, alpha_prev=0.0)
    assert torch.equal(out, x)


def test_temporal_harmonizer_stub_blends_prev() -> None:
    stub = TemporalHarmonizerStub()
    x = torch.ones(1, 4, 4, 4)
    prev = torch.zeros(1, 4, 4, 4)
    out = stub.condition_latent(x, prev, alpha_prev=0.5)
    assert torch.allclose(out, 0.5 * x + 0.5 * prev)
