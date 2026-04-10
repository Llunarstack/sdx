"""Light tests for ``research/`` prototypes (CPU-only)."""

from __future__ import annotations

import torch
from research import (
    autoregressive_plans,
    diffusion_noise_structures,
    hybrid_sampling_schedules,
    latent_agreement,
    quality_timestep_weights,
)


def test_merge_vit_and_diffusion_weights():
    out = quality_timestep_weights.merge_vit_and_diffusion_weights([1.0, 3.0], [2.0, 2.0], alpha=0.5)
    assert len(out) == 2
    assert abs(sum(out) - 1.0) < 1e-5


def test_revisit_schedule_length():
    base = autoregressive_plans.raster_block_order(2)
    seq = autoregressive_plans.revisit_schedule(base, cycles=3, stride=1)
    assert len(seq) == 3 * len(base)


def test_split_budget_geometric():
    b = hybrid_sampling_schedules.split_budget_geometric(20, ar_fraction=0.25, min_ar=2, min_diffusion=4)
    assert b.total == 20
    assert b.ar_refine_steps >= 2
    assert b.diffusion_steps >= 4


def test_interleave_phases():
    b = hybrid_sampling_schedules.HybridStepBudget(3, 5)
    tags = hybrid_sampling_schedules.interleave_phases(b, chunk_ar=2, chunk_diffusion=2)
    assert tags[:3] == ["ar", "ar", "ar"]
    assert tags.count("diff") == 5


def test_orthogonal_noise_directions():
    q = diffusion_noise_structures.orthogonal_noise_directions((4, 4), k=3)
    assert q.shape == (3, 4, 4)
    flat = q.reshape(3, -1)
    gram = flat @ flat.T
    eye = torch.eye(3, dtype=gram.dtype)
    assert torch.allclose(gram, eye, atol=1e-4, rtol=1e-3)


def test_weighted_ensemble_latents():
    a = torch.ones(2, 3)
    b = torch.zeros(2, 3)
    m = diffusion_noise_structures.weighted_ensemble_latents([a, b], [1.0, 1.0], dim=0)
    assert m.shape == (2, 3)
    assert torch.allclose(m, torch.full((2, 3), 0.5))


def test_agreement_loss_cosine():
    x = torch.randn(2, 4, 4)
    assert latent_agreement.agreement_loss_cosine(x, x).item() < 1e-5
