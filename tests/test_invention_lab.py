"""Tests for Invention Lab modules."""

from __future__ import annotations

import torch
from utils.generation.inventions.anatomon import apply_anatomon_prompts
from utils.generation.inventions.bindlock import plan_bindlock
from utils.generation.inventions.consensus_denoise import blend_consensus_latents, plan_consensus_seeds
from utils.generation.inventions.countgate import plan_countgate
from utils.generation.inventions.failure_oracle import diagnose_failures, repair_plan_from_failures
from utils.generation.inventions.friction_texture import apply_friction_to_prompts, friction_noise_scale
from utils.generation.inventions.hydra_slots import plan_hydra_slots
from utils.generation.inventions.negatron import apply_negatron
from utils.generation.inventions.spectra_routing import spectra_cfg_scale, spectra_mix_latent
from utils.generation.inventions.stack import apply_invention_stack


def test_bindlock_and_countgate():
    bl = plan_bindlock("a red cube left of a blue sphere")
    assert bl.triples
    assert "attribute swap" in bl.negative_addon
    cg = plan_countgate("four cats and 2girls")
    assert any(s.n == 4 for s in cg.specs) or any(s.n == 2 for s in cg.specs)
    assert cg.boxes


def test_negatron_strips_and_negates():
    pos, neg, plan = apply_negatron("1girl, no hat, without watermark, smiling")
    assert "hat" in plan.forbidden or any("hat" in f for f in plan.forbidden)
    assert "hat" in neg.lower()
    assert "no hat" not in pos.lower()


def test_spectra_and_friction_and_consensus():
    assert spectra_cfg_scale(7.5, 0.1) > 7.5
    x = torch.randn(1, 4, 8, 8)
    y = spectra_mix_latent(x, 0.8)
    assert y.shape == x.shape
    p, n = apply_friction_to_prompts("portrait", "")
    assert "plastic" in n.lower() or "pore" in p.lower()
    assert friction_noise_scale(0.9) > 0
    a, b = plan_consensus_seeds()
    assert a != b
    xa = torch.randn(1, 4, 4, 4)
    xb = xa + 0.01
    out = blend_consensus_latents(xa, xb, progress=0.1)
    assert out.shape == xa.shape


def test_oracle_and_stack():
    rep = repair_plan_from_failures(diagnose_failures("three red apples left of two blue cups, no text, 1girl hands"))
    assert rep.risks
    assert rep.repairs
    res = apply_invention_stack(
        "four red cubes left of blue spheres, no watermark, 1girl",
        enable="auto",
    )
    assert "exactly" in res.positive.lower() or "count" in res.positive.lower() or res.reports
    hy = plan_hydra_slots("4 cats")
    assert len(hy.slots) == 4
    pos, neg, ap = apply_anatomon_prompts("1girl standing, hands visible")
    assert ap.active
    assert "finger" in neg.lower() or "anatomy" in neg.lower()
