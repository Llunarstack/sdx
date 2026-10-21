"""Tests for the invention prompt passes, priors, oracle and self-healing plan."""

from __future__ import annotations

import json

from utils.generation.inventions.bindlock import plan_bindlock
from utils.generation.inventions.countgate import plan_countgate, plan_hydra_slots
from utils.generation.inventions.failure_oracle import (
    FAILURE_ATLAS,
    diagnose_failures,
    repair_plan_from_failures,
    save_failure_atlas_jsonl,
)
from utils.generation.inventions.negatron import apply_negatron
from utils.generation.inventions.priors import apply_anatomon_prompts, apply_friction_to_prompts, scene_priors
from utils.generation.inventions.stack import apply_invention_stack
from utils.generation.inventions.wire import adaptive_nfe, plan_self_healing


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


def test_friction_adds_texture_language():
    p, n = apply_friction_to_prompts("portrait", "")
    assert "pores" in p.lower() and "plastic" in n.lower()
    assert apply_friction_to_prompts("portrait", "", strength=0.0) == ("portrait", "")


def test_scene_priors_follow_subject_and_medium():
    photo = " | ".join(p for p, _ in scene_priors("photo of a woman holding a cup"))
    assert "pores" in photo and "five fingers" in photo and "lens" in photo
    anime = " | ".join(p for p, _ in scene_priors("anime illustration of a castle at dusk"))
    assert "pores" not in anime and "lens" not in anime and "fingers" not in anime
    assert "z-order" in " | ".join(p for p, _ in scene_priors("a cat in front of a house"))


def test_oracle_and_stack():
    rep = repair_plan_from_failures(diagnose_failures("three red apples left of two blue cups, no text, 1girl hands"))
    assert rep.risks
    assert rep.repairs
    res = apply_invention_stack("four red cubes left of blue spheres, no watermark, 1girl", enable="auto")
    assert "exactly" in res.positive.lower()
    assert res.box_layout and res.box_layout["regions"]
    assert res.reports["priors"]
    hy = plan_hydra_slots("4 cats")
    assert len(hy.slots) == 4 and len(hy.box_layout["regions"]) == 4
    pos, neg, ap = apply_anatomon_prompts("1girl standing, hands visible")
    assert ap.active
    assert "finger" in neg.lower()


def test_failure_atlas_jsonl(tmp_path):
    path = save_failure_atlas_jsonl(tmp_path / "atlas.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == sum(len(prompts) for _, _, prompts in FAILURE_ATLAS)
    assert {r["mode"] for r in rows} == {mode for mode, _, _ in FAILURE_ATLAS}


def test_adaptive_nfe_and_self_heal():
    assert adaptive_nfe("four cats left of dogs, no text, hands") > adaptive_nfe("a cat") >= 12
    plan = plan_self_healing("2girls different outfits, no watermark")
    assert plan.sample_argv[plan.sample_argv.index("--scheduler") + 1] == "apex"
    assert plan.sample_argv[-2:] == ["--invention-stack", "off"]
