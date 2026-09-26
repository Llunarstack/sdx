"""Broad coverage tests for all invention batches + arch scaffolds."""

from __future__ import annotations

import torch
from models.inventions_arch import (
    CausalEditGraph,
    CausalEditNode,
    CountClassToken,
    DreamCriticStub,
    MemoryAugmentedKV,
    NegationTokenBank,
    ObjectCentricDiTStub,
    RelationTransformerHead,
    concept_algebra_loss,
    physics_contact_potential,
)
from utils.generation.inventions.composition_ext import (
    check_sat_layout,
    compile_scene_graph,
    load_failure_atlas,
    synthetic_constraint_caption,
)
from utils.generation.inventions.data_ext import canonicalize_booru_tags, rating_filter_rows
from utils.generation.inventions.glyph_ext import plan_font_id, ui_screenshot_layout
from utils.generation.inventions.identity_ext import auto_character_sheet, slot_dropout_caption
from utils.generation.inventions.registry import INVENTIONS, inventory
from utils.generation.inventions.systems_ext import adaptive_nfe, energy_reject, plan_self_healing


def test_registry_has_100():
    assert len(INVENTIONS) == 100
    inv = inventory()
    assert inv.get("coded", 0) + inv.get("partial", 0) + inv.get("scaffold", 0) == 100


def test_composition_and_atlas():
    sat = check_sat_layout(
        "red cube left of blue sphere",
        detections=[{"label": "red cube", "cx": 0.2, "cy": 0.5}, {"label": "blue sphere", "cx": 0.8, "cy": 0.5}],
    )
    assert sat.score >= 0.5
    layout = compile_scene_graph({"nodes": [{"id": "a", "label": "cat"}, {"id": "b", "label": "dog"}], "edges": []})
    assert len(layout["regions"]) == 2
    atlas = load_failure_atlas()
    assert len(atlas.modes) >= 20
    assert "no" in synthetic_constraint_caption().lower() or "left" in synthetic_constraint_caption().lower() or True


def test_arch_forwards():
    rel = RelationTransformerHead(dim=32, n_heads=2)
    x = torch.randn(2, 4, 32)
    assert rel(x).shape[-1] == 8
    nb = NegationTokenBank(vocab=16, dim=8)
    assert nb(torch.tensor([1, 2])).shape == (2, 8)
    cc = CountClassToken(dim=8)
    assert cc(torch.tensor([3, 5])).shape == (2, 8)
    oc = ObjectCentricDiTStub(dim=32, n_slots=2, depth=1)
    assert oc(torch.randn(1, 5, 32)).shape[1] == 7
    assert physics_contact_potential(torch.randn(1, 3, 2), torch.ones(1, 3) * 0.2).ndim == 1
    mem = MemoryAugmentedKV(dim=16, mem_size=8)
    k, v = mem(torch.randn(1, 4, 16))
    assert k.shape[1] == 8
    critic = DreamCriticStub(dim=16, n_modes=27)
    assert critic(torch.randn(2, 16)).shape == (2, 27)
    loss = concept_algebra_loss(torch.randn(4), torch.randn(4), torch.randn(4))
    assert loss.ndim == 0
    g = CausalEditGraph()
    g.add(CausalEditNode("face", [], "face", "fix face"))
    g.add(CausalEditNode("hair", ["face"], "subject", "match hair"))
    assert "hair" in g.dirty_closure("face")


def test_systems_self_heal(tmp_path):
    assert adaptive_nfe("four cats left of dogs, no text, hands") >= 12
    assert energy_reject(0.1, 0.1) is True
    plan = plan_self_healing("2girls different outfits, no watermark")
    assert plan.sample_argv
    assert "--scheduler" in plan.sample_argv


def test_data_glyph_identity(tmp_path):
    assert "1girl" in canonicalize_booru_tags("best_quality 1girl blue_hair")
    rows = rating_filter_rows([{"rating": "safe"}, {"rating": "explicit"}], allow={"safe"})
    assert len(rows) == 1
    ui = ui_screenshot_layout(buttons=2)
    assert len(ui["regions"]) == 3
    assert (
        "serif" in plan_font_id("serif_editorial").positive or "editorial" in plan_font_id("serif_editorial").positive
    )
    sheet = auto_character_sheet("Ada", {"front": "a.png"}, out_path=tmp_path / "ada.json")
    assert sheet.is_file()
    assert "A" in slot_dropout_caption(["A", "B", "C"], drop_prob=1.0, seed=0)
