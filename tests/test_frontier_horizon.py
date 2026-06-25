"""Tests for horizon / ahead-of-curve frontier modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from frontier.adherence import TokenEmphasisPlanner
from frontier.blend import StyleDNABlender
from frontier.causality import PhysicalPlausibilityScanner
from frontier.counterfactual import PreserveEditPlanner
from frontier.economy import ComputeBudgetPlanner, GuidanceTier
from frontier.inverse import LayoutSketchInferer
from frontier.provenance import build_audit_bundle, write_audit_bundle
from frontier.registry import idea_by_id, list_ideas
from frontier.retrieval import StyleFactRetriever
from frontier.semantics import SceneRelationParser
from frontier.synthesis import analyze_deep, deep_sample_kwargs
from frontier.temporal import StoryboardPlanner
from frontier.uncertainty import ConfidenceGate
from frontier.world import CharacterRecord, WorldBible, WorldLock


class TestHorizonRegistry:
    def test_new_ideas_registered(self):
        assert idea_by_id("physical_plausibility").status == "implemented"
        assert idea_by_id("deep_frontier").status == "implemented"
        impl = list_ideas(status="implemented")
        assert len(impl) >= 20


class TestPhysicalPlausibility:
    def test_rain_missing_wet(self):
        flags = PhysicalPlausibilityScanner().scan("heavy rain on asphalt")
        assert flags
        assert flags[0].category == "weather"


class TestComputeBudget:
    def test_early_full_with_regions(self):
        plan = ComputeBudgetPlanner(num_steps=20).plan(risk_score=0.5, layout_regions=3)
        assert plan.tiers[0] == GuidanceTier.FULL
        assert plan.estimated_cost < 1.0


class TestWorldBible:
    def test_detect_and_lock(self):
        bible = WorldBible(
            characters={
                "mira": CharacterRecord(
                    id="mira",
                    display_name="Mira",
                    visual_tokens="silver hair, green coat",
                    negative_tokens="blonde hair",
                    tags=["hero"],
                )
            }
        )
        hits = bible.detect_characters("Mira walks the market")
        assert len(hits) == 1
        prompt = bible.apply_lock("market scene", WorldLock(character_ids=["mira"]))
        assert "silver hair" in prompt


class TestCounterfactual:
    def test_change_parse(self):
        edit = PreserveEditPlanner().parse("change the sky to purple", base_prompt="red dragon, blue sky, castle")
        assert edit is not None
        assert "purple" in edit.edited.lower()
        assert edit.preserve.locked_phrases


class TestUncertainty:
    def test_overloaded_scene(self):
        prompt = ", ".join([f"object{i}" for i in range(8)])
        rep = ConfidenceGate().analyze(prompt)
        assert rep.score >= 0.12
        assert rep.signals


class TestInverseLayout:
    def test_infers_regions(self):
        img = torch.zeros(3, 64, 64)
        img[0, :32, :32] = 1.0
        img[1, 32:, 32:] = 1.0
        sketch = LayoutSketchInferer(grid=4).infer_from_tensor(img, global_prompt="test")
        assert sketch.regions


class TestProvenance:
    def test_write_audit(self):
        audit = build_audit_bundle(prompt="cat", seed=42, ckpt_path="model.pt")
        with tempfile.TemporaryDirectory() as td:
            path = write_audit_bundle(audit, Path(td) / "audit.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["seed"] == 42


class TestFactRag:
    def test_overlap_rank(self):
        rag = StyleFactRetriever([{"text": "teal shadows cinematic", "tags": ["noir"]}])
        hits = rag.query("cinematic teal portrait")
        assert hits
        merged = rag.inject_prompt("portrait", hits)
        assert "teal" in merged


class TestStyleDNA:
    def test_blend_profiles(self):
        dna = StyleDNABlender().blend([("noir", 0.6), ("editorial", 0.4)])
        assert "noir" in dna.blended_positive or "film noir" in dna.blended_positive


class TestSceneRelations:
    def test_left_of(self):
        g = SceneRelationParser().parse("knight on the left of dragon")
        assert g.edges
        hints = g.to_regional_hints()
        assert hints


class TestStoryboard:
    def test_scene_list(self):
        board = StoryboardPlanner().from_scene_list(["establishing shot", "close-up face"], title="demo")
        board = StoryboardPlanner().with_carry(board, ["same outfit"])
        prompts = board.expanded_prompts()
        assert "same outfit" in prompts[1]


class TestTokenEmphasis:
    def test_text_boost(self):
        plan = TokenEmphasisPlanner().plan("logo text on neon sign")
        assert plan.weights
        assert plan.cfg_multiplier > 1.0


class TestDeepSynthesis:
    def test_deep_plan_kwargs(self):
        plan = analyze_deep("rain at night with crowd and logo sign", layout_regions=1)
        assert plan.plausibility or plan.uncertainty
        kw = deep_sample_kwargs(plan, base_negative="ugly")
        assert kw["prompt"]
        assert "frontier_guidance_tiers" in kw
