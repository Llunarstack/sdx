"""Tests for perfect frontier and scene-quality modules."""

from __future__ import annotations

import pytest
from frontier.atmosphere import AtmospherePlanner
from frontier.lighting import LightingPlanner
from frontier.materials import MaterialPlanner
from frontier.optics import LensCharacterPlanner
from frontier.perfect import analyze_perfect
from frontier.registry import idea_by_id
from frontier.safety import ContentPolicy, PolicyDecision, PolicyTier
from frontier.typography import TypographyPlanner


class TestSceneModules:
    def test_lighting_rembrandt(self):
        pos, _ = LightingPlanner().fragments("Rembrandt portrait studio")
        assert "Rembrandt" in pos or "triangle" in pos.lower()

    def test_atmosphere_fog(self):
        pos, _ = AtmospherePlanner().fragments("foggy city street")
        assert "fog" in pos.lower()

    def test_materials_metal_glass(self):
        pos, neg = MaterialPlanner().fragments("chrome glass window metal table")
        assert "metal" in pos.lower() or "Fresnel" in pos
        assert len(neg) > 10

    def test_typography_quotes(self):
        plan = TypographyPlanner().plan('poster with "HELLO WORLD" text')
        assert plan.quoted_text
        assert plan.cfg_boost > 1.0

    def test_lens_anamorphic(self):
        pos, _ = LensCharacterPlanner().fragments("anamorphic cinematic shot")
        assert "anamorphic" in pos.lower() or "bokeh" in pos.lower()


class TestContentPolicy:
    def test_refuse_sexual_minor(self):
        rep = ContentPolicy(PolicyTier.MODERATE).evaluate("nude young girl")
        assert rep.decision == PolicyDecision.REFUSE

    def test_allow_landscape(self):
        rep = ContentPolicy(PolicyTier.MODERATE).evaluate("mountain landscape sunset")
        assert rep.decision == PolicyDecision.ALLOW


class TestPerfectFrontier:
    def test_perfect_merge(self):
        plan = analyze_perfect(
            'photoreal Rembrandt portrait, foggy city, anamorphic, "OPEN" sign',
            safety_tier=PolicyTier.OFF,
        )
        assert not plan.refused
        assert plan.final_prompt
        assert plan.lighting_pos or plan.atmosphere_pos

    def test_registry_perfect(self):
        assert idea_by_id("perfect_frontier").status == "implemented"


class TestPerfectRefuse:
    def test_refused_plan(self):
        plan = analyze_perfect("explicit loli", safety_tier=PolicyTier.MODERATE)
        assert plan.refused

    def test_sample_features_raises(self):
        from utils.generation.sample_features import _apply_frontier_perfect

        class Args:
            prompt = "nude child"
            negative_prompt = ""
            frontier_perfect = True
            frontier_subject = False
            safety_tier = "moderate"
            frontier_serendipity = 0.25
            frontier_auto_resolve = False
            cfg_scale = 7.5
            _box_layout_spec = None

        with pytest.raises(ValueError, match="refused"):
            _apply_frontier_perfect(Args(), steps=28)
