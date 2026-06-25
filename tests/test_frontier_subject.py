"""Tests for subject-aware frontier: anatomy, creatures, mature, medium, realism."""

from __future__ import annotations

from frontier.anatomy import BodyMode, BodyPlanner
from frontier.creatures import CreatureFamily, CreatureTaxonomy
from frontier.mature import MatureClass, MatureGuidance
from frontier.medium import BrushPlanner, detect_extended_medium_ids, extended_guidance_fragments
from frontier.realism import AntiSlopScanner, PhotorealStack, RealismTier
from frontier.registry import idea_by_id
from frontier.subject import analyze_subject, subject_sample_kwargs


class TestBodyPlanner:
    def test_creature_not_human_negatives(self):
        plan = BodyPlanner().plan("ancient red dragon perched on cliff")
        assert plan.mode == BodyMode.CREATURE
        assert "extra fingers" not in plan.negative

    def test_human_hands_risk(self):
        plan = BodyPlanner().plan("photoreal woman holding a cup, detailed hands")
        assert plan.risk.hand_focus
        assert plan.cfg_bias >= 1.1


class TestCreatureTaxonomy:
    def test_dragon_family(self):
        cp = CreatureTaxonomy().plan("fire dragon with wings")
        assert cp is not None
        assert cp.family == CreatureFamily.DRACONIC
        assert "scale" in cp.surface_hint.lower()

    def test_insect_legs(self):
        cp = CreatureTaxonomy().plan("giant mantis insect creature")
        assert cp is not None
        assert "six legs" in cp.limb_hint


class TestMatureGuidance:
    def test_boudoir_class(self):
        plan = MatureGuidance().plan("intimate boudoir portrait in lingerie")
        assert plan.content_class == MatureClass.BOUDOIR
        assert "skin" in plan.positive.lower()

    def test_none_for_landscape(self):
        plan = MatureGuidance().plan("mountain landscape at dawn")
        assert plan.content_class == MatureClass.NONE


class TestExtendedMediums:
    def test_detect_ukiyo(self):
        ids = detect_extended_medium_ids("hokusai ukiyo-e wave woodblock")
        assert "ukiyo_e" in ids

    def test_extended_fragments_merge(self):
        pos, neg = extended_guidance_fragments(
            "spray paint graffiti mural with impasto",
            include_photography=False,
            base_mode="auto",
        )
        assert "overspray" in pos.lower() or "impasto" in pos.lower()
        assert len(neg) > 10


class TestBrushPlanner:
    def test_impasto(self):
        plan = BrushPlanner().plan("thick impasto oil with palette knife")
        assert plan is not None
        assert plan.primary_style.value == "impasto"


class TestRealism:
    def test_anti_slop_tier(self):
        plan = AntiSlopScanner().plan("hyperreal 8k photo portrait dslr")
        assert plan.tier == RealismTier.HYPERREAL
        assert "plastic skin" in plan.negative

    def test_photoreal_lens(self):
        pos, _ = PhotorealStack().fragments("portrait photo 85mm bokeh")
        assert "bokeh" in pos.lower() or "depth" in pos.lower()


class TestSubjectOrchestration:
    def test_full_merge(self):
        plan = analyze_subject("photoreal dragon and woman hands, watercolor sky, nsfw boudoir")
        assert plan.merged_positive
        assert plan.merged_negative
        kw = subject_sample_kwargs(plan)
        assert kw["subject_body_mode"] in ("creature", "realistic_human", "stylized_human", "chimera")

    def test_registry_entries(self):
        assert idea_by_id("body_mode_router").status == "implemented"
        assert idea_by_id("anti_slop_realism").status == "implemented"
