"""Tests for frontier outside-the-box generation modules.

Also holds what used to be: test_frontier_creative, test_frontier_horizon, test_frontier_ideas, test_frontier_perfect, test_frontier_regressions, test_frontier_subject.
"""

from __future__ import annotations

import frontier.subject as realism
import pytest
import torch
from config.defaults.art_mediums import detect_medium_ids, guidance_fragments
from frontier import FrontierEngine, analyze_prompt
from frontier.attention import build_attention_layout_plan
from frontier.engine import (
    AbsenceExtractor,
    ContradictionScanner,
    EntropyBudgetAllocator,
    GenerationEchoMemory,
    MomentPhase,
    SerendipityInjector,
    TemporalMomentAnalyzer,
    WitnessPerspective,
    WitnessPerspectiveAnalyzer,
)
from frontier.imagination import (
    CreativeConstraintEngine,
    DreamLogicPlanner,
    GlitchPlanner,
    MoodPhysics,
    MutationAxis,
    ParadoxKeeper,
    PromptMutator,
    ShotGrammar,
    SynesthesiaEngine,
    analyze_imagination,
    imagination_sample_kwargs,
)
from frontier.layout import (
    OmostCanvas,
    RegionFusionSchedule,
    bind_coordinates_to_prompt,
    canvas_to_box_layout,
    fusion_weight_at_step,
    score_layout_masks,
)
from frontier.perfect import (
    AtmospherePlanner,
    ContentPolicy,
    MaterialPlanner,
    MotionPlanner,
    MotionStyle,
    PolicyDecision,
    PolicyTier,
    TypographyPlanner,
    analyze_perfect,
)
from frontier.registry import idea_by_id, list_ideas
from frontier.subject import (
    AntiSlopScanner,
    BodyMode,
    BodyPlanner,
    BrushPlanner,
    CreatureFamily,
    CreatureTaxonomy,
    MatureClass,
    MatureGuidance,
    PhotorealStack,
    RealismTier,
    analyze_subject,
    subject_sample_kwargs,
)
from frontier.synthesis import (
    ComputeBudgetPlanner,
    ConfidenceGate,
    GuidanceTier,
    PhysicalPlausibilityScanner,
    SceneRelationParser,
    TokenEmphasisPlanner,
    analyze_deep,
    deep_sample_kwargs,
)
from utils.generation.regional_box_prompting import build_latent_region_masks, parse_box_layout

# ==================================================================================================
# test_frontier
# ==================================================================================================


class TestContradictionScanner:
    def test_detects_time_conflict(self):
        hits = ContradictionScanner().scan("busy plaza at noon with a dramatic sunset")
        assert hits
        assert hits[0].category == "time"

    def test_suggest_rewrite_drops_side(self):
        out = ContradictionScanner().suggest_rewrite("empty street with a huge crowd", pick="left")
        assert "crowd" not in out.lower()


class TestAbsenceExtractor:
    def test_no_people(self):
        constraints = AbsenceExtractor().extract("wide shot, no people, clear sky")
        subjects = {c.subject for c in constraints}
        assert "people" in subjects

    def test_merge_negative(self):
        ext = AbsenceExtractor()
        c = ext.extract("no text on signs")
        neg = ext.merge_negative_prompt("blurry", c)
        assert "text" in neg.lower()


class TestWitness:
    def test_cctv_guard(self):
        frame = WitnessPerspectiveAnalyzer().analyze("security camera footage of a hallway")
        assert frame.perspective == WitnessPerspective.GUARD
        assert frame.confidence > 0


class TestMoment:
    def test_anticipation_curve(self):
        cue = TemporalMomentAnalyzer(num_steps=10).analyze("statue about to fall")
        assert cue.phase.value == "anticipation"
        assert len(cue.step_emphasis) == 10
        assert cue.step_emphasis[0] > cue.step_emphasis[-1]


class TestChaos:
    def test_serendipity_peak_mid_schedule(self):
        curve = SerendipityInjector(num_steps=20).curve(dial=0.5)
        assert curve.peak_step > 2
        assert curve.peak_step < 18

    def test_entropy_budget_sums(self):
        b = EntropyBudgetAllocator(num_steps=12).allocate(total=1.0)
        assert abs(sum(b.per_step) - 1.0) < 1e-5

    def test_serendipity_noise_scale(self):
        inj = SerendipityInjector(num_steps=8)
        curve = inj.curve(0.4)
        n = torch.ones(2, 4)
        out = inj.apply_to_noise(n, curve.peak_step, curve)
        assert out.mean() > 1.0


class TestGenerationEcho:
    def test_remembers_failure_tags(self):
        mem = GenerationEchoMemory(capacity=8)
        mem.record_failure("red dragon on cliff", ["extra claws", "melted wings"])
        suffix = mem.negative_suffix("red dragon on cliff at dawn")
        assert "extra claws" in suffix


class TestFrontierEngine:
    def test_full_plan(self):
        plan = analyze_prompt("empty plaza at noon, sunset sky, CCTV view, about to rain")
        assert plan.contradictions
        assert plan.witness is not None
        assert plan.augmented_prompt

    def test_sample_kwargs(self):
        eng = FrontierEngine(serendipity_dial=0.2)
        plan = eng.analyze("candid street photo, no cars")
        kw = eng.sample_kwargs(plan, base_negative="ugly")
        assert "prompt" in kw
        assert kw["frontier_serendipity_scales"]
        assert "cars" in kw["negative_prompt"].lower() or "ugly" in kw["negative_prompt"].lower()


# ==================================================================================================
# test_frontier_creative: Tests for creative frontier — imagination, mutations, mood physics.
# ==================================================================================================


class TestCreativeModules:
    def test_dream_boosts_serendipity(self):
        _, _, boost = DreamLogicPlanner().fragments("surreal melting dreamscape")
        assert boost > 0

    def test_paradox_suppresses_resolve(self):
        _, _, suppress = ParadoxKeeper().fragments("Escher impossible stairs")
        assert suppress

    def test_mutator_changes_prompt(self):
        muts = PromptMutator().mutate_batch("sunset landscape", seed=1, count=3)
        assert len(muts) >= 2
        assert any(m.mutated != m.original for m in muts)

    def test_random_constraint(self):
        pack = CreativeConstraintEngine().suggest_random(seed=3)
        assert pack.rules

    def test_synesthesia_knobs(self):
        ser, cfg = SynesthesiaEngine().diffusion_knobs("jazz club smoky portrait")
        assert ser > 0

    def test_shot_grammar_ots(self):
        pos, _ = ShotGrammar().fragments("over the shoulder dialogue scene")
        assert "shoulder" in pos.lower()

    def test_mood_physics_curve(self):
        plan = MoodPhysics().analyze("ominous dread forest")
        curve = MoodPhysics().step_emphasis_curve(plan, 10)
        assert len(curve) == 10

    def test_glitch_vhs(self):
        _, _, boost = GlitchPlanner().fragments("VHS tracking error aesthetic")
        assert boost > 0


class TestImaginationEngine:
    def test_full_plan(self):
        plan = analyze_imagination(
            "surreal Escher city, jazz mood, over the shoulder, VHS glitch, rusted gate, steampunk cyberpunk crowd",
            mutate_count=2,
            mutate_seed=42,
        )
        assert plan.creative_trace
        assert plan.serendipity_dial > 0.2
        kw = imagination_sample_kwargs(plan)
        assert kw["frontier_creative"]
        assert plan.mutations

    def test_registry(self):
        assert idea_by_id("imagination_engine").status == "implemented"


class TestNewCreativeModules:
    def test_fusion(self):
        from frontier.imagination import GenreCollisionEngine

        pos, _ = GenreCollisionEngine().fragments("steampunk city with cyberpunk neon")
        assert "steampunk" in pos.lower() or "brass" in pos.lower()

    def test_crowd(self):
        from frontier.imagination import CrowdGrammar

        pos, neg = CrowdGrammar().fragments("concert crowd at stadium")
        assert "duplicate" in neg.lower()

    def test_creative_refine_mutations(self):

        # dry-run: only test mutation list building via analyze_imagination
        from frontier.imagination import analyze_imagination

        plan = analyze_imagination("sunset dragon", mutate_count=3, mutate_seed=1)
        assert len(plan.mutations) >= 2


class TestPromptReplace:
    def test_replace_prompt_in_cmd(self):
        from utils.generation.sample_features import _replace_or_append_prompt

        cmd = ["python", "sample.py", "--prompt", "old", "--ckpt", "x.pt"]
        out = _replace_or_append_prompt(cmd, "new prompt")
        assert "--prompt" in out
        assert "new prompt" in out
        assert "old" not in out


# ==================================================================================================
# test_frontier_horizon: Tests for horizon / ahead-of-curve frontier modules.
# ==================================================================================================


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


class TestUncertainty:
    def test_overloaded_scene(self):
        prompt = ", ".join([f"object{i}" for i in range(8)])
        rep = ConfidenceGate().analyze(prompt)
        assert rep.score >= 0.12
        assert rep.signals


class TestSceneRelations:
    def test_left_of(self):
        g = SceneRelationParser().parse("knight on the left of dragon")
        assert g.edges
        hints = g.to_regional_hints()
        assert hints


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


# ==================================================================================================
# test_frontier_ideas: Tests for research-backed frontier modules.
# ==================================================================================================


class TestResearchRegistry:
    def test_has_implemented_ideas(self):
        impl = list_ideas(status="implemented")
        assert len(impl) >= 8
        assert idea_by_id("omost_canvas").status == "implemented"


class TestOmostCanvas:
    def test_exports_box_layout(self):
        c = OmostCanvas()
        c.set_global_description("sunset landscape")
        c.add_local_description("red dragon", anchor="left", name="dragon")
        c.add_local_description("castle", box=(0.55, 0.2, 0.95, 0.9), name="castle")
        d = canvas_to_box_layout(c)
        assert d["global_prompt"] == "sunset landscape"
        assert len(d["regions"]) == 2


class TestCoordinateBind:
    def test_token_prefix_is_idempotent(self):
        p = bind_coordinates_to_prompt("a cat", (0.1, 0.2, 0.5, 0.8))
        assert p == "<loc_0.1_0.2_0.5_0.8> a cat"
        assert bind_coordinates_to_prompt(p, (0.1, 0.2, 0.5, 0.8)) == p


class TestLamicSchedule:
    def test_early_more_isolated(self):
        s = RegionFusionSchedule()
        early = fusion_weight_at_step(1, 30, s)
        late = fusion_weight_at_step(28, 30, s)
        assert late > early


class TestLayoutMetrics:
    def test_scores_masks(self):
        spec = parse_box_layout(
            {
                "regions": [
                    {"name": "a", "box": [0, 0, 0.5, 1], "prompt": "x"},
                    {"name": "b", "box": [0.5, 0, 1, 1], "prompt": "y"},
                ]
            }
        )
        rm, bg = build_latent_region_masks(spec, 16, 16, device=torch.device("cpu"))
        rep = score_layout_masks(rm, bg)
        assert rep.inclusion_ratio == 1.0


class TestAttentionPlan:
    def test_enforce_early_steps(self):
        spec = parse_box_layout({"regions": [{"name": "a", "box": [0, 0, 1, 1], "prompt": "x"}]})
        plan = build_attention_layout_plan(spec.regions, num_steps=20, inject_frac=0.5)
        assert len(plan.enforce_steps) == 10


class TestRegionalInjectConfig:
    def test_parse_inject_keys(self):
        spec = parse_box_layout(
            {
                "mask_inject_steps": 8,
                "base_ratio": 0.2,
                "coordinate_tokens": True,
                "regions": [{"name": "a", "box": [0, 0, 1, 1], "prompt": "sky"}],
            }
        )
        assert spec.inject.mask_inject_steps == 8
        assert spec.inject.base_ratio == 0.2
        assert spec.inject.use_coordinate_tokens is True


# ==================================================================================================
# test_frontier_perfect: Tests for perfect frontier and scene-quality modules.
# ==================================================================================================


class TestSceneModules:
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
        assert plan.atmosphere_pos
        assert plan.art_direction.light_pattern == "rembrandt"
        assert plan.art_direction.perspective == "anamorphic"

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


# ==================================================================================================
# test_frontier_regressions: Regression tests for frontier planner bugs fixed during the polish pass.
# ==================================================================================================


def test_just_before_is_anticipation_not_aftermath():
    cue = TemporalMomentAnalyzer(num_steps=8).analyze("diver just before the jump")
    assert cue.phase == MomentPhase.ANTICIPATION
    assert TemporalMomentAnalyzer().analyze("street just after the storm").phase == MomentPhase.AFTERMATH


def test_motion_cues_match_at_prompt_start():
    assert MotionPlanner().plan("racing car on a wet track").style == MotionStyle.PAN
    assert MotionPlanner().plan("dive into a coral reef").style == MotionStyle.FALL


def test_palette_invert_does_not_duplicate_an_inverted_prompt():
    mutator = PromptMutator(axes=[MutationAxis.PALETTE_INVERT])
    (mutation,) = mutator.mutate_batch("inverted neon city", count=1)
    assert mutation.mutated == "inverted neon city"


def test_detail_zoom_on_empty_prompt_has_no_leading_comma():
    (mutation,) = PromptMutator(axes=[MutationAxis.DETAIL_ZOOM]).mutate_batch("", count=1)
    assert mutation.mutated == "extreme macro detail on focal texture"


def test_realism_exports_both_planners():
    assert {"AntiSlopScanner", "PhotorealStack", "RealismTier"} <= set(realism.__all__)


# ==================================================================================================
# test_frontier_subject: Tests for subject-aware frontier: anatomy, creatures, mature, medium, realism.
# ==================================================================================================


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
        ids = detect_medium_ids("hokusai ukiyo-e wave woodblock", include_photography=False)
        assert "ukiyo_e" in ids

    def test_extended_fragments_merge(self):
        pos, neg = guidance_fragments(
            "spray paint graffiti mural with impasto",
            "auto",
            include_photography=False,
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
