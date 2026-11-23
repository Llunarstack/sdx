"""Tests for frontier outside-the-box generation modules.

Also holds what used to be: test_frontier_creative, test_frontier_horizon, test_frontier_ideas, test_frontier_perfect, test_frontier_regressions, test_frontier_subject, test_bridge_2d3d, test_creative_3d, test_field_training, test_mesh_export, test_neus_render, test_radiance_and_rigging, test_triplane_spatial, test_view_dependent_render, test_perfect_model_helpers.
"""

from __future__ import annotations

import frontier.planning as realism
import numpy as np
import pytest
import torch
from config.styles import detect_medium_ids, guidance_fragments
from frontier import FrontierEngine, analyze_prompt
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
    build_attention_layout_plan,
    canvas_to_box_layout,
    fusion_weight_at_step,
    score_layout_masks,
)
from frontier.multiview import (
    AABB,
    AssetGraph,
    BridgeConfig,
    Lift2Dto3D,
    NeusRenderer,
    SemanticPart,
    SHRadianceDecoder,
    Skeleton,
    SpatialLatentField,
    TriPlaneConfig,
    View,
    auto_skeleton_from_points,
    blend_conditioning,
    build_spatial_field,
    contact_sheet,
    detokenize_skeleton,
    eikonal_loss,
    eval_sh_basis,
    extract_mesh,
    fit_from_views,
    fit_sdf,
    forward_kinematics,
    linear_blend_skinning,
    look_at_rays,
    mutate_conditioning,
    perfect_angles,
    rig_mesh,
    sdf_and_gradient,
    sdf_regression_loss,
    sh_num_coeffs,
    skinning_matrices,
    skinning_weights,
    slerp,
    spatial_graft,
    sphere_sdf,
    surface_normal,
    tokenize_skeleton,
    validate_mesh,
    viewpoint_rays,
    volume_render_sdf,
    write_obj,
)
from frontier.planning import (
    AntiSlopScanner,
    AtmospherePlanner,
    BodyMode,
    BodyPlanner,
    BrushPlanner,
    ComputeBudgetPlanner,
    ConfidenceGate,
    ContentPolicy,
    CreatureFamily,
    CreatureTaxonomy,
    GuidanceTier,
    MaterialPlanner,
    MatureClass,
    MatureGuidance,
    MotionPlanner,
    MotionStyle,
    PhotorealStack,
    PhysicalPlausibilityScanner,
    PolicyDecision,
    PolicyTier,
    RealismTier,
    SceneRelationParser,
    TokenEmphasisPlanner,
    TypographyPlanner,
    analyze_deep,
    analyze_perfect,
    analyze_subject,
    deep_sample_kwargs,
    subject_sample_kwargs,
)
from frontier.registry import idea_by_id, list_ideas
from utils.generation.regional import build_latent_region_masks, parse_box_layout

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
        from utils.generation.sample_prompt import _replace_or_append_prompt

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
        from utils.generation.sample_prompt import _apply_frontier_perfect

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


# ==================================================================================================
# test_bridge_2d3d: Tests for the 2D->3D conditioning bridge and end-to-end field wiring.
# ==================================================================================================


BCFG = BridgeConfig(text_dim=64, image_dim=48, cond_dim=32, num_heads=4, hidden=64)


@pytest.fixture
def bridge() -> Lift2Dto3D:
    torch.manual_seed(0)
    return Lift2Dto3D(BCFG)


def test_text_only_shape(bridge: Lift2Dto3D) -> None:
    cond = bridge(torch.randn(3, BCFG.text_dim))
    assert cond.shape == (3, BCFG.cond_dim)


def test_text_plus_image_shape(bridge: Lift2Dto3D) -> None:
    cond = bridge(torch.randn(2, BCFG.text_dim), torch.randn(2, 10, BCFG.image_dim))
    assert cond.shape == (2, BCFG.cond_dim)


def test_image_tokens_change_output(bridge: Lift2Dto3D) -> None:
    bridge.eval()
    text = torch.randn(1, BCFG.text_dim)
    a = bridge(text)
    b = bridge(text, torch.randn(1, 10, BCFG.image_dim))
    assert not torch.allclose(a, b)


def test_padding_mask_ignores_padded_tokens(bridge: Lift2Dto3D) -> None:
    bridge.eval()
    text = torch.randn(1, BCFG.text_dim)
    real = torch.randn(1, 4, BCFG.image_dim)
    padded = torch.cat([real, torch.randn(1, 3, BCFG.image_dim)], dim=1)
    mask = torch.tensor([[False, False, False, False, True, True, True]])
    out_real = bridge(text, real)
    out_masked = bridge(text, padded, image_token_mask=mask)
    assert torch.allclose(out_real, out_masked, atol=1e-5)


def test_image_disabled_rejects_tokens() -> None:
    b = Lift2Dto3D(BridgeConfig(text_dim=64, image_dim=0, cond_dim=32))
    with pytest.raises(ValueError):
        b(torch.randn(1, 64), torch.randn(1, 5, 48))


def test_end_to_end_text_to_field() -> None:
    """The bridge output must drive a real 3D field forward pass."""
    torch.manual_seed(1)
    bridge = Lift2Dto3D(BridgeConfig(text_dim=64, image_dim=0, cond_dim=32))
    field = SpatialLatentField(TriPlaneConfig(cond_dim=32, plane_channels=8, grid_res=16, feature_dim=16))

    cond = bridge(torch.randn(2, 64))
    planes = field.encode(cond)
    pts = torch.empty(2, 20, 3).uniform_(-1, 1)
    sdf = field.decode_sdf(planes, pts)
    assert sdf.shape == (2, 20, 1)

    # gradient flows from the 3D field all the way back into the 2D bridge
    sdf.mean().backward()
    assert any(p.grad is not None for p in bridge.parameters())


def test_field_to_mesh_export() -> None:
    """Full path: conditioning -> field -> watertight mesh with vertex colors."""
    torch.manual_seed(2)
    field = SpatialLatentField(TriPlaneConfig(cond_dim=16, plane_channels=8, grid_res=16, feature_dim=16))
    planes = field.encode(torch.randn(1, 16))
    verts, faces, colors = field.to_mesh(planes, resolution=20, with_color=True)
    # An untrained field may or may not cross zero; assert the contract holds either way.
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    if len(verts):
        assert colors.shape == verts.shape
        assert faces.max() < len(verts)


# ==================================================================================================
# test_creative_3d: Tests for the creative SDX-3D features drawn from the original design notes:
# ==================================================================================================


def _sphere_sdf(radius: float, center=(0.0, 0.0, 0.0)):
    c = np.asarray(center)

    def fn(p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(p - c, axis=1) - radius

    return fn


# --------------------------------------------------------------------------- #
# Perfect Angles
# --------------------------------------------------------------------------- #


def test_viewpoint_camera_positions():
    o, d = viewpoint_rays("front", radius=2.5, height=8, width=8)
    # Camera origin sits on the -Z axis at the requested radius, all rays share it.
    assert torch.allclose(o[0], torch.tensor([0.0, 0.0, -2.5]), atol=1e-5)
    assert torch.allclose(d.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_top_view_up_vector_is_nondegenerate():
    # up is parallel to view dir for top/bottom; rays must still be finite & unit.
    _, d = viewpoint_rays("top", height=6, width=6)
    assert torch.isfinite(d).all()
    assert torch.allclose(d.norm(dim=-1), torch.ones(36), atol=1e-5)


def test_perfect_angles_returns_all_views_same_object():
    torch.manual_seed(0)
    field = SpatialLatentField(TriPlaneConfig(cond_dim=16, plane_channels=8, grid_res=16, feature_dim=16))
    renderer = NeusRenderer(n_samples=24)
    planes = field.encode(torch.zeros(1, 16))
    imgs = perfect_angles(field, planes, renderer, names=["front", "right", "top"], resolution=12)
    assert set(imgs) == {"front", "right", "top"}
    for im in imgs.values():
        assert im.shape == (12, 12, 3)
        assert (im >= 0).all() and (im <= 1).all()


def test_contact_sheet_tiles_images():
    imgs = [np.full((10, 10, 3), v, dtype=np.float32) for v in (0.2, 0.4, 0.6, 0.8)]
    sheet = contact_sheet(imgs, cols=2, pad=2)
    # 2x2 grid of 10px tiles + 3 pads of 2px each way.
    assert sheet.shape == (2 * 10 + 3 * 2, 2 * 10 + 3 * 2, 3)


# --------------------------------------------------------------------------- #
# Semantic Asset Graph — non-destructive editing
# --------------------------------------------------------------------------- #


def test_editing_one_part_leaves_others_bit_identical():
    graph = (
        AssetGraph()
        .add(SemanticPart("a", _sphere_sdf(0.25, (-0.5, 0, 0)), AABB.around((-0.5, 0, 0), 0.35)))
        .add(SemanticPart("b", _sphere_sdf(0.25, (0.5, 0, 0)), AABB.around((0.5, 0, 0), 0.35)))
    )
    v0, _ = graph.to_mesh(resolution=40)
    b_region = v0[v0[:, 0] > 0.1]
    b_before = np.array(sorted(map(tuple, np.round(b_region, 6))))

    # Edit only part "a" (grow it). Part "b" must not move at all.
    graph.edit_part("a", _sphere_sdf(0.32, (-0.5, 0, 0)))
    v1, _ = graph.to_mesh(resolution=40)
    b_after = np.array(sorted(map(tuple, np.round(v1[v1[:, 0] > 0.1], 6))))

    assert b_before.shape == b_after.shape
    assert np.array_equal(b_before, b_after)


def test_graph_union_contains_both_parts():
    graph = (
        AssetGraph()
        .add(SemanticPart("a", _sphere_sdf(0.25, (-0.5, 0, 0)), AABB.around((-0.5, 0, 0), 0.35)))
        .add(SemanticPart("b", _sphere_sdf(0.25, (0.5, 0, 0)), AABB.around((0.5, 0, 0), 0.35)))
    )
    v, _ = graph.to_mesh(resolution=40)
    assert (v[:, 0] < -0.1).any() and (v[:, 0] > 0.1).any()  # both lobes present


# --------------------------------------------------------------------------- #
# Reality-check validator
# --------------------------------------------------------------------------- #


def test_validator_on_sphere_matches_analytic_quantities():
    radius = 0.6
    v, f = extract_mesh(_sphere_sdf(radius), resolution=48)
    report = validate_mesh(v, f)
    assert report.watertight
    assert report.euler_characteristic == 2
    # Volume 4/3 pi r^3, area 4 pi r^2 — within a few % for this resolution.
    assert abs(report.volume - (4 / 3) * np.pi * radius**3) < 0.03
    assert abs(report.surface_area - 4 * np.pi * radius**2) < 0.1
    assert max(abs(c) for c in report.center_of_mass) < 0.02  # centered
    assert all(abs(d - 2 * radius) < 0.05 for d in report.dimensions)


def test_validator_flags_open_mesh():
    # A single triangle: open, degenerate volume -> warnings, not watertight.
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    f = np.array([[0, 1, 2]], dtype=np.int64)
    report = validate_mesh(v, f)
    assert not report.watertight
    assert report.warnings


# --------------------------------------------------------------------------- #
# Genetic cross-breeding
# --------------------------------------------------------------------------- #


def test_slerp_hits_endpoints():
    a = torch.randn(2, 16)
    b = torch.randn(2, 16)
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-4)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-4)


def test_blend_conditioning_is_between_parents():
    a = torch.randn(1, 32)
    b = torch.randn(1, 32)
    mid = blend_conditioning(a, b, 0.5)
    assert mid.shape == a.shape
    assert not torch.allclose(mid, a) and not torch.allclose(mid, b)


def test_spatial_graft_takes_lower_from_a_upper_from_b():
    a = _sphere_sdf(0.5)  # bigger body (A) for the lower half
    b = _sphere_sdf(0.3)  # smaller body (B) for the upper half
    chimera = spatial_graft(a, b, axis=1, split=0.0, smooth=0.05)

    below = np.array([[0.0, -0.4, 0.0]])
    above = np.array([[0.0, 0.4, 0.0]])
    assert np.isclose(chimera(below), a(below), atol=1e-3)  # A dominates below seam
    assert np.isclose(chimera(above), b(above), atol=1e-3)  # B dominates above seam

    # And it still meshes to something watertight.
    v, f = extract_mesh(chimera, resolution=40)
    assert len(v) > 0
    assert validate_mesh(v, f).watertight


def test_mutation_perturbs_but_stays_close():
    cond = torch.zeros(1, 16)
    m = mutate_conditioning(cond, strength=0.1, seed=1)
    assert m.shape == cond.shape
    assert 0.0 < m.abs().mean() < 1.0


# ==================================================================================================
# test_field_training: Tests for the spatial-field training objective.
# ==================================================================================================


def _small_field(seed: int = 0) -> SpatialLatentField:
    torch.manual_seed(seed)
    return SpatialLatentField(
        TriPlaneConfig(cond_dim=32, plane_channels=16, grid_res=32, feature_dim=32, decoder_hidden=64)
    )


def test_eikonal_zero_for_unit_gradients() -> None:
    grad = torch.nn.functional.normalize(torch.randn(2, 50, 3), dim=-1)
    assert eikonal_loss(grad).item() < 1e-6


def test_eikonal_positive_for_nonunit_gradients() -> None:
    grad = 2.0 * torch.nn.functional.normalize(torch.randn(2, 50, 3), dim=-1)
    assert abs(eikonal_loss(grad).item() - 1.0) < 1e-5  # (2 - 1)^2 = 1


def test_regression_weights_surface_more_than_interior() -> None:
    # Same absolute error (0.2), once near the surface (target≈0) and once far
    # (target=1). The near-surface error must be penalized more.
    near = sdf_regression_loss(torch.tensor([[[0.2]]]), torch.tensor([[[0.0]]]))
    far = sdf_regression_loss(torch.tensor([[[1.2]]]), torch.tensor([[[1.0]]]))
    assert near > far


def test_sdf_and_gradient_supports_double_backward() -> None:
    """Eikonal needs a differentiable gradient; verify second-order flow works
    through grid_sample and reaches the network parameters."""
    field = _small_field()
    planes = field.encode(torch.zeros(1, 32))
    points = torch.empty(1, 64, 3).uniform_(-1, 1)
    sdf, grad = sdf_and_gradient(field, planes, points)
    assert sdf.shape == (1, 64, 1)
    assert grad.shape == (1, 64, 3)
    eikonal_loss(grad).backward()
    assert any(p.grad is not None for p in field.parameters())


def test_surface_normal_is_unit() -> None:
    field = _small_field()
    planes = field.encode(torch.zeros(1, 32))
    n = surface_normal(field, planes, torch.empty(1, 40, 3).uniform_(-1, 1))
    norms = n.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_fit_sphere_learns_and_meshes_to_a_sphere() -> None:
    field = _small_field(seed=3)
    radius = 0.5
    result = fit_sdf(field, sphere_sdf(radius), steps=250, n_points=1024, lr=2e-3)

    # 1) The regression loss dropped substantially.
    first = float(np.mean(result.losses[:20]))
    last = float(np.mean(result.losses[-20:]))
    assert last < 0.3 * first, f"loss did not fall enough: {first:.4f} -> {last:.4f}"

    # 2) The trained field now meshes to a real, non-empty surface...
    planes = field.encode(result.cond)
    verts, faces = field.to_mesh(planes, resolution=32)
    assert len(verts) > 0 and len(faces) > 0

    # 3) ...and that surface is a sphere of about the right radius.
    r = np.linalg.norm(verts, axis=1)
    assert 0.35 < float(np.median(r)) < 0.65


# ==================================================================================================
# test_mesh_export: Tests for SDF -> mesh extraction (marching tetrahedra).
# ==================================================================================================


def _sphere_sdf_mesh_export(radius: float):
    def fn(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points, axis=1) - radius

    return fn


def test_sphere_mesh_is_nonempty():
    verts, faces = extract_mesh(_sphere_sdf_mesh_export(0.6), resolution=24)
    assert len(verts) > 0
    assert len(faces) > 0
    assert faces.max() < len(verts)  # faces index valid vertices


def test_sphere_vertices_lie_on_surface():
    radius = 0.6
    verts, _ = extract_mesh(_sphere_sdf_mesh_export(radius), resolution=32)
    r = np.linalg.norm(verts, axis=1)
    cell = 2.0 / 32
    # Linear interpolation puts vertices within ~one cell of the true surface.
    assert np.abs(r - radius).max() < cell


def test_sphere_is_closed_manifold():
    """Euler characteristic V - E + F == 2 for a watertight genus-0 sphere."""
    verts, faces = extract_mesh(_sphere_sdf_mesh_export(0.6), resolution=24)
    v = len(verts)
    f = len(faces)
    edges = set()
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            edges.add((min(x, y), max(x, y)))
    e = len(edges)
    assert v - e + f == 2, f"not closed: V={v} E={e} F={f}, chi={v - e + f}"


def test_every_edge_shared_by_two_faces():
    """A watertight mesh has no boundary edges — each is used exactly twice."""
    _, faces = extract_mesh(_sphere_sdf_mesh_export(0.6), resolution=24)
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            key = (min(x, y), max(x, y))
            counts[key] = counts.get(key, 0) + 1
    assert all(n == 2 for n in counts.values())


def test_empty_field_yields_empty_mesh():
    # SDF positive everywhere -> no surface.
    verts, faces = extract_mesh(lambda p: np.ones(len(p)), resolution=8)
    assert len(verts) == 0
    assert len(faces) == 0


def test_write_obj_roundtrip(tmp_path):
    verts, faces = extract_mesh(_sphere_sdf_mesh_export(0.6), resolution=16)
    out = tmp_path / "sphere.obj"
    colors = np.abs(verts)  # any [0,1]-ish colors
    write_obj(str(out), verts, faces, vertex_colors=colors)
    text = out.read_text(encoding="utf-8")
    assert text.count("\nv ") + text.startswith("v ") >= len(verts) - 1
    assert "f " in text
    # face indices are 1-based and within range
    for line in text.splitlines():
        if line.startswith("f "):
            idx = [int(t) for t in line.split()[1:]]
            assert all(1 <= i <= len(verts) for i in idx)


# ==================================================================================================
# test_neus_render: Tests for NeuS-style volume rendering and photometric multi-view training.
# ==================================================================================================


INV_S = torch.tensor(30.0)


def _sphere_scene(radius: float):
    def sdf_fn(p: torch.Tensor) -> torch.Tensor:
        return p.norm(dim=-1) - radius

    def color_fn(p: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * torch.nn.functional.normalize(p, dim=-1)  # normal-as-color

    return sdf_fn, color_fn


def test_look_at_rays_shapes_and_unit():
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3), height=8, width=8)
    assert o.shape == (64, 3) and d.shape == (64, 3)
    assert torch.allclose(d.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_analytic_sphere_silhouette_is_a_centered_disk():
    radius = 0.6
    sdf_fn, color_fn = _sphere_scene(radius)
    h = w = 24
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3), fov_deg=40, height=h, width=w)
    out = volume_render_sdf(sdf_fn, color_fn, o, d, INV_S, near=1.5, far=4.0, n_samples=128)
    opacity = out["opacity"].reshape(h, w)

    center = opacity[h // 2, w // 2]
    corner = opacity[0, 0]
    assert center > 0.9, f"sphere center should be opaque, got {center:.3f}"
    assert corner < 0.1, f"corner should see empty space, got {corner:.3f}"


def test_analytic_sphere_depth_matches_geometry():
    radius = 0.6
    dist = 3.0
    sdf_fn, color_fn = _sphere_scene(radius)
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -dist]), torch.zeros(3), fov_deg=40, height=24, width=24)
    out = volume_render_sdf(sdf_fn, color_fn, o, d, INV_S, near=1.5, far=4.0, n_samples=256)
    depth = out["depth"].reshape(24, 24)
    center_depth = float(depth[12, 12])
    # Front of the sphere sits at (dist - radius) along the central ray.
    assert abs(center_depth - (dist - radius)) < 0.1


def test_weights_are_a_valid_partition():
    sdf_fn, color_fn = _sphere_scene(0.6)
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3), height=8, width=8)
    out = volume_render_sdf(sdf_fn, color_fn, o, d, INV_S, n_samples=64)
    w = out["weights"]
    assert torch.all(w >= 0)
    assert torch.all(out["opacity"] <= 1.0 + 1e-4)


def test_field_render_is_differentiable():
    torch.manual_seed(0)
    field = SpatialLatentField(TriPlaneConfig(cond_dim=16, plane_channels=8, grid_res=16, feature_dim=16))
    renderer = NeusRenderer(n_samples=32)
    planes = field.encode(torch.zeros(1, 16))
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -2.0]), torch.zeros(3), height=6, width=6)
    out = renderer.render(field, planes, o, d, near=0.5, far=3.5)
    assert out["rgb"].shape == (36, 3)
    out["rgb"].mean().backward()
    assert any(p.grad is not None for p in field.parameters())
    assert renderer.deviation.variance.grad is not None


def test_photometric_overfit_to_sphere_views_decreases_loss():
    """End-to-end: supervise the field with posed images of a sphere."""
    torch.manual_seed(1)
    radius = 0.6
    sdf_fn, color_fn = _sphere_scene(radius)
    h = w = 14

    # Ground-truth renders from three viewpoints.
    cams = [
        torch.tensor([0.0, 0.0, -2.5]),
        torch.tensor([2.5, 0.0, 0.0]),
        torch.tensor([0.0, 2.2, -1.0]),
    ]
    views = []
    for c in cams:
        o, d = look_at_rays(c, torch.zeros(3), fov_deg=45, height=h, width=w)
        gt = volume_render_sdf(sdf_fn, color_fn, o, d, INV_S, near=0.5, far=4.0, n_samples=96)
        views.append(View(o, d, gt["rgb"], gt["opacity"].clamp(0, 1)))

    field = SpatialLatentField(
        TriPlaneConfig(cond_dim=16, plane_channels=16, grid_res=32, feature_dim=32, decoder_hidden=64)
    )
    renderer = NeusRenderer(n_samples=48)
    hist = fit_from_views(field, renderer, views, steps=120, rays_per_step=196, lr=3e-3, near=0.5, far=4.0)

    first = float(np.mean(hist[:15]))
    last = float(np.mean(hist[-15:]))
    assert last < 0.6 * first, f"photometric loss did not fall enough: {first:.4f} -> {last:.4f}"


# ==================================================================================================
# test_radiance_and_rigging: Tests for view-dependent SH radiance and the auto-rigging machinery.
# ==================================================================================================


# --------------------------------------------------------------------------- #
# Spherical-harmonic view-dependent radiance
# --------------------------------------------------------------------------- #


def test_sh_coeff_counts():
    assert [sh_num_coeffs(d) for d in range(4)] == [1, 4, 9, 16]


def test_sh_degree0_is_constant_over_directions():
    d1 = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    d2 = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    b1 = eval_sh_basis(d1, 0)
    b2 = eval_sh_basis(d2, 0)
    assert b1.shape == (10, 1)
    assert torch.allclose(b1, b2)  # DC term ignores direction
    assert torch.allclose(b1, torch.full_like(b1, 0.28209479177387814))


def test_sh_basis_shape_and_direction_dependence():
    dirs = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    basis = eval_sh_basis(dirs, 3)
    assert basis.shape == (5, 16)
    # Different directions produce different higher-order responses.
    a = eval_sh_basis(torch.tensor([[1.0, 0.0, 0.0]]), 3)
    b = eval_sh_basis(torch.tensor([[0.0, 1.0, 0.0]]), 3)
    assert not torch.allclose(a, b)


def test_sh_radiance_degree0_is_view_independent():
    torch.manual_seed(0)
    dec = SHRadianceDecoder(feature_dim=16, degree=0)
    feat = torch.randn(4, 16)
    d1 = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    d2 = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    c1 = dec(feat, d1)
    c2 = dec(feat, d2)
    assert c1.shape == (4, 3)
    assert torch.allclose(c1, c2)  # no directional terms -> flat color


def test_sh_radiance_degree3_varies_with_view_and_is_bounded():
    torch.manual_seed(1)
    dec = SHRadianceDecoder(feature_dim=16, degree=3)
    feat = torch.randn(4, 16)
    d1 = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    d2 = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    c1, c2 = dec(feat, d1), dec(feat, d2)
    assert not torch.allclose(c1, c2)  # reflections move with the camera
    assert torch.all((c1 >= 0) & (c1 <= 1))


def test_sh_radiance_gradients_flow():
    dec = SHRadianceDecoder(feature_dim=8, degree=2)
    feat = torch.randn(3, 8, requires_grad=True)
    dirs = torch.nn.functional.normalize(torch.randn(3, 3), dim=-1)
    dec(feat, dirs).mean().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


# --------------------------------------------------------------------------- #
# Auto-rigging: skeleton, tokenization, skinning, LBS, FK
# --------------------------------------------------------------------------- #


def _bar_points(n=400):
    """A long thin bar along +x — an obvious principal axis for rigging."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, n)
    y = rng.uniform(-0.05, 0.05, n)
    z = rng.uniform(-0.05, 0.05, n)
    return np.stack([x, y, z], axis=1)


def test_auto_skeleton_lies_on_principal_axis_and_chains():
    skel = auto_skeleton_from_points(_bar_points(), n_joints=5)
    assert skel.n_joints == 5
    # Joints should be spread along x and near-zero in y, z.
    assert np.ptp(skel.joints[:, 0]) > 1.0
    assert np.abs(skel.joints[:, 1:]).max() < 0.2
    # Parent chain: -1, 0, 1, 2, 3
    assert list(skel.parents) == [-1, 0, 1, 2, 3]


def test_skeleton_tokenization_roundtrip():
    skel = auto_skeleton_from_points(_bar_points(), n_joints=6)
    tokens = tokenize_skeleton(skel)
    assert len(tokens) == 6
    back = detokenize_skeleton(tokens)
    assert np.allclose(back.joints, skel.joints)
    assert np.array_equal(back.parents, skel.parents)


def test_skinning_weights_are_a_partition():
    pts = _bar_points()
    skel = auto_skeleton_from_points(pts, n_joints=5)
    w = skinning_weights(pts, skel)
    assert w.shape == (len(pts), 5)
    assert np.all(w >= 0)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_skinning_binds_vertex_to_nearest_joint():
    skel = auto_skeleton_from_points(_bar_points(), n_joints=5)
    # A vertex sitting right at the last joint should be dominated by it.
    v = skel.joints[-1][None, :]
    w = skinning_weights(v, skel, temperature=0.05)
    assert int(np.argmax(w[0])) == skel.n_joints - 1


def test_forward_kinematics_rest_pose_returns_joints():
    skel = auto_skeleton_from_points(_bar_points(), n_joints=5)
    rot = np.broadcast_to(np.eye(3), (skel.n_joints, 3, 3))
    g = forward_kinematics(skel, rot)
    assert np.allclose(g[:, :3, 3], skel.joints, atol=1e-9)


def test_lbs_identity_is_a_noop():
    pts = _bar_points()
    skel, w = rig_mesh(pts, n_joints=5)
    transforms = np.broadcast_to(np.eye(4), (skel.n_joints, 4, 4))
    out = linear_blend_skinning(pts, w, transforms)
    assert np.allclose(out, pts, atol=1e-9)


def test_lbs_bends_the_bar_when_a_joint_rotates():
    pts = _bar_points()
    skel, w = rig_mesh(pts, n_joints=5)
    # Rotate the last joint 90° about z. (The skeleton's axis sign is arbitrary,
    # so identify vertices by their binding weight, not by a hard-coded x-side.)
    rots = np.broadcast_to(np.eye(3), (skel.n_joints, 3, 3)).copy()
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    rots[-1] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mats = skinning_matrices(skel, rots)
    posed = linear_blend_skinning(pts, w, mats)

    shift = np.linalg.norm(posed - pts, axis=1)
    bound = w[:, -1] > 0.5  # vertices controlled by the rotated joint
    free = w[:, -1] < 0.01  # vertices essentially unbound to it
    assert bound.any() and free.any()
    assert shift[bound].mean() > 0.05  # they actually swung
    assert shift[free].mean() < 0.2 * shift[bound].mean()  # the rest stayed ~put


def test_skeleton_validates_length_mismatch():
    import pytest

    with pytest.raises(ValueError):
        Skeleton(joints=np.zeros((3, 3)), parents=np.array([-1, 0]))


# ==================================================================================================
# test_triplane_spatial: Tests for the tri-plane spatial latent field (SDX-Spatial 3D core).
# ==================================================================================================


# Small config so the whole suite runs fast on CPU.
CFG = TriPlaneConfig(
    cond_dim=32,
    plane_channels=8,
    grid_res=16,
    base_channels=32,
    feature_dim=16,
    decoder_hidden=32,
)


@pytest.fixture
def field() -> SpatialLatentField:
    torch.manual_seed(0)
    return SpatialLatentField(CFG)


def _planes_and_points(field: SpatialLatentField, b: int = 2, n: int = 64):
    cond = torch.randn(b, CFG.cond_dim)
    planes = field.encode(cond)
    points = torch.empty(b, n, 3).uniform_(-1.0, 1.0)
    return planes, points


def test_encode_plane_shapes(field: SpatialLatentField) -> None:
    planes, _ = _planes_and_points(field)
    assert len(planes) == 3
    for p in planes:
        assert p.shape == (2, CFG.plane_channels, CFG.grid_res, CFG.grid_res)


def test_query_feature_shape(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field)
    feat = field.query(planes, points)
    assert feat.shape == (2, 64, CFG.feature_dim)


def test_splat_decoder_ranges_and_shapes(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field)
    s = field.decode_splats(planes, points)
    assert s.positions.shape == (2, 64, 3)
    assert s.scales.shape == (2, 64, 3)
    assert torch.all(s.scales > 0)
    assert torch.all((s.opacities >= 0) & (s.opacities <= 1))
    assert torch.all((s.colors >= 0) & (s.colors <= 1))
    # rotations are unit quaternions
    norms = s.rotations.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # splat position stays near its query point (bounded offset)
    assert torch.all((s.positions - points).abs() <= 0.1 + 1e-4)


def test_sdf_shape(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field)
    sdf = field.decode_sdf(planes, points)
    assert sdf.shape == (2, 64, 1)
    assert torch.isfinite(sdf).all()


def test_pbr_ranges(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field)
    pbr = field.decode_pbr(planes, points)
    assert torch.all((pbr.albedo >= 0) & (pbr.albedo <= 1))
    assert torch.all((pbr.roughness >= 0) & (pbr.roughness <= 1))
    assert torch.all((pbr.metallic >= 0) & (pbr.metallic <= 1))
    normal_norms = pbr.normal.norm(dim=-1)
    assert torch.allclose(normal_norms, torch.ones_like(normal_norms), atol=1e-5)


def test_multiview_consistency_is_view_independent(field: SpatialLatentField) -> None:
    """The core guarantee: same object-space point -> identical surface, no
    matter the query order or how the 'camera' reached it."""
    field.eval()
    cond = torch.randn(1, CFG.cond_dim)
    planes = field.encode(cond)
    pts = torch.empty(1, 32, 3).uniform_(-0.9, 0.9)

    # "Front camera" queries the points directly.
    front = field.decode_pbr(planes, pts)
    # "Side camera" reaches the same points in a shuffled order.
    perm = torch.randperm(32)
    side = field.decode_pbr(planes, pts[:, perm])

    assert torch.allclose(front.albedo[:, perm], side.albedo, atol=1e-6)
    assert torch.allclose(front.roughness[:, perm], side.roughness, atol=1e-6)


def test_different_conditioning_gives_different_field(field: SpatialLatentField) -> None:
    a = field.encode(torch.randn(1, CFG.cond_dim))
    b = field.encode(torch.randn(1, CFG.cond_dim))
    assert not torch.allclose(a[0], b[0])


def test_render_view_runs(field: SpatialLatentField) -> None:
    planes = field.encode(torch.randn(1, CFG.cond_dim))
    cam_origin = torch.tensor([[0.0, 0.0, -2.0]])
    # A tiny 4x4 pinhole ray bundle pointing down +z.
    ys, xs = torch.meshgrid(torch.linspace(-0.3, 0.3, 4), torch.linspace(-0.3, 0.3, 4), indexing="ij")
    dirs = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.ones(16)), dim=-1)
    dirs = torch.nn.functional.normalize(dirs, dim=-1).unsqueeze(0)
    rgb, depth, hit = field.render_view(planes, cam_origin, dirs, steps=8)
    assert rgb.shape == (1, 16, 3)
    assert depth.shape == (1, 16)
    assert hit.shape == (1, 16)
    assert torch.isfinite(rgb).all()
    assert torch.all((rgb >= 0) & (rgb <= 1))


def test_gradients_flow_through_all_heads(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field, b=1, n=16)
    sdf = field.decode_sdf(planes, points)
    pbr = field.decode_pbr(planes, points)
    splats = field.decode_splats(planes, points)
    loss = sdf.mean() + pbr.albedo.mean() + splats.opacities.mean()
    loss.backward()
    grads = [p.grad for p in field.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_builder_helper() -> None:
    f = build_spatial_field(cond_dim=16, grid_res=8, plane_channels=4, feature_dim=8)
    assert f.cfg.cond_dim == 16
    assert f.cfg.grid_res == 8


def test_invalid_grid_res_rejected() -> None:
    with pytest.raises(ValueError):
        TriPlaneConfig(grid_res=24)  # not a power of two


# ==================================================================================================
# test_view_dependent_render: Tests for wiring SH view-dependent radiance into the NeuS renderer (3D).
# ==================================================================================================


CFG_view_dependent_render = TriPlaneConfig(cond_dim=16, plane_channels=8, grid_res=16, feature_dim=16)


def _setup(sh: bool):
    torch.manual_seed(0)
    field = SpatialLatentField(CFG_view_dependent_render)
    decoder = SHRadianceDecoder(feature_dim=CFG_view_dependent_render.feature_dim, degree=3) if sh else None
    renderer = NeusRenderer(n_samples=24, sh_decoder=decoder)
    planes = field.encode(torch.zeros(1, CFG_view_dependent_render.cond_dim))
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -2.0]), torch.zeros(3), height=6, width=6)
    return field, renderer, planes, o, d


def test_sh_renderer_runs_and_is_bounded():
    field, renderer, planes, o, d = _setup(sh=True)
    out = renderer.render(field, planes, o, d, near=0.5, far=3.5)
    assert out["rgb"].shape == (36, 3)
    assert torch.isfinite(out["rgb"]).all()
    assert torch.all((out["rgb"] >= 0) & (out["rgb"] <= 1))


def test_sh_decoder_receives_gradient():
    field, renderer, planes, o, d = _setup(sh=True)
    out = renderer.render(field, planes, o, d, near=0.5, far=3.5)
    out["rgb"].mean().backward()
    assert any(p.grad is not None for p in renderer.sh_decoder.parameters())


def test_sh_path_differs_from_flat_albedo():
    # Same field + camera, with and without the SH decoder, should differ:
    # confirms the view-dependent path is actually engaged.
    field, renderer_sh, planes, o, d = _setup(sh=True)
    renderer_flat = NeusRenderer(n_samples=24)  # no sh_decoder
    rgb_sh = renderer_sh.render(field, planes, o, d, near=0.5, far=3.5)["rgb"]
    rgb_flat = renderer_flat.render(field, planes, o, d, near=0.5, far=3.5)["rgb"]
    assert not torch.allclose(rgb_sh, rgb_flat)


def test_flat_renderer_unchanged_by_default():
    # Default renderer (no decoder) still returns valid albedo colors.
    field, renderer, planes, o, d = _setup(sh=False)
    assert renderer.sh_decoder is None
    out = renderer.render(field, planes, o, d, near=0.5, far=3.5)
    assert torch.all((out["rgb"] >= 0) & (out["rgb"] <= 1))


# ==================================================================================================
# test_perfect_model_helpers: Tests for Dense Diffusion wiring, glyph projector, hand/count helpers.
# ==================================================================================================


def test_region_masks_and_bias():
    from frontier.layout import AttentionLayoutPlan, bias_cross_attention, region_masks_from_boxes

    masks = region_masks_from_boxes(((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)), latent_h=8, latent_w=8)
    assert masks.shape == (2, 1, 8, 8)
    plan = AttentionLayoutPlan(
        region_names=("a", "b"),
        boxes=((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)),
        enforce_steps=(0, 1, 2),
        strength=0.9,
    )
    logits = torch.zeros(2, 4, 64, 16)
    out = bias_cross_attention(
        logits,
        region_masks=masks,
        step_index=0,
        plan=plan,
        cond_batch_size=1,
    )
    assert out.shape == logits.shape
    # Cond row biased; uncond (batch 1) unchanged
    assert not torch.allclose(out[0], logits[0])
    assert torch.allclose(out[1], logits[1])


def test_bias_pools_latent_masks_to_patch_grid():
    """Latent 8x8 masks vs 4x4 patch tokens must pool spatially (not flat-crop)."""
    from frontier.layout import AttentionLayoutPlan, bias_cross_attention, region_masks_from_boxes

    # Region A: top-left quarter of 8x8 latent; region B: bottom-right quarter.
    masks = region_masks_from_boxes(((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)), latent_h=8, latent_w=8)
    plan = AttentionLayoutPlan(
        region_names=("a", "b"),
        boxes=((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)),
        enforce_steps=(0,),
        strength=1.0,
    )
    # 4x4 patch grid => 16 spatial tokens
    logits = torch.zeros(1, 1, 16, 8)
    out = bias_cross_attention(
        logits,
        region_masks=masks,
        step_index=0,
        plan=plan,
        num_patch_tokens=16,
    )
    # Text chunk 0 gets region A (top-left of 4x4); chunk 1 gets region B (bottom-right).
    # Token layout is row-major: indices 0..3 top row, 12..15 bottom row.
    bias_a = (out[0, 0, :, 0] - logits[0, 0, :, 0]).view(4, 4)
    bias_b = (out[0, 0, :, 4] - logits[0, 0, :, 4]).view(4, 4)
    assert bias_a[:2, :2].mean() > bias_a[2:, 2:].mean()
    assert bias_b[2:, 2:].mean() > bias_b[:2, :2].mean()
    # Opposite corners should stay near-zero for the mismatched region
    assert float(bias_a[3, 3]) < float(bias_a[0, 0])
    assert float(bias_b[0, 0]) < float(bias_b[3, 3])


def test_layout_attn_context_cross_attention():
    from frontier.layout import AttentionLayoutPlan, region_masks_from_boxes
    from models.attention import clear_layout_attn, set_layout_attn
    from models.dit_text import CrossAttention

    attn = CrossAttention(hidden_size=32, num_heads=4, text_dim=32, qk_norm=False)
    x = torch.randn(1, 16, 32)
    text = torch.randn(1, 8, 32)
    plan = AttentionLayoutPlan(
        region_names=("r0",),
        boxes=((0.0, 0.0, 1.0, 1.0),),
        enforce_steps=tuple(range(10)),
        strength=1.0,
    )
    masks = region_masks_from_boxes(plan.boxes, latent_h=4, latent_w=4)
    set_layout_attn(plan=plan, region_masks=masks, step_index=0, cond_batch_size=1, num_patch_tokens=16)
    try:
        out = attn(x, text, use_xformers=False)
        assert out.shape == x.shape
    finally:
        clear_layout_attn()


def test_hand_structure_score_in_unit_range():
    from utils.quality.critics import score_hand_structure

    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img[80:, :, :] = np.clip(
        img[80:, :, :].astype(np.int16) + np.random.randint(-40, 40, img[80:].shape),
        0,
        255,
    ).astype(np.uint8)
    s = score_hand_structure(img)
    assert 0.0 <= s <= 1.0


def test_frontier_consume_sanitize_and_emphasis():
    from types import SimpleNamespace

    from utils.generation.guidance import (
        apply_token_emphasis_to_args,
        merge_step_emphasis_into_noise_scales,
        sanitize_diffusion_feature_kwargs,
    )

    args = SimpleNamespace(prompt="a woman with detailed hands holding a sign", cfg_scale=4.0)
    meta = apply_token_emphasis_to_args(args)
    assert "weights" in meta
    assert "hands" in str(meta["weights"]).lower() or "(hands" in args.prompt.lower()
    assert args.cfg_scale > 4.0

    merged = merge_step_emphasis_into_noise_scales(
        step_emphasis=[2.0, 1.0, 0.5],
        step_noise_scales=[1.0, 1.0, 1.0],
    )
    assert merged is not None and len(merged) == 3
    assert merged[0] > merged[-1]

    clean = sanitize_diffusion_feature_kwargs(
        {
            "step_emphasis": [1.5, 1.0],
            "step_noise_scales": [1.0, 1.0],
            "frontier_token_emphasis": {"hands": 1.2},
            "attention_layout_plan": object(),
        }
    )
    assert "step_emphasis" not in clean
    assert "frontier_token_emphasis" not in clean
    assert "step_noise_scales" in clean
    assert "attention_layout_plan" in clean


def test_layout_attn_on_qk_norm():
    from frontier.layout import AttentionLayoutPlan, region_masks_from_boxes
    from models.attention import clear_layout_attn, set_layout_attn
    from models.dit_text import CrossAttention

    attn = CrossAttention(hidden_size=32, num_heads=4, text_dim=32, qk_norm=True, qk_norm_eps=1e-6)
    x = torch.randn(1, 16, 32)
    text = torch.randn(1, 8, 32)
    plan = AttentionLayoutPlan(
        region_names=("r0",),
        boxes=((0.0, 0.0, 1.0, 1.0),),
        enforce_steps=tuple(range(10)),
        strength=1.0,
    )
    masks = region_masks_from_boxes(plan.boxes, latent_h=4, latent_w=4)
    set_layout_attn(plan=plan, region_masks=masks, step_index=0, cond_batch_size=1, num_patch_tokens=16)
    try:
        out = attn(x, text, use_xformers=False)
        assert out.shape == x.shape
    finally:
        clear_layout_attn()
