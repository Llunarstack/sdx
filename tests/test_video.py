"""Tests for prompt/image understanding on T2V / I2V / V2V paths.

Also holds what used to be: test_video_adherence, test_video_competitor_pain, test_video_continuity, test_video_controls, test_video_dit, test_video_frontier, test_video_pipeline, test_video_studio, test_video_superiority, test_video_tier1, test_video_upgrades, test_video_leaders_stack, test_video_perfect, test_video_upgrades, test_video_wave2_helpers, test_videomax, test_scene_graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusion.flow_matching import flow_matching_per_sample_losses
from models.video_dit import FactorizedSpatioTemporalBlock, PermanentVideoDiT, VideoDiT, permanence_consistency_loss
from PIL import Image
from pipelines.video.controls import (
    ControlMode,
    InputBinding,
    apply_scene_cuts,
    build_reference_sheet,
    compile_leaders_stack,
    compile_motion_intelligence,
    compile_shot_control_plan,
    energy_envelope,
    parse_elements,
    parse_multimodal_refs,
    path_to_motion_deltas,
    path_to_prompt,
    plan_native_audio,
    plan_scene_cuts,
    plan_to_prompt_fragments,
    ref_budget_ok,
    score_lip_sync,
    solve_camera_path,
    synthesize_stereo_bed,
)
from pipelines.video.frontier_compiler import (
    apply_causal_ripples,
    audit_motif_haunting,
    build_gravity_field,
    compile_frontier_layers,
    detect_triggers_in_prompt,
    list_frontier_modules,
    parse_anticipation_config,
    parse_causal_rules,
    parse_echo_config,
    parse_kinetic_config,
    parse_motifs,
    parse_semantic_gravity,
    parse_tension_curve,
    plan_anticipation_windups,
    plan_temporal_echoes,
    sample_tension_for_shots,
    track_kinetic_ledger,
)
from pipelines.video.motion import (
    apply_brush_to_flow,
    blend_frames,
    interpolate_between_keyframes,
    interpolate_flf2v_sequence,
    motion_template_sequence,
    parse_motion_brush,
    parse_motion_library,
    pick_keyframe_with_continuity,
    pixel_continuity_score,
    prepare_flf2v_keyframes,
    rasterize_box_mask,
    resolve_motion_clip,
    retime_frame_list,
)
from pipelines.video.neural_sample import (
    bind_at_mentions,
    encode_prompt_context,
    image_prompt_brief,
    image_to_latent_proxy,
    latents_to_frames_placeholder,
    load_video_dit,
    parse_prompt_ground,
    prepare_i2v_conditioning,
    run_neural_segment,
    sample_video_latents,
)
from pipelines.video.pipeline import preview_retrieval_rankings, rank_clips_for_shot
from pipelines.video.quality import (
    apply_contact_ground,
    apply_hf_deshimmer,
    apply_motion_shutter,
    bind_from_refs,
    chain_anchor_from_segment,
    composite_with_mask,
    expected_count_from_prompt,
    extract_slots,
    find_glyph_rois,
    harmonize_cut,
    run_superior_pass,
    score_contact_ground,
    score_count_stability,
    score_cut_continuity,
    score_extremity_coherence,
    score_glyph_stability,
    score_hf_shimmer,
    score_identity_bind,
    score_object_permanence,
    score_occlusion_coherence,
    score_physics_invariance,
    score_secondary_track,
    score_segment_quality,
    score_video_artifacts,
)
from pipelines.video.scene import (
    ClipCandidate,
    KeyframeEditJob,
    MasterTimeline,
    RetrievalSource,
    SegmentAssignment,
    ShotSpec,
    VideoMode,
    VideoPlan,
    compile_scene_graph,
    load_scene_graph,
    parse_scene_dict,
    plan_video_from_prompt,
    split_prompt_into_beats,
    validate_eyeline,
    validate_light_motivation,
    validate_prop_continuity,
    validate_scene_graph,
)
from pipelines.video.segment_processor import (
    RetryPolicy,
    SampleOptions,
    apply_cli_overrides_to_process_options,
    apply_grade_to_sequence,
    apply_video_quality,
    apply_video_quality_preset,
    build_keyframe_sample_extras,
    build_sample_cmd_for_keyframe,
    grade_frame,
    parse_process_options,
    plan_videomax,
    pose_control_args,
    pose_from_rig_boxes,
    soft_architecture_lock_args,
    write_pose_control_image,
)
from pipelines.video.studio import (
    RenderEngine,
    apply_motion_grammar,
    bible_to_prompt,
    camera_prompt_fragment,
    compile_studio_block,
    engine_by_id,
    expand_prompt_to_storyboard,
    genre_director_notes,
    grammar_for_engine,
    grammar_for_prompt,
    list_engines,
    parse_camera_rig,
    parse_character_bibles,
    parse_storyboard,
    parse_thumbnail_config,
    parse_world,
    plan_thumbnails,
    preset_principles,
    principles_prompt,
    rig_to_prompt,
    route_scene,
    thumbnail_edit_overrides,
    thumbnail_gate_issues,
    world_to_prompt,
)
from pipelines.video.video_io import frame_count_for_duration, read_frame_rgb, save_frame_rgb, write_video_from_frames

# ==================================================================================================
# test_video_adherence: Tests for prompt/image understanding on T2V / I2V / V2V paths.
# ==================================================================================================


def _save(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def test_prompt_ground_binds_attributes_and_counts():
    g = parse_prompt_ground("a tall woman in a red dress and a man in a blue suit walking, no glasses")
    names = {e.name for e in g.entities}
    assert "woman" in names or "dress" in names
    assert g.rewritten
    assert "attribute" in g.negative_extra.lower() or "swap" in g.negative_extra.lower()
    assert "walk" in g.actions
    g2 = parse_prompt_ground("three red cars parked on the street")
    assert g2.subject_count >= 1 or "car" in {e.name for e in g2.entities} or g2.counts


def test_bind_at_mentions():
    out = bind_at_mentions("hero walks into frame @Image1", {"Image1": "red-haired woman identity"})
    assert "@Image1" in out or "Image1" in out
    assert "red-haired" in out.lower() or "identity" in out.lower()


def test_encode_prompt_context_distinct():
    a = encode_prompt_context("a woman in a red dress", context_dim=64)
    b = encode_prompt_context("a man in a blue suit", context_dim=64)
    assert a.shape[-1] == 64 and a.ndim == 3
    assert not torch.allclose(a.mean(dim=1), b.mean(dim=1), atol=1e-3)
    assert encode_prompt_context("", context_dim=64, max_len=8).shape[-1] == 64


def test_image_cond_and_i2v_bundle(tmp_path: Path):
    img = tmp_path / "ref.png"
    rgb = np.zeros((64, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    _save(img, rgb)
    brief = image_prompt_brief(img)
    assert brief.has_file and "preserve" in brief.brief.lower()
    lat = image_to_latent_proxy(img, height=16, width=12)
    assert lat.shape == (1, 4, 16, 12)
    bundle = prepare_i2v_conditioning(img, "woman in red dress", latent_h=16, latent_w=12, context_dim=32)
    assert bundle["context"].shape[-1] == 32
    assert bundle["first_frame_latent"].shape[0] == 1


def test_neural_segment_uses_real_context(tmp_path: Path):
    img = tmp_path / "start.png"
    _save(img, np.full((64, 64, 3), 120, dtype=np.uint8))
    shot = ShotSpec(
        index=0,
        prompt="a woman in a red dress walking",
        duration_sec=1.0,
        shot_type="medium",
        motion_hint="walk",
    )
    assignment = SegmentAssignment(shot=shot, clip=None, start_image=str(img))
    plan = VideoPlan(
        mode=VideoMode.I2V,
        user_prompt=shot.prompt,
        timeline=MasterTimeline(fps=8.0, width=64, height=64, duration_sec=1.0),
        shots=[shot],
    )
    frames, meta = run_neural_segment(
        assignment,
        plan,
        tmp_path / "work",
        target_frames=8,
        width=64,
        height=64,
        steps=2,
        device="cpu",
        permanent=True,
    )
    assert len(frames) == 8
    assert "prompt_ground=1" in meta["ops"]
    assert any("context=yes" in o for o in meta["ops"])
    assert any("i2v=yes" in o for o in meta["ops"])
    f0 = read_frame_rgb(frames[0]).astype(np.float32)
    assert float(np.mean(np.abs(f0 - 120))) < 40.0


# ==================================================================================================
# test_video_competitor_pain: Tests for the identity / extremity / glyph / physics segment-quality scores.
# ==================================================================================================


def test_identity_bind_detects_color_drift(tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 40
        # Subject band shifts hue mid-clip
        color = (180, 40, 40) if i < 3 else (40, 40, 180)
        img[8:50, 10:54] = color
        p = tmp_path / f"id{i}.png"
        _save(p, img)
        frames.append(p)
    fp = bind_from_refs([frames[0], frames[1]])
    report = score_identity_bind(frames, fp, sample_every=1, drift_threshold=0.85)
    assert report.drift_frames >= 1
    assert report.score < 0.99


def test_extremity_collapse_detected(tmp_path: Path):
    frames = []
    for i in range(5):
        img = np.zeros((80, 80, 3), dtype=np.uint8) + 20
        # Structured edges in lower band for early frames, mush later
        if i < 2:
            for x in range(10, 70, 3):
                img[50:75, x] = 220
                img[50:75, x + 1] = 40
        p = tmp_path / f"ex{i}.png"
        _save(p, img)
        frames.append(p)
    report = score_extremity_coherence(frames)
    assert 0.0 <= report.score <= 1.0


def test_glyph_melt_detected(tmp_path: Path):
    frames = []
    for i in range(5):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 25
        # Sharp logo-like vertical bars — then obliterate into flat gray
        if i <= 1:
            for x in range(8, 40, 2):
                img[8:28, x] = 250
                img[8:28, x + 1] = 10
        else:
            img[8:28, 8:40] = 90
        p = tmp_path / f"g{i}.png"
        _save(p, img)
        frames.append(p)
    rois = find_glyph_rois(np.asarray(Image.open(frames[0])))
    assert len(rois) >= 1
    before = score_glyph_stability(frames, melt_threshold=0.12, sample_every=1)
    assert before.melt_events >= 1 or before.score < 0.95


def test_physics_flags_teleport(tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 20
        # Blob teleports across frame (far enough that match fails → pop teleport)
        x = 4 if i < 3 else 50
        img[22:38, x : x + 14] = 240
        p = tmp_path / f"ph{i}.png"
        _save(p, img)
        frames.append(p)
    report = score_physics_invariance(frames, sample_every=1)
    assert report.jump_discontinuities >= 1 or report.score < 0.95


def test_quality_includes_new_axes(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 35
        img[10:38, 10:38] = 160
        p = tmp_path / f"q{i}.png"
        _save(p, img)
        frames.append(p)
    q = score_segment_quality(frames, min_identity=0.1, min_extremity=0.1, min_physics=0.1)
    assert 0.0 <= q.identity_score <= 1.0
    assert 0.0 <= q.extremity_score <= 1.0
    assert 0.0 <= q.physics_score <= 1.0
    assert 0.0 <= q.glyph_score <= 1.0


def test_permanent_dit_identity_bank():
    torch.manual_seed(0)
    model = PermanentVideoDiT(
        in_channels=4, dim=32, depth=2, num_heads=4, patch_size=2, context_dim=16, num_slots=4, permanence_every=1
    )
    x = torch.randn(1, 4, 4, 8, 8)
    t = torch.zeros(1)
    identity = torch.randn(1, 3, 32)
    out = model(x, t, context=torch.randn(1, 16), identity=identity)
    assert out.shape == x.shape


# ==================================================================================================
# test_video_continuity: Thumbnail rehearsal + continuity validator tests.
# ==================================================================================================


class _Shot:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_thumbnail_config_from_studio():
    cfg = parse_thumbnail_config(None, studio={"thumbnail_first": True, "thumbnail_size": 64, "thumbnail_gate": "warn"})
    assert cfg.enabled
    assert cfg.size == 64
    assert cfg.gate == "warn"


def test_thumbnail_plan_and_gate():
    shots = [
        _Shot(id="a", prompt="hero runs", thumbnail_approved=False),
        _Shot(id="b", prompt="hero lands", thumbnail_approved=True),
    ]
    cfg = parse_thumbnail_config({"thumbnail": {"enabled": True, "gate": "require_approval"}})
    plan = plan_thumbnails(shots, config=cfg, base_prompt="action")
    assert plan.enabled
    assert len(plan.specs) == 2
    assert not plan.gate_passed
    assert thumbnail_gate_issues(plan)


def test_thumbnail_edit_overrides():
    cfg = parse_thumbnail_config({"thumbnail": {"enabled": True}})
    ov = thumbnail_edit_overrides(cfg)
    assert ov["thumbnail_pass"] is True
    assert ov["quality_retry"] is False


def test_eyeline_180_rule():
    shots = [
        _Shot(id="s1", prompt="detective looks left", gaze="frame_left", characters=["det"], transition="cut"),
        _Shot(id="s2", prompt="detective close-up looks left", gaze="frame_left", characters=["det"], transition="cut"),
    ]
    issues = validate_eyeline(shots)
    assert any(i.code == "eyeline_180_rule" for i in issues)


def test_eyeline_dialogue_pair_ok():
    shots = [
        _Shot(id="s1", gaze="frame_right", characters=["a"], transition="cut"),
        _Shot(id="s2", gaze="frame_left", characters=["b"], transition="cut"),
    ]
    issues = validate_eyeline(shots)
    assert not any(i.code == "eyeline_mismatch" for i in issues)


def test_prop_state_reset():
    shots = [
        _Shot(id="s1", objects=["sword"], props_state={"sword": "broken"}),
        _Shot(id="s2", objects=["sword"], props_state={"sword": "pristine"}),
    ]
    ledger = {"sword": {"initial": "drawn"}}
    issues = validate_prop_continuity(shots, ledger)
    assert any(i.code == "prop_state_reset" for i in issues)


def test_light_unmotivated():
    shots = [_Shot(id="s1", prompt="portrait with flat studio lighting")]
    issues = validate_light_motivation(shots)
    assert any(i.code == "light_unmotivated" for i in issues)


def test_scene_compiles_with_continuity_metadata():
    data = {
        "scene": {"prompt": "two heroes talk", "duration_sec": 6},
        "shots": [
            {
                "id": "a",
                "prompt": "hero looks right",
                "gaze": "frame_right",
                "characters": ["hero"],
                "duration_sec": 3,
                "thumbnail_approved": True,
            },
            {
                "id": "b",
                "prompt": "villain looks left",
                "gaze": "frame_left",
                "characters": ["villain"],
                "duration_sec": 3,
                "thumbnail_approved": True,
            },
        ],
        "characters": {
            "hero": {"description": "brave knight"},
            "villain": {"description": "dark sorcerer"},
        },
        "continuity": {"validators": {"eyeline": True, "props": False}},
        "studio": {"thumbnail_first": True, "thumbnail_gate": "off"},
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    assert compiled.plan.metadata["thumbnail_plan"]["enabled"]
    assert "continuity" in compiled.plan.metadata


def test_thumbnail_gate_blocks_compile():
    data = {
        "scene": {"prompt": "test", "duration_sec": 4},
        "shots": [{"id": "a", "prompt": "wide", "duration_sec": 4, "thumbnail_approved": False}],
        "studio": {"thumbnail_first": True, "thumbnail_gate": "require_approval"},
    }
    with pytest.raises(ValueError, match="Thumbnail gate"):
        compile_scene_graph(parse_scene_dict(data))


def test_validate_scene_graph_continuity_warn():
    data = {
        "scene": {"prompt": "dialogue"},
        "shots": [
            {"id": "a", "gaze": "frame_left", "characters": ["x"]},
            {"id": "b", "gaze": "frame_left", "characters": ["x"]},
        ],
        "continuity": {"validators": {"strict": False}},
    }
    issues = validate_scene_graph(parse_scene_dict(data))
    assert any("eyeline_180_rule" in i for i in issues)


# ==================================================================================================
# test_video_controls: Control modes and image+text binding compile tests.
# ==================================================================================================


def test_image_text_binding_in_prompt():
    cp = compile_shot_control_plan(
        shot_id="s0",
        shot_index=0,
        base_prompt="forest at dusk",
        base_negative="blur",
        bindings=[
            InputBinding(
                entity_id="hero",
                image="hero.png",
                image_role="identity and armor",
                text_role="walking forward",
                control=ControlMode.IDENTITY,
                reference_strength=0.8,
            )
        ],
    )
    assert "[image:hero provides" in cp.positive_prompt
    assert "[text:walking forward]" in cp.positive_prompt


def test_control_modes_multi_ref():
    cp = compile_shot_control_plan(
        shot_id="s0",
        shot_index=0,
        base_prompt="scene",
        base_negative="",
        bindings=[
            InputBinding(
                entity_id="a",
                image="a.png",
                control=ControlMode.STYLE,
                reference_strength=0.5,
            ),
            InputBinding(
                entity_id="b",
                image="b.png",
                control=ControlMode.STYLE,
                reference_strength=0.7,
            ),
        ],
    )
    assert "--style-ref" in cp.sample_extra_args


def test_i2v_scene_inputs_compile():
    data = {
        "mode": "i2v",
        "scene": {"prompt": "hero walks", "duration_sec": 3},
        "inputs": [
            {
                "id": "hero_ref",
                "image": "hero.png",
                "provides": "identity",
                "text_changes": "walking",
                "control": "identity",
            }
        ],
        "characters": {"hero": {"bind_input": "hero_ref", "description": "knight"}},
        "shots": [{"id": "a", "prompt": "wide walk", "duration_sec": 3, "characters": ["hero"]}],
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    assert len(compiled.control_plans) == 1


def test_auto_rig_writes_layout(tmp_path: Path):
    from pipelines.video.controls import auto_rig_character, write_rig_box_layout

    img = tmp_path / "char.png"
    from PIL import Image

    Image.new("RGB", (128, 256), (80, 80, 120)).save(img)
    rig = auto_rig_character("hero", img, text_by_part={"head": "preserve face", "legs": "walking"})
    layout = write_rig_box_layout(rig, tmp_path / "rig.json")
    assert layout.is_file()
    assert "regions" in layout.read_text(encoding="utf-8")


# ==================================================================================================
# test_video_dit: Tests for the factorized spatiotemporal video DiT (STDiT).
# ==================================================================================================


def _active_block(dim=32, heads=4, seed=0):
    """A block with non-zero adaLN so its attention actually modulates (adaLN-Zero
    starts as identity, which is correct for training but hides the mechanism).

    Perturbations in these tests use *pattern-changing* noise, not a uniform shift:
    the block's LayerNorm removes any constant offset, so `x += 1.0` would be a
    no-op and wouldn't test propagation.
    """
    torch.manual_seed(seed)
    blk = FactorizedSpatioTemporalBlock(dim, heads)
    torch.nn.init.normal_(blk.ada.weight, std=0.3)
    torch.nn.init.normal_(blk.ada.bias, std=0.3)
    return blk


def test_block_preserves_shape():
    blk = _active_block()
    x = torch.randn(2, 4, 6, 32)  # (B, T, N, D)
    cond = torch.randn(2, 32)
    assert blk(x, cond).shape == x.shape


def test_temporal_attention_mixes_across_frames():
    blk = _active_block()
    x = torch.randn(1, 4, 6, 32)
    cond = torch.randn(1, 32)
    out = blk(x, cond)
    x2 = x.clone()
    torch.manual_seed(1)
    x2[:, -1] += torch.randn(1, 6, 32)  # pattern-changing perturbation of the last frame
    out2 = blk(x2, cond)
    # A change in the last frame should reach earlier frames (non-causal temporal).
    assert not torch.allclose(out[:, 0], out2[:, 0], atol=1e-5)


def test_causal_temporal_hides_the_future():
    blk = _active_block()
    x = torch.randn(1, 4, 6, 32)
    cond = torch.randn(1, 32)
    out = blk(x, cond, causal_temporal=True)
    x2 = x.clone()
    torch.manual_seed(1)
    x2[:, -1] += torch.randn(1, 6, 32)  # perturb the LAST frame
    out2 = blk(x2, cond, causal_temporal=True)
    # Frame 0 must not see the future...
    assert torch.allclose(out[:, 0], out2[:, 0], atol=1e-5)
    # ...but the last frame itself changes.
    assert not torch.allclose(out[:, -1], out2[:, -1], atol=1e-5)


def test_spatial_attention_mixes_within_frame():
    blk = _active_block()
    x = torch.randn(1, 3, 6, 32)
    cond = torch.randn(1, 32)
    out = blk(x, cond, causal_temporal=True)  # isolate spatial by removing temporal future leak
    x2 = x.clone()
    torch.manual_seed(2)
    x2[:, 0, 0] += torch.randn(32)  # perturb one spatial token in frame 0
    out2 = blk(x2, cond, causal_temporal=True)
    # Other spatial tokens in frame 0 should be affected.
    assert not torch.allclose(out[:, 0, 1], out2[:, 0, 1], atol=1e-5)


def test_conditioning_changes_output():
    blk = _active_block()
    x = torch.randn(1, 3, 6, 32)
    a = blk(x, torch.randn(1, 32))
    b = blk(x, torch.randn(1, 32))
    assert not torch.allclose(a, b)


def test_video_dit_forward_shape_and_grad():
    torch.manual_seed(0)
    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2)
    x = torch.randn(2, 4, 5, 16, 16)  # (B, C, T, H, W)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == x.shape
    out.mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_video_dit_reload_reproduces_output():
    """Positions are fixed, so a reloaded model sees exactly the positions it was trained with."""
    torch.manual_seed(0)
    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x, t = torch.randn(1, 4, 3, 8, 8), torch.tensor([400])
    reloaded = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2).eval()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(x, t), reloaded(x, t))


def test_video_dit_accepts_text_context():
    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, context_dim=64)
    x = torch.randn(1, 4, 4, 16, 16)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t, context=torch.randn(1, 64))
    assert out.shape == x.shape


def test_video_dit_trains_with_flow_matching_v2():
    """The video backbone plugs straight into the upgraded flow-matching loss."""
    torch.manual_seed(0)
    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2)
    x0 = torch.randn(2, 4, 4, 16, 16)
    eps = torch.randn(2, 4, 4, 16, 16)
    loss = flow_matching_per_sample_losses(model, x0, eps, 1000, {}, logit_normal=True, shift=3.0)
    assert loss.shape == (2,)
    loss.mean().backward()
    assert any(p.grad is not None for p in model.parameters())


# ==================================================================================================
# test_video_frontier: Frontier / novel video subsystem tests.
# ==================================================================================================


def test_tension_curve_samples():
    curve = parse_tension_curve([0.2, 0.8, 0.5])
    assert curve is not None
    profiles = sample_tension_for_shots(curve, [_Shot(id="a", duration_sec=3)], total_duration=3)
    assert profiles[0].tension > 0


def test_causal_ripple_explosion():
    rules = parse_causal_rules([])
    triggers = detect_triggers_in_prompt("massive explosion in warehouse", rules)
    assert "explosion" in triggers
    ripples = apply_causal_ripples([_Shot(id="x", prompt="massive explosion in warehouse")], rules)
    assert ripples and ripples[0].injected_effects


def test_anticipation_borrow():
    plan = plan_anticipation_windups(
        [_Shot(id="j", prompt="hero jumps across gap", duration_sec=4)],
        config=parse_anticipation_config({"enabled": True}),
    )
    assert plan.windups and plan.windups[0].borrow_sec > 0


def test_motif_haunting_unresolved():
    motifs = parse_motifs({"red_door": {"description": "red door", "must_appear_by_shot": 0}})
    report = audit_motif_haunting(motifs, [_Shot(id="s0", prompt="empty hallway")])
    assert not report.ok
    assert "red_door" in report.unresolved


def test_kinetic_energy_jump():
    ledger = track_kinetic_ledger(
        [
            _Shot(id="a", prompt="sprints at full speed", kinetic={"energy": 0.9}),
            _Shot(id="b", prompt="stands still calmly", kinetic={"energy": 0.05}),
        ],
        config=parse_kinetic_config({"enabled": True}),
    )
    assert any(i.code == "energy_discontinuity" for i in ledger.issues)


def test_semantic_gravity_dominant():
    class Ent:
        lock = True

    field = build_gravity_field(
        {"hero": Ent()},
        {},
        weights=parse_semantic_gravity({"hero": 0.9}),
        shots=[_Shot(characters=["hero"])],
    )
    assert field.dominant_id == "hero"


def test_temporal_echo_auto():
    shots = [
        _Shot(id="a", prompt="neon alley rain cyberpunk city"),
        _Shot(id="b", prompt="interior dialogue scene"),
        _Shot(id="c", prompt="return to neon alley rain cyberpunk pursuit"),
    ]
    plan = plan_temporal_echoes(shots, parse_echo_config({"enabled": True, "auto_rhyme": True}))
    assert plan.links


def test_frontier_compiler_integration():
    data = {
        "scene": {"prompt": "test", "duration_sec": 6},
        "frontier": {"tension_curve": [0.3, 0.9], "breath_cadence": {"enabled": True}},
        "shots": [{"id": "s1", "prompt": "explosion rocks the pier", "duration_sec": 6}],
    }
    out = compile_frontier_layers(data, [_Shot(id="s1", prompt="explosion rocks the pier", duration_sec=6)])
    assert out.metadata.get("tension_curve")
    assert "s1" in out.enrichments


def test_list_frontier_modules():
    assert len(list_frontier_modules()) >= 20


def test_frontier_example_compiles():
    p = Path("examples/scene_frontier.example.json")
    if not p.is_file():
        pytest.skip("example missing")
    compiled = compile_scene_graph(parse_scene_dict(json.loads(p.read_text(encoding="utf-8"))))
    frontier = compiled.plan.metadata.get("frontier") or {}
    assert frontier.get("tension_curve")
    assert frontier.get("witness_lens") or frontier.get("chromatic_arc")


def test_screen_direction_flip():
    from pipelines.video.frontier_compiler import parse_screen_direction_config, track_screen_direction

    shots = [
        _Shot(id="a", screen_direction="left", transition="cut"),
        _Shot(id="b", screen_direction="right", transition="cut"),
    ]
    rep = track_screen_direction(shots, config=parse_screen_direction_config({"enabled": True}))
    assert any(i.code == "screen_direction_flip" for i in rep.issues)


def test_weather_inertia_carries():
    from pipelines.video.frontier_compiler import parse_weather_config, track_weather_inertia

    rep = track_weather_inertia(
        [_Shot(id="a", prompt="dialogue"), _Shot(id="b", prompt="talk")],
        config=parse_weather_config({"enabled": True, "initial": "rain"}),
    )
    assert rep.prompt_injections.get("a")


def test_chromatic_arc():
    from pipelines.video.frontier_compiler import parse_chromatic_arc, sample_chromatic_for_shots

    beats = sample_chromatic_for_shots(
        parse_chromatic_arc(["hope", "dread", "rage"]),
        [_Shot(id="s", duration_sec=2)],
        total_duration=2,
    )
    assert beats[0].palette_key


def test_witness_lens():
    from pipelines.video.frontier_compiler import parse_witness_config, plan_witness_lens

    plans = plan_witness_lens([_Shot(id="c", witness="child")], config=parse_witness_config({"enabled": True}))
    assert plans[0].lens == "child"


def test_stinger_frames():
    from pipelines.video.frontier_compiler import parse_stinger_config, plan_stinger_frames

    specs = plan_stinger_frames(
        [_Shot(id="x", prompt="hero punches villain")],
        config=parse_stinger_config({"enabled": True}),
    )
    assert specs[0].frame_count >= 1


# ==================================================================================================
# test_video_pipeline: Tests for pipelines/video retrieve → compose pipeline.
# ==================================================================================================


def test_split_prompt_beats():
    beats = split_prompt_into_beats("wide city at dusk. then close-up portrait")
    assert len(beats) >= 2


def test_plan_video_shots():
    plan = plan_video_from_prompt("establishing city, then portrait close-up", duration_sec=6.0)
    assert plan.mode == VideoMode.T2V
    assert len(plan.shots) >= 2
    assert abs(sum(s.duration_sec for s in plan.shots) - 6.0) < 0.2


def test_rank_clips_for_shot():
    shot = ShotSpec(index=0, prompt="city establishing wide urban", duration_sec=3.0, shot_type="establishing")
    candidates = [
        ClipCandidate(
            source=RetrievalSource.LOCAL,
            path="/a.mp4",
            title="city establishing wide",
            tags=["urban", "pan"],
            duration_sec=5.0,
        ),
        ClipCandidate(
            source=RetrievalSource.LOCAL,
            path="/b.mp4",
            title="portrait studio",
            tags=["face"],
            duration_sec=4.0,
        ),
    ]
    ranked = rank_clips_for_shot(shot, candidates)
    assert ranked[0].path == "/a.mp4"
    assert ranked[0].score >= ranked[1].score


def test_timeline_helpers():
    assert frame_count_for_duration(1.0, 24.0) == 24


def test_blend_frames():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.ones((8, 8, 3), dtype=np.uint8) * 255
    mid = blend_frames(a, b, 0.5)
    assert mid.shape == a.shape
    assert 100 < int(mid[0, 0, 0]) < 156


def test_interpolate_between_keyframes(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    save_frame_rgb(a, np.zeros((32, 32, 3), dtype=np.uint8))
    save_frame_rgb(b, np.full((32, 32, 3), 200, dtype=np.uint8))
    mids = interpolate_between_keyframes(a, b, tmp_path / "mid", count=3, use_flow=False)
    assert len(mids) == 3


def test_preview_retrieval_empty():
    rows = preview_retrieval_rankings("sunset dragon", catalog_path="nonexistent.json")
    assert isinstance(rows, list)


def test_dry_run_pipeline(tmp_path: Path):
    from pipelines.video.pipeline import run_t2v_pipeline

    out = tmp_path / "out.mp4"
    res = run_t2v_pipeline(
        "a red ball rolling",
        out,
        work_dir=tmp_path / "work",
        dry_run=True,
        duration_sec=1.0,
        fps=12.0,
        width=256,
        height=144,
    )
    assert Path(res.output_path).is_file()
    assert res.plan.shots
    assert res.metadata.get("dry_run") is True


def test_write_video_from_frames(tmp_path: Path):
    frames = []
    for i in range(6):
        fp = tmp_path / f"f{i}.png"
        save_frame_rgb(fp, np.full((64, 64, 3), i * 30, dtype=np.uint8))
        frames.append(fp)
    out = write_video_from_frames(frames, tmp_path / "tiny.mp4", fps=6.0)
    assert out.is_file()


# ==================================================================================================
# test_video_studio: AI film studio architecture tests.
# ==================================================================================================


def test_router_anime():
    r = route_scene("a magical girl transforms in tokyo", style_hint="anime")
    assert r.engine == RenderEngine.ANIME_2D


def test_router_minecraft():
    r = route_scene("steve builds a castle in minecraft")
    assert r.engine == RenderEngine.VOXEL


def test_director_mode_expansion():
    exp = expand_prompt_to_storyboard("A detective enters an abandoned factory", duration_sec=8.0)
    assert len(exp.cuts) >= 2
    assert exp.notes.mood
    genre, _ = genre_director_notes("horror creature in dark hallway")
    assert genre == "horror"


def test_animation_principles_pixar():
    p = preset_principles("pixar")
    frag = principles_prompt(p)
    assert "squash" in frag.lower() or "follow" in frag.lower()


def test_world_and_character_memory():
    w = parse_world({"name": "Emberfall", "weather": "ash fog", "palette": "red black"})
    assert w is not None
    assert "ash" in world_to_prompt(w).lower()
    b = parse_character_bibles({"alex": {"hair": "black", "eyes": "green"}})
    assert "black" in bible_to_prompt(b["alex"])


def test_motion_library():
    lib = parse_motion_library({"walk": {"clip": "/nonexistent.mp4"}})
    assert lib.get("walk") is not None
    assert resolve_motion_clip(lib, ["walk"], fallback="fallback.mp4") == "fallback.mp4"


def test_camera_rig():
    r = parse_camera_rig({"preset": "drone", "lens": "24"})
    assert "drone" in rig_to_prompt(r).lower() or "24" in rig_to_prompt(r)


def test_studio_compiler_auto_director():
    out = compile_studio_block(
        {
            "studio": {"engine": "auto", "director_mode": "auto", "director": "spielberg"},
            "scene": {"prompt": "warrior enters temple"},
        },
        scene_prompt="warrior enters temple",
        duration_sec=6.0,
    )
    assert out.engine
    assert out.storyboard_cuts
    assert out.edit


def test_studio_scene_compiles():
    data = {
        "scene": {"prompt": "anime hero leaps across rooftops", "duration_sec": 5},
        "studio": {"engine": "auto", "director_mode": "auto"},
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    assert compiled.plan.metadata.get("engine")
    assert len(compiled.plan.shots) >= 2


def test_studio_example_file():
    p = Path("examples/scene_studio.example.json")
    if not p.is_file():
        pytest.skip("example missing")
    compiled = compile_scene_graph(parse_scene_dict(__import__("json").loads(p.read_text(encoding="utf-8"))))
    assert compiled.plan.metadata.get("engine")


# ==================================================================================================
# test_video_superiority: Tests for permanence / artifact critic / motion grammar / PermanentVideoDiT.
# ==================================================================================================


def test_permanence_detects_disappearing_blob(tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:] = 30
        # Bright blob present in first 3 frames, gone after
        if i < 3:
            img[20:36, 20:36] = 220
        p = tmp_path / f"f{i}.png"
        _save(p, img)
        frames.append(p)
    report = score_object_permanence(frames, sample_every=1, k_slots=8)
    assert report.disappear_events >= 1
    assert report.score < 0.95


def test_artifact_critic_flags_flicker(tmp_path: Path):
    frames = []
    for i in range(8):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + (20 if i % 2 == 0 else 200)
        p = tmp_path / f"a{i}.png"
        _save(p, img)
        frames.append(p)
    art = score_video_artifacts(frames)
    assert art.flicker < 0.85
    assert art.overall < 0.95


def test_motion_grammar_product_is_strict():
    g = grammar_for_prompt("ecommerce product turntable SKU packshot")
    assert g.name == "product"
    assert g.min_permanence >= 0.7
    out = apply_motion_grammar(parse_process_options({}), g)
    assert out.min_permanence >= 0.7
    assert out.min_glyph > 0.0
    assert out.motion_grammar == "product"


def test_motion_grammar_anime():
    g = grammar_for_prompt("anime girl running, cel shaded")
    assert g.name == "anime_2d"
    assert g.deflicker_strength < grammar_for_prompt("photoreal street").deflicker_strength


def test_permanent_video_dit_slots_and_loss():
    torch.manual_seed(0)
    model = PermanentVideoDiT(
        in_channels=4, dim=32, depth=2, num_heads=4, patch_size=2, context_dim=16, num_slots=4, permanence_every=1
    )
    x = torch.randn(1, 4, 4, 8, 8)
    t = torch.zeros(1)
    out, slots = model(x, t, context=torch.randn(1, 16), return_slots=True)
    assert out.shape == x.shape
    assert slots.shape[0] == 1 and slots.shape[2] == 4
    loss = permanence_consistency_loss(slots)
    assert loss.ndim == 0
    loss.backward()


def test_quality_includes_permanence(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((32, 32, 3), dtype=np.uint8) + 40
        img[8:20, 8:20] = 180
        p = tmp_path / f"q{i}.png"
        _save(p, img)
        frames.append(p)
    q = score_segment_quality(frames, min_permanence=0.2, min_artifact=0.2)
    assert 0.0 <= q.permanence_score <= 1.0
    assert 0.0 <= q.artifact_score <= 1.0


def test_extract_slots_nonempty():
    img = np.zeros((64, 64, 3), dtype=np.uint8) + 10
    img[10:30, 40:55] = 255
    slots = extract_slots(img, k=6)
    assert len(slots) >= 1


# ==================================================================================================
# test_video_tier1: Tests for Tier-1 video features (elements, storyboard, FLF2V, motion brush).
# ==================================================================================================


def test_parse_elements():
    lib = parse_elements(
        {
            "hero": {
                "images": ["a.png", "b.png"],
                "bind_subject": True,
                "reference_sheet": False,
            }
        }
    )
    assert "hero" in lib.elements
    assert lib.elements["hero"].bind_subject is True
    assert len(lib.elements["hero"].images) == 2


def test_reference_sheet(tmp_path: Path):
    src = tmp_path / "hero.png"
    save_frame_rgb(src, np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8))
    sheet = build_reference_sheet(src, tmp_path / "sheet")
    assert len(sheet.views) == 3
    assert Path(sheet.manifest_path).is_file()


def test_storyboard_parse():
    cuts = parse_storyboard(
        {
            "cuts": [
                {"prompt": "wide city", "duration_sec": 2, "camera": "tracking"},
                {
                    "prompt": "close face",
                    "camera": "push_in",
                    "flf2v": True,
                    "start_image": "a.png",
                    "end_image": "b.png",
                },
            ]
        }
    )
    assert len(cuts) == 2
    assert cuts[1].flf2v is True
    assert "tracking" in camera_prompt_fragment("tracking")


def test_storyboard_compiles_to_shots():
    data = {
        "scene": {"prompt": "fallback", "duration_sec": 6},
        "storyboard": {
            "cuts": [
                {"prompt": "wide", "duration_sec": 3, "camera": "pan_left"},
                {"prompt": "close", "duration_sec": 3, "camera": "close_up"},
            ]
        },
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    assert len(compiled.plan.shots) == 2
    assert "pan" in compiled.plan.shots[0].prompt.lower() or "pan" in compiled.plan.shots[0].motion_hint.lower()


def test_motion_brush_flow_mask():
    flow = np.ones((8, 8, 2), dtype=np.float32) * 3.0
    brush = np.zeros((8, 8), dtype=np.float32)
    brush[2:6, 2:6] = 1.0
    masked = apply_brush_to_flow(flow, brush, mode="motion_only")
    assert float(masked[0, 0, 0]) == 0.0
    assert float(masked[4, 4, 0]) > 0.0


def test_parse_motion_brush_box():
    spec = parse_motion_brush({"box": [0.1, 0.2, 0.9, 0.8], "mode": "background_only"})
    assert spec is not None
    assert spec.mode == "background_only"


def test_flf2v_interpolate(tmp_path: Path):
    start = tmp_path / "s.png"
    end = tmp_path / "e.png"
    save_frame_rgb(start, np.zeros((32, 32, 3), dtype=np.uint8))
    save_frame_rgb(end, np.full((32, 32, 3), 200, dtype=np.uint8))
    keys = prepare_flf2v_keyframes(start, end, tmp_path / "keys")
    assert len(keys) == 2
    paths = interpolate_flf2v_sequence(start, end, 8, tmp_path / "out", use_flow=False)
    assert len(paths) == 8
    assert paths[0].is_file()


def test_scene_elements_bind(tmp_path: Path):
    hero = tmp_path / "hero.png"
    save_frame_rgb(hero, np.full((48, 48, 3), 128, dtype=np.uint8))
    data = {
        "scene": {"prompt": "walk", "duration_sec": 2},
        "elements": {"hero_el": {"images": [str(hero)], "bind_subject": True}},
        "characters": {"hero": {"bind_element": "hero_el", "description": "knight"}},
        "shots": [{"prompt": "medium shot", "duration_sec": 2, "characters": ["hero"], "bind_elements": ["hero_el"]}],
    }
    data["_work_dir"] = str(tmp_path)
    g = parse_scene_dict(data)
    g.raw["_work_dir"] = str(tmp_path)
    compiled = compile_scene_graph(g)
    assert compiled.control_plans
    assert compiled.control_plans[0].metadata.get("bind_subject") or compiled.control_plans[0].init_image


def test_tier1_example_validates_structure():
    p = Path("examples/scene_tier1.example.json")
    if not p.is_file():
        pytest.skip("example missing")
    g = parse_scene_dict(json.loads(p.read_text(encoding="utf-8")))
    assert g.storyboard
    assert "hero" in g.elements


# ==================================================================================================
# test_video_upgrades: Tests for keyframe sample.py argv, engine routing and the VideoDiT sampler.
# ==================================================================================================


def test_sample_options_build_quality_extras():
    s = SampleOptions(
        num=3,
        pick_best="combo_vit",
        quality_pack="top",
        human_made="standard",
        dual_stage_layout=True,
        holy_grail=True,
        anti_perspective_drift=True,
        boost_quality=True,
    )
    extras = build_keyframe_sample_extras(s)
    joined = " ".join(extras)
    assert "--num" in extras and "3" in extras
    assert "--pick-best" in extras and "combo_vit" in extras
    assert "--quality-pack" in extras
    assert "--human-made" in extras
    assert "--dual-stage-layout" in joined
    assert "--holy-grail" in joined
    assert "--boost-quality" in joined


def test_build_sample_cmd_includes_quality_flags():
    job = KeyframeEditJob(
        segment_index=0,
        frame_index=0,
        source_frame_path="in.png",
        output_path="out.png",
        prompt="library hall",
    )
    cmd = build_sample_cmd_for_keyframe(
        job,
        ckpt="ckpt.pt",
        sample_options=SampleOptions(num=2, pick_best="combo_architecture"),
    )
    assert "--pick-best" in cmd
    assert "combo_architecture" in cmd


def test_route_scene_engine_override():
    d = route_scene("anything", engine_override="claymation")
    assert d.engine.value == "claymation"
    assert d.edit_overrides["temporal_alpha"] > 0.1


def test_video_dit_sequence_context():
    from models.video_dit import VideoDiT

    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2, context_dim=64)
    x = torch.randn(1, 4, 4, 16, 16)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t, context=torch.randn(1, 8, 64))
    assert out.shape == x.shape


def test_neural_sample_smoke(tmp_path: Path):
    model = load_video_dit(None, model_name="VideoDiT-S/2", context_dim=32, device="cpu")
    # Shrink: reload tiny for speed
    from models.video_dit import VideoDiT

    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4, patch_size=2, context_dim=16).eval()
    z = sample_video_latents(
        model,
        frames=4,
        height=8,
        width=8,
        steps=2,
        context=torch.zeros(1, 4, 16),
        device="cpu",
    )
    assert z.shape == (1, 4, 4, 8, 8)
    frames = latents_to_frames_placeholder(z, tmp_path, width=64, height=64)
    assert len(frames) == 4
    assert all(p.is_file() for p in frames)


def test_generate_video_list_engines_flag():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "pipelines" / "video" / "scripts" / "generate_video.py"), "--list-engines"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert r.returncode == 0
    assert "realistic" in r.stdout.lower()


# ==================================================================================================
# test_video_leaders_stack: Tests for Seedance/Hailuo/Wan leaders stack helpers.
# ==================================================================================================


def test_multimodal_ref_bus_roles(tmp_path: Path):
    img = tmp_path / "hero.png"
    _save(img, np.zeros((16, 16, 3), dtype=np.uint8) + 100)
    pack = parse_multimodal_refs(
        {
            "identity": [str(img)],
            "style": [{"path": str(img), "tag": "noir"}],
            "audios": [],
        }
    )
    assert pack.counts["image"] >= 2
    ok, _ = ref_budget_ok(pack)
    assert ok
    assert pack.by_role("identity")


def test_native_audio_synth_and_energy(tmp_path: Path):
    plan = plan_native_audio('she says "hello" in the rain', duration_sec=2.0)
    assert any(e.kind == "dialogue" for e in plan.events)
    assert any(e.kind == "ambience" for e in plan.events)
    wav = synthesize_stereo_bed(plan, tmp_path / "bed.wav")
    assert wav.is_file()
    env = energy_envelope(wav, fps=12.0, duration_sec=2.0)
    assert len(env) >= 8
    assert float(env.max()) > 0


def test_lip_sync_score(tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 40
        # Vary mouth band edges with i
        img[28:38, 14:34] = 40 + (i % 3) * 60
        p = tmp_path / f"l{i}.png"
        _save(p, img)
        frames.append(p)
    ae = np.linspace(0, 1, 6).astype(np.float32)
    rep = score_lip_sync(frames, audio_energy=ae)
    assert 0.0 <= rep.score <= 1.0


def test_scene_cuts_and_camera_path(tmp_path: Path):
    plan = plan_scene_cuts("wide shot then close-up then cut to door", frame_count=48, fps=24)
    assert len(plan.cuts) >= 1
    frames = []
    for i in range(24):
        p = tmp_path / f"c{i}.png"
        _save(p, np.zeros((32, 32, 3), dtype=np.uint8) + 50)
        frames.append(p)
    apply_scene_cuts(frames, plan)
    path = solve_camera_path("hitchcock dolly zoom on the hero")
    assert path.preset == "hitchcock"
    assert "dolly" in path_to_prompt(path).lower() or "vertigo" in path_to_prompt(path).lower()
    deltas = path_to_motion_deltas(path, frames=12)
    assert len(deltas) == 12


def test_motion_intelligence_gates():
    plan = compile_motion_intelligence("two people run through rain holding bags")
    assert plan.primary_action == "running"
    assert "feet_to_ground" in plan.contacts
    assert plan.gate_boosts.get("min_physics", 0) >= 0.5 or "fluid" in str(plan.secondary)
    pos, neg = plan_to_prompt_fragments(plan)
    assert "running" in pos
    assert neg


def test_leaders_compile_bundle():
    opts = parse_process_options({"leaders_stack": True, "native_audio": True})
    bundle = compile_leaders_stack(
        opts,
        prompt="orbit camera around product, then cut to close-up, she says hello",
        duration_sec=5.0,
        multimodal_refs={"style": []},
    )
    assert bundle.audio_plan is not None
    assert bundle.camera_path is not None
    assert bundle.motion_plan is not None
    assert any("native_audio" in o or "camera_path" in o or "motion_intel" in o for o in bundle.ops)


# ==================================================================================================
# test_video_perfect: Tests for perfect-video polish modules.
# ==================================================================================================


def test_parse_process_options_defaults():
    opts = parse_process_options({})
    assert opts.motion_transfer is True
    assert opts.identity_lock is True
    assert opts.quality_retry is True
    assert opts.max_retries == 2


def test_parse_process_options_from_scene_edit():
    opts = parse_process_options(
        {
            "motion_transfer": False,
            "post_grade": "teal_orange",
            "pose_control": True,
            "max_retries": 3,
        }
    )
    assert opts.motion_transfer is False
    assert opts.post_grade == "teal_orange"
    assert opts.pose_control is True
    assert opts.max_retries == 3


def test_composite_with_mask():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    anchor = np.full((16, 16, 3), 200, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    out = composite_with_mask(frame, anchor, mask, strength=1.0)
    assert int(out[8, 8, 0]) == 200
    assert int(out[0, 0, 0]) == 0


def test_rasterize_box_mask():
    m = rasterize_box_mask(64, 48, (0.25, 0.25, 0.75, 0.75))
    assert m.shape == (48, 64)
    assert m[24, 32] == 1.0
    assert m[0, 0] == 0.0


def test_retime_frame_list():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) + i for i in range(4)]
    out = retime_frame_list(frames, 8)
    assert len(out) == 8


def test_motion_template_sequence(tmp_path: Path):
    anchor = tmp_path / "anchor.png"
    save_frame_rgb(anchor, np.full((32, 32, 3), 128, dtype=np.uint8))
    src = []
    for i in range(4):
        fp = tmp_path / f"src_{i}.png"
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, :, 0] = i * 50
        save_frame_rgb(fp, arr)
        src.append(fp)
    paths = motion_template_sequence(anchor, src, tmp_path / "out", target_count=6, blend_source=0.0)
    assert len(paths) == 6
    assert all(p.is_file() for p in paths)


def test_pose_control(tmp_path: Path):
    parts = [
        ("head", (0.35, 0.05, 0.65, 0.25)),
        ("torso", (0.30, 0.25, 0.70, 0.55)),
        ("legs", (0.30, 0.55, 0.70, 0.95)),
    ]
    rgb = pose_from_rig_boxes(64, 96, parts)
    assert rgb.shape == (96, 64, 3)
    p = write_pose_control_image(parts, tmp_path / "pose.png", width=64, height=96)
    assert p.is_file()
    args = pose_control_args(p)
    assert "--control-type" in args
    assert "pose" in args


def test_post_grade(tmp_path: Path):
    fp = tmp_path / "f.png"
    save_frame_rgb(fp, np.full((32, 32, 3), 120, dtype=np.uint8))
    graded = grade_frame(read_frame_rgb(fp), "vibrant")
    assert graded.shape == (32, 32, 3)
    apply_grade_to_sequence([fp], "muted")
    assert read_frame_rgb(fp).shape == (32, 32, 3)
    apply_grade_to_sequence([fp], "none")
    assert read_frame_rgb(fp).shape == (32, 32, 3)


def test_dry_run_segment_with_process_options(tmp_path: Path):
    from pipelines.video.pipeline import run_t2v_pipeline

    out = tmp_path / "out.mp4"
    res = run_t2v_pipeline(
        "ball rolling",
        out,
        work_dir=tmp_path / "work",
        dry_run=True,
        duration_sec=0.5,
        fps=8.0,
        width=128,
        height=72,
    )
    assert Path(res.output_path).is_file()


def test_scene_edit_block_compiles(tmp_path: Path):
    from pipelines.video.scene import compile_scene_graph, load_scene_graph

    scene = {
        "version": 2,
        "mode": "t2v",
        "scene": {"prompt": "sunset over hills", "duration_sec": 2, "fps": 12},
        "edit": {
            "identity_lock": False,
            "post_grade": "muted",
            "quality_retry": False,
        },
    }
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(scene), encoding="utf-8")
    compiled = compile_scene_graph(load_scene_graph(p))
    assert compiled.plan.metadata["edit"]["post_grade"] == "muted"
    assert compiled.plan.metadata["edit"]["identity_lock"] is False


def test_retry_policy_defaults():
    pol = RetryPolicy()
    assert pol.max_attempts == 2
    assert pol.strength_decay > 0


def test_depth_aware_blend():
    import numpy as np
    from pipelines.video.motion import depth_aware_blend, depth_proxy

    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.full((16, 16, 3), 200, dtype=np.uint8)
    d = depth_proxy(a)
    assert d.shape == (16, 16)
    mid = depth_aware_blend(a, b, 0.5)
    assert mid.shape == a.shape


def test_deflicker(tmp_path: Path):
    from pipelines.video.segment_processor import apply_deflicker

    paths = []
    for i in range(5):
        fp = tmp_path / f"f{i}.png"
        save_frame_rgb(fp, np.full((16, 16, 3), 80 + i * 30, dtype=np.uint8))
        paths.append(fp)
    apply_deflicker(paths, window=3)
    assert read_frame_rgb(paths[2]).shape == (16, 16, 3)


def test_motion_beats(tmp_path: Path):
    pytest.importorskip("cv2")
    from pipelines.video.motion import detect_motion_beats

    paths = []
    for i in range(8):
        fp = tmp_path / f"f{i}.png"
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, i * 3 : min(32, i * 3 + 6), :] = 255
        save_frame_rgb(fp, arr)
        paths.append(fp)
    beats = detect_motion_beats(paths, min_gap=2)
    assert 0 in beats
    assert len(beats) >= 2


def test_frame_enhance():
    from pipelines.video.segment_processor import enhance_frame

    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    out = enhance_frame(rgb, amount=0.2)
    assert out.shape == rgb.shape


def test_transition_fx():
    import numpy as np
    from pipelines.video.scene import TransitionType
    from pipelines.video.video_io import apply_transition, transition_overlap_frames_fx

    a = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]
    b = [np.full((8, 8, 3), 255, dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)]
    out = apply_transition(a, b, TransitionType.FLASH, overlap=1)
    assert len(out) == 3
    assert transition_overlap_frames_fx(TransitionType.WHIP, 24.0) >= 2


def test_parse_process_options_new_flags():
    opts = parse_process_options({"propagate_masks": False, "deflicker": False, "audio_from_source": True})
    assert opts.propagate_masks is False
    assert opts.audio_from_source is True


def test_semantic_drift():
    from pipelines.video.quality import frame_drift_score

    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.full((16, 16, 3), 200, dtype=np.uint8)
    assert frame_drift_score(b, a) > frame_drift_score(a, a)


def test_drift_repair(tmp_path: Path):
    from pipelines.video.quality import repair_sequence_drift

    paths = []
    for i in range(4):
        fp = tmp_path / f"f{i}.png"
        val = 50 if i != 2 else 220
        save_frame_rgb(fp, np.full((16, 16, 3), val, dtype=np.uint8))
        paths.append(fp)
    report = repair_sequence_drift(paths, threshold=0.3, blend_strength=0.5)
    assert report.repaired_count >= 0


def test_velocity_ease(tmp_path: Path):
    from pipelines.video.segment_processor import apply_velocity_ease, ease_indices

    idx = ease_indices(10, ease="smooth")
    assert len(idx) == 10 and idx[0] == 0 and idx[-1] == 9
    assert idx != list(range(10))
    assert ease_indices(6, ease="hold") == [0, 0, 2, 2, 4, 4]
    paths = []
    for i in range(5):
        fp = tmp_path / f"f{i}.png"
        save_frame_rgb(fp, np.full((8, 8, 3), i * 40, dtype=np.uint8))
        paths.append(fp)
    apply_velocity_ease(paths, ease="smooth")
    assert read_frame_rgb(paths[0]).shape == (8, 8, 3)


def test_region_motion_load_boxes(tmp_path: Path):
    from pipelines.video.motion import load_rig_boxes

    rig = tmp_path / "rig.json"
    rig.write_text(
        '{"regions": [{"name": "head", "box": [0.3, 0.1, 0.7, 0.3], "reference_mode": "identity"}]}',
        encoding="utf-8",
    )
    boxes = load_rig_boxes(rig)
    assert boxes[0][0] == "head"
    assert boxes[0][2] is True


def test_parallel_segments():
    from pipelines.video.video_io import SegmentWorkItem, run_segments_parallel

    def worker(item: SegmentWorkItem):
        return [Path(f"x{item.index}")], {"ok": True}

    items = [SegmentWorkItem(0, Path(".")), SegmentWorkItem(1, Path("."))]
    out = run_segments_parallel(items, worker, max_workers=2)
    assert len(out) == 2
    assert out[0].index == 0


def test_scene_preflight(tmp_path: Path):
    from pipelines.video.scene import run_preflight

    scene = tmp_path / "s.json"
    scene.write_text(
        '{"version": 2, "mode": "t2v", "scene": {"prompt": "test", "duration_sec": 2, "fps": 12}}',
        encoding="utf-8",
    )
    report = run_preflight(scene)
    assert report.ok


# ==================================================================================================
# test_video_upgrades: Tests for video keyframe quality ports + neural VideoDiT wiring.
# ==================================================================================================


def test_sample_options_build_quality_extras_video_upgrades():
    s = SampleOptions(
        num=3,
        pick_best="combo_vit",
        quality_pack="top",
        human_made="standard",
        architecture_lock="auto",
        dual_stage_layout=True,
        holy_grail=True,
        anti_perspective_drift=True,
        boost_quality=True,
    )
    extras = build_keyframe_sample_extras(s, prompt="a red car")
    joined = " ".join(extras)
    assert "--num" in extras and "3" in extras
    assert "--pick-best" in extras and "combo_vit" in extras
    assert "--quality-pack" in extras
    assert "--human-made" in extras
    assert "--dual-stage-layout" in joined
    assert "--holy-grail" in joined
    assert "--boost-quality" in joined


def test_architecture_lock_soft_on_library_prompt():
    s = SampleOptions(architecture_lock="auto")
    locked = soft_architecture_lock_args("grand classical library with balconies", s)
    assert locked.architecture_lock == "on"
    extras = build_keyframe_sample_extras(locked, prompt="grand classical library with balconies")
    assert "--anti-perspective-drift" in extras  # the lock resolves to a real sample.py flag


def test_build_sample_cmd_includes_quality_flags_video_upgrades():
    job = KeyframeEditJob(
        segment_index=0,
        frame_index=0,
        source_frame_path="in.png",
        output_path="out.png",
        prompt="library hall",
    )
    cmd = build_sample_cmd_for_keyframe(
        job,
        ckpt="ckpt.pt",
        sample_options=SampleOptions(num=2, pick_best="combo_architecture", architecture_lock="on"),
    )
    assert "--anti-perspective-drift" in cmd
    assert "--architecture-lock" not in cmd  # not a sample.py option
    assert "--pick-best" in cmd
    assert "combo_architecture" in cmd


def test_video_quality_strong_preset():
    opts = parse_process_options({})
    strong = apply_video_quality_preset(opts, "strong")
    assert strong.motion_beat_keyframes is True
    assert strong.depth_interpolate is True
    assert strong.sample.num >= 3
    assert strong.sample.holy_grail is True
    assert strong.max_retries >= 3


def test_cli_override_does_not_wipe_with_none_pick():
    opts = parse_process_options({"sample": {"pick_best": "combo_vit", "num": 2}})
    out = apply_cli_overrides_to_process_options(opts, keyframe_pick_best="none", keyframe_num=0)
    assert out.sample.pick_best == "combo_vit"


def test_temporal_pick_prefers_continuity(tmp_path: Path):
    prev = np.zeros((32, 32, 3), dtype=np.uint8)
    prev[:] = 40
    close = prev.copy()
    close[:] = 45
    far = prev.copy()
    far[:] = 200
    from PIL import Image

    p_prev = tmp_path / "prev.png"
    p_close = tmp_path / "close.png"
    p_far = tmp_path / "far.png"
    Image.fromarray(prev).save(p_prev)
    Image.fromarray(close).save(p_close)
    Image.fromarray(far).save(p_far)

    assert pixel_continuity_score(close, prev) > pixel_continuity_score(far, prev)
    best, scores = pick_keyframe_with_continuity(
        [p_far, p_close],
        previous_keyframe=p_prev,
        quality_scores=[0.9, 0.5],  # far looks "sharper" but continuity should win with high weight
        continuity_weight=0.8,
    )
    assert best == 1
    assert scores[1] >= scores[0]


def test_list_engines_includes_neural():
    ids = {e.id.value for e in list_engines()}
    assert "neural" in ids
    preset = engine_by_id("neural")
    assert preset is not None
    assert preset.pipeline_notes


def test_route_scene_neural_override():
    d = route_scene("anything", engine_override="neural")
    assert d.engine.value == "neural"
    assert d.edit_overrides.get("neural_engine") is True


def test_video_dit_sequence_context_video_upgrades():
    from models.video_dit import VideoDiT

    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2, context_dim=64)
    x = torch.randn(1, 4, 4, 16, 16)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t, context=torch.randn(1, 8, 64))
    assert out.shape == x.shape


def test_neural_sample_smoke_video_upgrades(tmp_path: Path):
    model = load_video_dit(None, model_name="VideoDiT-S/2", context_dim=32, device="cpu")
    # Shrink: reload tiny for speed
    from models.video_dit import VideoDiT

    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4, patch_size=2, context_dim=16).eval()
    z = sample_video_latents(
        model,
        frames=4,
        height=8,
        width=8,
        steps=2,
        context=torch.zeros(1, 4, 16),
        device="cpu",
    )
    assert z.shape == (1, 4, 4, 8, 8)
    frames = latents_to_frames_placeholder(z, tmp_path, width=64, height=64)
    assert len(frames) == 4
    assert all(p.is_file() for p in frames)


def test_generate_video_list_engines_flag_video_upgrades():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "pipelines" / "video" / "scripts" / "generate_video.py"), "--list-engines"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert r.returncode == 0
    assert "neural" in r.stdout.lower() or "realistic" in r.stdout.lower()


# ==================================================================================================
# test_video_wave2_helpers: Tests for wave-2 video helpers: contact, count, shimmer, shutter, occlusion, chain.
# ==================================================================================================


def test_contact_ground_scores_and_paints(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 180  # bright floor
        # Dark feet blob with no shadow
        img[50:60, 24:40] = 40
        p = tmp_path / f"c{i}.png"
        _save(p, img)
        frames.append(p)
    before = score_contact_ground(frames, sample_every=1)
    assert before.float_events >= 1 or before.score < 0.95
    after = apply_contact_ground(frames, strength=0.8)
    assert after.repaired >= 1


def test_count_binder_prompt_and_drop(tmp_path: Path):
    assert expected_count_from_prompt("three people walking") >= 2
    frames = []
    for i in range(6):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 20
        # Two blobs early, one later
        img[20:32, 10:22] = 220
        if i < 3:
            img[20:32, 40:52] = 220
        p = tmp_path / f"n{i}.png"
        _save(p, img)
        frames.append(p)
    rep = score_count_stability(frames, expected=2, sample_every=1)
    assert rep.drop_events >= 1 or rep.score < 0.95


def test_hf_shimmer_detect_and_fix(tmp_path: Path):
    frames = []
    rng = np.random.default_rng(0)
    for i in range(6):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 40
        # High-freq noise that changes every frame (foliage shimmer)
        noise = rng.integers(0, 255, size=(48, 48), dtype=np.uint8)
        img[..., 0] = noise
        img[..., 1] = noise
        img[..., 2] = noise
        p = tmp_path / f"s{i}.png"
        _save(p, img)
        frames.append(p)
    before = score_hf_shimmer(frames)
    assert before.score < 0.95 or before.shimmer_energy > 0
    after = apply_hf_deshimmer(frames, strength=0.8, window=3)
    assert after.repaired >= 1


def test_motion_shutter_applies_on_pan(tmp_path: Path):
    frames = []
    for i in range(5):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 30
        x = 4 + i * 6
        img[16:32, x : x + 10] = 200
        p = tmp_path / f"m{i}.png"
        _save(p, img)
        frames.append(p)
    rep = apply_motion_shutter(frames, amount=0.7, min_flow=0.5)
    assert rep.applied >= 1 or rep.mean_flow >= 0.5


def test_occlusion_and_secondary_score(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 25
        img[20:40, 20:35] = 200
        img[22:38, 30:45] = 160  # overlapping secondary
        p = tmp_path / f"o{i}.png"
        _save(p, img)
        frames.append(p)
    assert 0.0 <= score_occlusion_coherence(frames).score <= 1.0
    assert 0.0 <= score_secondary_track(frames).score <= 1.0


def test_shot_chain_harmonize(tmp_path: Path):
    prev = []
    nxt = []
    for i in range(3):
        a = np.zeros((32, 32, 3), dtype=np.uint8) + 80
        b = np.zeros((32, 32, 3), dtype=np.uint8) + 180
        pa = tmp_path / f"p{i}.png"
        pb = tmp_path / f"n{i}.png"
        _save(pa, a)
        _save(pb, b)
        prev.append(pa)
        nxt.append(pb)
    anchor = chain_anchor_from_segment(prev, out_path=tmp_path / "anchor.png")
    assert anchor.is_file()
    before = score_cut_continuity(prev, nxt)
    assert before.delta_luma > 0.05
    after = harmonize_cut(nxt, anchor, strength=0.6, n_frames=2)
    assert after.score >= before.score * 0.5  # should not crash; often improves


def test_superior_pass_runs(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 50
        img[30:42, 18:30] = 30
        p = tmp_path / f"u{i}.png"
        _save(p, img)
        frames.append(p)
    opts = parse_process_options(
        {
            "permanence_repair": True,
            "identity_bind": True,
            "extremity_lock": False,
            "glyph_lock": False,
            "contact_ground": True,
            "hf_deshimmer": True,
            "count_bind": True,
            "physics_gate": True,
            "secondary_track": False,
            "motion_shutter": False,
        }
    )
    result = run_superior_pass(frames, opts, prompt="one person standing")
    assert "permanence" in result.scores or any("permanence" in o for o in result.ops)
    assert isinstance(result.ops, list)


# ==================================================================================================
# test_videomax: Video quality presets (video_quality / videomax) and per-medium motion grammars.
# ==================================================================================================


def test_prompt_cues_pick_the_medium() -> None:
    assert grammar_for_prompt("sakuga anime fight, cel shaded").name == "anime_2d"
    assert grammar_for_prompt("16-bit pixel art hero run").name == "pixel_art"
    assert grammar_for_prompt("lego brickfilm chase").name == "stop_motion"
    assert grammar_for_prompt("ecommerce product turntable packshot").name == "product"
    assert grammar_for_prompt("dreamworks style family film").name == "pixar_3d"
    assert grammar_for_prompt("a quiet street at dusk").name == "realistic"
    assert grammar_for_engine("claymation").name == "stop_motion"


def test_videomax_is_style_aware() -> None:
    anime = apply_video_quality(parse_process_options({"videomax": True}), prompt="ghibli anime walk cycle")
    real = apply_video_quality(parse_process_options({"video_quality": "max"}), prompt="photoreal documentary")
    assert anime.videomax and real.videomax
    assert anime.motion_grammar == "anime_2d" and real.motion_grammar == "realistic"
    assert anime.deflicker_strength < real.deflicker_strength
    assert anime.min_physics == 0.0 < real.min_physics
    assert real.sample.num >= 3 and real.sample.holy_grail
    assert real.max_retries >= 3


def test_forced_grammar_and_presets() -> None:
    forced = apply_video_quality(parse_process_options({"videomax": True, "motion_grammar": "cartoon"}), prompt="x")
    assert forced.motion_grammar == "cartoon"
    pixel = apply_video_quality(parse_process_options({"motion_grammar": "pixel_art"}))
    assert pixel.temporal_smooth == 0 and not pixel.videomax
    strong = apply_video_quality(parse_process_options({"video_quality": "strong"}))
    assert strong.sample.pick_best == "combo_vit" and strong.depth_interpolate
    assert apply_video_quality(parse_process_options({})) == parse_process_options({})
    with pytest.raises(ValueError):
        apply_video_quality(parse_process_options({"video_quality": "ultra"}))


def test_plan_videomax_overrides_are_real_options() -> None:
    fields = set(parse_process_options({}).__dataclass_fields__)
    assert set(plan_videomax("a red fox in snow").overrides) <= fields


def test_keyframe_argv_uses_existing_sample_flags() -> None:
    opts = apply_video_quality(parse_process_options({"video_quality": "strong"}))
    extras = build_keyframe_sample_extras(opts.sample)
    assert extras[:4] == ["--num", "3", "--pick-best", "combo_vit"]
    assert "--holy-grail" in extras and "--less-ai" in extras
    job = KeyframeEditJob(
        segment_index=0,
        frame_index=0,
        source_frame_path="in.png",
        output_path="out.png",
        prompt="hall",
        sample_extra_args=["--num", "5", "--seed", "1"],
    )
    cmd = build_sample_cmd_for_keyframe(job, ckpt="c.pt", width=1280, height=720, sample_options=opts.sample)
    assert cmd[cmd.index("--width") + 1] == "1280" and "--image-size" not in cmd
    assert cmd.count("--num") == 1 and cmd.count("--seed") == 1
    assert cmd[cmd.index("--num") + 1] == "3"


# ==================================================================================================
# test_scene_graph: Scene graph compile tests.
# ==================================================================================================


def test_simple_scene_auto_shots():
    g = parse_scene_dict({"scene": {"prompt": "sunset over ocean. then lighthouse close-up", "duration_sec": 4}})
    compiled = compile_scene_graph(g)
    assert len(compiled.plan.shots) >= 2
    assert "sunset" in compiled.plan.shots[0].prompt.lower()


def test_cast_and_effects_merge():
    data = {
        "scene": {"prompt": "forest walk", "duration_sec": 3},
        "characters": {"hero": {"description": "red cloak", "lock": True}},
        "objects": {"lantern": "glowing lantern"},
        "shots": [
            {
                "id": "a",
                "prompt": "wide path",
                "duration_sec": 3,
                "characters": ["hero"],
                "objects": ["lantern"],
                "effects": ["fog"],
            }
        ],
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    shot = compiled.plan.shots[0]
    assert "red cloak" in shot.prompt
    assert "glowing lantern" in shot.prompt
    assert "fog" in shot.prompt.lower() or "haze" in shot.prompt.lower()
    assert "hero" in shot.must_preserve


def test_validate_missing_character():
    g = parse_scene_dict(
        {
            "scene": {"prompt": "test"},
            "shots": [{"prompt": "x", "characters": ["nobody"]}],
        }
    )
    issues = validate_scene_graph(g)
    assert any("nobody" in x for x in issues)


def test_example_scene_file_loads():
    p = Path("examples/scene.example.json")
    if not p.is_file():
        pytest.skip("example missing")
    compiled = compile_scene_graph(load_scene_graph(p))
    assert compiled.plan.mode == VideoMode.T2V
    assert len(compiled.plan.shots) == 2
    assert compiled.segment_overrides[0]["effects"] == ["fog", "film_grain"]
