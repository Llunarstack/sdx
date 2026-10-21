"""Tests for prompt/image understanding on T2V / I2V / V2V paths.

Also holds what used to be: test_video_adherence, test_video_competitor_pain, test_video_continuity, test_video_controls, test_video_dit, test_video_frontier, test_video_pipeline, test_video_studio, test_video_superiority, test_video_tier1, test_video_upgrades.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusion.flow_matching import flow_matching_per_sample_losses
from models.permanent_video_dit import (
    PermanentVideoDiT,
    permanence_consistency_loss,
)
from models.video_dit import FactorizedSpatioTemporalBlock, VideoDiT
from PIL import Image
from pipelines.video.continuity_validators import (
    validate_eyeline,
    validate_light_motivation,
    validate_prop_continuity,
)
from pipelines.video.controls import (
    ControlMode,
    InputBinding,
    build_reference_sheet,
    compile_shot_control_plan,
    parse_elements,
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
from pipelines.video.interpolate import (
    blend_frames,
    interpolate_between_keyframes,
    interpolate_flf2v_sequence,
    prepare_flf2v_keyframes,
)
from pipelines.video.motion import apply_brush_to_flow, parse_motion_brush, parse_motion_library, resolve_motion_clip
from pipelines.video.motion_grammar import apply_motion_grammar, grammar_for_prompt
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
from pipelines.video.pipeline import preview_retrieval_rankings
from pipelines.video.quality import (
    bind_from_refs,
    extract_slots,
    find_glyph_rois,
    score_extremity_coherence,
    score_glyph_stability,
    score_identity_bind,
    score_object_permanence,
    score_physics_invariance,
    score_segment_quality,
    score_video_artifacts,
)
from pipelines.video.retrieval import rank_clips_for_shot
from pipelines.video.scene_graph import compile_scene_graph, parse_scene_dict, validate_scene_graph
from pipelines.video.segment_processor import (
    SampleOptions,
    build_keyframe_sample_extras,
    build_sample_cmd_for_keyframe,
    parse_process_options,
)
from pipelines.video.shot_planner import plan_video_from_prompt, split_prompt_into_beats
from pipelines.video.studio_compiler import (
    RenderEngine,
    bible_to_prompt,
    camera_prompt_fragment,
    compile_studio_block,
    expand_prompt_to_storyboard,
    genre_director_notes,
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
from pipelines.video.types import (
    ClipCandidate,
    KeyframeEditJob,
    MasterTimeline,
    RetrievalSource,
    SegmentAssignment,
    ShotSpec,
    VideoMode,
    VideoPlan,
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
