"""Tests for utils.quality and utils.visual_design: scorers, realism, art/game-art, consistency locks.
Also holds what used to be: test_ai, test_architecture_consistency, test_art, test_camera_isp, test_data_quality_pipeline, test_game_art, test_glyph_and_contact, test_human_likeness, test_human_made, test_scene_physics, test_test_time_pick, test_variation_guard, test_realism_and_hard_prompts, test_visual_design.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from config.prompts import detect_shortcoming_ids, merge_csv_unique, mitigation_fragments, spec_by_id
from config.styles import detect_medium_ids, guidance_fragments
from data.caption_utils import apply_training_guidance_to_caption_pair
from scipy import ndimage
from utils.generation.editing import plan_ideogram_layout
from utils.native import native_stack_summary, quality_highlight_frac
from utils.prompt.art_direction import apply_art_direction_to_args, gate_tag, plan_art_direction
from utils.prompt.game import apply_game_prompt, detect_game_profile, resolve_game_profile
from utils.prompt.realism import (
    apply_hard_prompts_to_args,
    apply_realism_to_args,
    apply_shot_spec_override,
    extract_expected_texts,
    plan_hard_prompt,
    plan_shot_spec,
)
from utils.quality import critics, test_time_pick
from utils.quality.architecture_consistency import (
    apply_architecture_consistency,
    build_story_box_layout,
    infer_architecture_profile,
    prompt_asks_novel_view,
    prompt_needs_architecture_lock,
    score_architecture_consistency,
    score_story_depth_order,
    score_vanishing_symmetry,
    synthetic_story_depth_map,
    write_synthetic_depth_control,
)
from utils.quality.critics import (
    FilterConfig,
    audit_variation,
    filter_jsonl_file,
    reseed_indices,
    score_human_likeness,
    score_not_teal_orange,
    score_occlusion,
    score_reflection,
    score_specular_life,
)
from utils.quality.finishing import (
    CAMERA_PRESETS,
    NO_LENS,
    CameraProfile,
    FilmStock,
    HumanMadeConfig,
    Lens,
    apply_finish,
    apply_human_made_pipeline,
    apply_human_made_prompt_flags,
    demosaic_malvar,
    human_made_preset,
    human_made_prompt_fragments,
    lateral_chromatic_aberration,
    linear_to_srgb,
    mosaic_rggb,
    plan_finish,
    poisson_gaussian_noise,
    reduce_edge_halos,
    save_rgb,
    simulate_capture,
    simulate_film,
    srgb_to_linear,
    to_uint8,
    vignette_gain,
)
from utils.quality.pixel_art import estimate_pixel_size, grid_score, icon_score, make_tileable, pixel_snap, seam_score
from utils.quality.realism import (
    highlight_rolloff,
    lens_for_focal,
    score_plastic_skin,
    score_waxy_highlights,
    tame_saturation,
    tune_finish,
)
from utils.quality.test_time_pick import _maybe_human_scores, _maybe_spatial_bind_scores
from utils.visual_design import (
    DOMAIN_NEGATIVES,
    DOMAIN_POSITIVES,
    VISUAL_DESIGN_PRESETS,
    apply_visual_design_preset_to_prompt,
    apply_visual_design_stage,
    build_visual_design_prompt_pair,
    design_pack_ids,
    extend_sample_argv_visual_design,
    merge_visual_fragments,
    prompt_suggests_domain,
    visual_design_cli_domain_choices,
)

# ==================================================================================================
# test_ai: Tests for utils.native (NumPy fallbacks always; native optional).
# ==================================================================================================


# ==================================================================================================
# test_ai_accel: Tests for utils.native (NumPy fallbacks always; native optional).
# ==================================================================================================


def test_quality_highlight_frac_matches_numpy():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:2] = 255
    assert abs(quality_highlight_frac(rgb, thr=250.0) - 0.25) < 1e-6


def test_native_backends_built_when_present():
    """If the Rust DLL exists from build_native, the wrapper must report available=True."""
    from sdx_native import ffi
    from sdx_native.native_tools import shared_library_path

    if shared_library_path("sdx_rs"):
        assert ffi.rust_available()
    assert "paths" in native_stack_summary()


# ==================================================================================================
# test_ai_image_shortcomings: Tests for config.prompts (prompt mitigation packs).
# ==================================================================================================


def test_merge_csv_unique_dedupes():
    assert merge_csv_unique("a, b", "b, c") == "a, b, c"


def test_detect_portrait_skin():
    ids = detect_shortcoming_ids("portrait, detailed face, natural skin", include_2d=False)
    assert "skin_detail_tangents" in ids


def test_detect_digital_painting_keywords():
    ids = detect_shortcoming_ids("warrior, digital painting, procreate, detailed", include_2d=False)
    assert "digital_painting" in ids


def test_detect_pixel_art():
    ids = detect_shortcoming_ids("16-bit pixel art sprite, rpg enemy", include_2d=False)
    assert "pixel_digital" in ids


def test_mitigation_auto_digital_art():
    pos, neg = mitigation_fragments(
        "concept art environment, matte painting, artstation", "auto", include_2d_pack=False
    )
    assert "design" in pos.lower() or "perspective" in pos.lower()
    assert "cutout" in neg.lower() or "mismatch" in neg.lower()


def test_detect_anime_requires_2d_flag():
    ids_no = detect_shortcoming_ids("1girl, anime style, cel shaded", include_2d=False)
    assert "style_drift_2d" not in ids_no
    ids_yes = detect_shortcoming_ids("1girl, anime style, cel shaded", include_2d=True)
    assert "style_drift_2d" in ids_yes


def test_mitigation_auto_non_empty_for_matched():
    pos, neg = mitigation_fragments("woman sitting on a chair in sunlight", "auto", include_2d_pack=False)
    assert pos
    assert neg
    assert "floating" in neg.lower() or "contradictory" in neg.lower()


def test_mitigation_all_includes_many_hints():
    pos, neg = mitigation_fragments("", "all", include_2d_pack=False)
    assert len(pos) > 80
    assert len(neg) > 80


def test_spec_by_id():
    assert spec_by_id("lighting_gi").id == "lighting_gi"


def test_training_guidance_shortcomings_none_passthrough():
    c, n = apply_training_guidance_to_caption_pair("hello", "bad", shortcomings_mode="none")
    assert c == "hello"
    assert n == "bad"


def test_training_guidance_shortcomings_auto_merges_negative():
    c, n = apply_training_guidance_to_caption_pair(
        "portrait, woman sitting in sunlight", "blurry", shortcomings_mode="auto"
    )
    assert "blurry" in n
    assert "floating" in n.lower() or "contradictory" in n.lower()
    assert len(c) > len("portrait, woman sitting in sunlight")


# ==================================================================================================
# test_ai_look: AI-look critic on synthetic images: each detector separates a tell from its natural counterpart.
# ==================================================================================================


def _pink(n: int = 256, seed: int = 0) -> np.ndarray:
    """1/f^2 power-spectrum field (natural-image statistics), float, mean 128 std 40."""
    rng = np.random.default_rng(seed)
    f = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :])
    f[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / f
    spec[0, 0] = 0.0
    x = np.real(np.fft.ifft2(spec))
    return (x - x.mean()) / x.std() * 40.0 + 128.0


def _rgb(gray: np.ndarray) -> np.ndarray:
    u = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([u, u, u], axis=-1)


def test_decoder_grid_flags_nearest_upsampling_not_one_over_f() -> None:
    small = np.random.default_rng(1).normal(128, 40, (32, 32))
    upsampled = _rgb(np.kron(small, np.ones((8, 8))))
    assert critics.score_decoder_grid(upsampled) == 0.0
    for seed in range(3):
        assert critics.score_decoder_grid(_rgb(_pink(seed=seed))) == 1.0


def test_decoder_grid_flags_a_faint_fingerprint() -> None:
    pattern = np.random.default_rng(2).normal(0, 3.0, (16, 16))
    stamped = _pink(seed=4) + np.tile(pattern, (16, 16))
    assert critics.score_decoder_grid(_rgb(stamped)) < 0.2


def test_oversaturated_image_scores_worse() -> None:
    rgb = np.stack([np.clip(_pink(seed=s) * 0.6 + m, 0, 255) for s, m in ((5, 60), (6, 50), (7, 40))], -1)
    natural = rgb.astype(np.uint8)
    hsv = cv2.cvtColor(natural, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 3.0, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    assert critics.score_oversaturation(natural) > critics.score_oversaturation(boosted)
    assert critics.ai_look_score(boosted).ai_look > critics.ai_look_score(natural).ai_look
    # A requested vivid look relaxes the ramp.
    assert critics.score_oversaturation(boosted, vivid_ok=True) >= critics.score_oversaturation(boosted)


def test_noise_floor_prefers_signal_dependent_capture_noise() -> None:
    ramp = np.tile(np.linspace(20, 235, 256, dtype=np.float32), (256, 1))
    rng = np.random.default_rng(3)
    lin = srgb_to_linear(ramp / 255.0)
    captured = linear_to_srgb(np.clip(lin + rng.normal(size=lin.shape) * np.sqrt(0.004 * lin + 4e-6), 0, 1)) * 255
    grain = ramp + rng.normal(0, 2.0, ramp.shape)
    clean = critics.score_noise_floor(_rgb(ramp))
    flat = critics.score_noise_floor(_rgb(grain))
    real = critics.score_noise_floor(_rgb(captured))
    assert clean == 0.0
    assert real is not None and flat is not None and real > flat > clean


def test_blur_reads_as_oversmoothed_and_steep() -> None:
    sharp, waxy = _rgb(_pink(seed=8)), _rgb(cv2.GaussianBlur(_pink(seed=8), (0, 0), 1.5))
    assert critics.score_oversmoothing(sharp) > critics.score_oversmoothing(waxy)
    assert critics.score_spectral_slope(sharp) > critics.score_spectral_slope(waxy)


def test_halo_detects_unsharp_mask_overshoot() -> None:
    scene = np.full((192, 192), 70.0, np.float32)
    cv2.circle(scene, (96, 96), 50, 190, -1)
    scene = cv2.GaussianBlur(scene, (0, 0), 1.2)
    sharpened = scene + 1.5 * (scene - cv2.GaussianBlur(scene, (0, 0), 2.0))
    assert critics.score_halo(_rgb(scene)) == 1.0
    assert critics.score_halo(_rgb(sharpened)) < 0.5
    assert critics.score_halo(_rgb(_pink())) is None  # texture is not a step edge


def test_mirror_symmetry_penalised_unless_requested() -> None:
    img = np.stack([np.clip(_pink(128, seed=s), 0, 255) for s in (9, 10, 11)], -1).astype(np.uint8)
    mirrored = img.copy()
    mirrored[:, 64:] = img[:, :64][:, ::-1]
    assert critics.score_symmetry(img) > critics.score_symmetry(mirrored)
    assert "symmetry" not in critics.ai_look_score(mirrored, "a symmetrical mandala").parts


def test_drawn_media_skip_capture_detectors() -> None:
    parts = critics.ai_look_score(_rgb(_pink()), "anime girl, cel shading").parts
    assert not set(parts) & {"noise_floor", "spectral_slope", "oversmoothing"}
    assert "grid" not in critics.ai_look_score(_rgb(_pink()), "pixel art castle").parts


# ==================================================================================================
# test_architecture_consistency: Tests for architecture / perspective consistency lock.
# ==================================================================================================


def test_detects_library_and_novel_view():
    assert prompt_needs_architecture_lock("grand classical library with balconies")
    assert prompt_asks_novel_view("show me the second floor of this library")
    assert not prompt_needs_architecture_lock("a red apple on a table")


def test_infer_profile_stories_and_focus():
    p = infer_architecture_profile("two-story library hall, balcony, vanishing point")
    assert p.stories >= 2
    assert p.has_balcony
    assert p.one_point
    nv = infer_architecture_profile("from the second floor balcony looking down the library aisle")
    assert nv.novel_view or nv.focus_story >= 1


def test_story_box_layout_has_recessed_upper_shelves():
    layout = build_story_box_layout("grand library with balcony and upper bookshelves")
    names = {r["name"] for r in layout["regions"]}
    assert "left_ground_shelves" in names
    assert "left_upper_shelves" in names
    assert "left_balcony_rail" in names
    ground = next(r for r in layout["regions"] if r["name"] == "left_ground_shelves")
    upper = next(r for r in layout["regions"] if r["name"] == "left_upper_shelves")
    # Upper face is inset (farther) → right edge of left-upper is left of ground's right edge.
    assert upper["box"][2] < ground["box"][2]


def test_synthetic_depth_upper_farther_than_rail():
    depth = synthetic_story_depth_map(128, 128, stories=2)
    # near=white
    order = score_story_depth_order(depth, stories=2)
    assert order > 0.55


def test_score_symmetry_prefers_mirror():
    h = w = 64
    good = np.zeros((h, w, 3), dtype=np.uint8)
    good[:, : w // 2] = 40
    good[:, w // 2 :] = 40
    # Strong vertical edges mirrored
    good[:, w // 4] = 200
    good[:, -w // 4] = 200
    bad = np.zeros((h, w, 3), dtype=np.uint8)
    bad[:, :10] = 255
    assert score_vanishing_symmetry(good) >= score_vanishing_symmetry(bad) - 1e-6
    s = score_architecture_consistency(good, "library hall vanishing point")
    assert 0.0 <= s <= 1.0


def test_apply_architecture_consistency_soft_wires(tmp_path):
    args = SimpleNamespace(
        prompt="grand classical library hall with second floor balcony and bookshelves",
        negative_prompt="",
        architecture_lock="on",
        architecture_depth_control=True,
        architecture_depth_strength=0.65,
        anti_perspective_drift=False,
        scene_domain="none",
        artist_composition="none",
        dual_stage_layout=False,
        box_attn_layout=False,
        box_layout="",
        control_image="",
        control=[],
        control_type="auto",
        image_size=256,
        num=3,
        pick_best="none",
        _box_layout_spec=None,
    )
    meta = apply_architecture_consistency(args)
    assert meta["enabled"] is True
    assert args.anti_perspective_drift is True
    assert args.dual_stage_layout is True
    assert args.box_attn_layout is True
    assert args._box_layout_spec is not None
    assert args.control and "depth" in args.control[0]
    assert args.pick_best == "combo_architecture"
    assert "story_box_layout" in meta["applied"]
    assert "synthetic_depth_control" in meta["applied"]


def test_write_depth_png(tmp_path):
    path = write_synthetic_depth_control(tmp_path / "d.png", height=64, width=64, stories=2)
    assert path.is_file()
    from PIL import Image

    arr = np.asarray(Image.open(path))
    assert arr.shape == (64, 64)


# ==================================================================================================
# test_art: Art critic on synthetic images, and intent-aware scoring that never penalises a requested look.
# ==================================================================================================


# ==================================================================================================
# test_art_critic: Art critic on synthetic images, and intent-aware scoring that never penalises a requested look.
# ==================================================================================================


def _seascape(tilt_deg: float) -> np.ndarray:
    h, w = 384, 576
    yy, xx = np.mgrid[0:h, 0:w]
    sky = yy < h * 0.55 + np.tan(np.radians(tilt_deg)) * (xx - w / 2)
    img = np.where(sky[..., None], np.array([150, 180, 215.0]), np.array([40, 70, 95.0]))
    img = img + np.random.default_rng(0).normal(0, 3, img.shape)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 0.8), 0, 255).astype(np.uint8)


def _disc(cx: float, cy: float, n: int = 256) -> np.ndarray:
    img = np.full((n, n, 3), 60, np.uint8)
    cv2.circle(img, (int(cx * n), int(cy * n)), n // 10, (220, 200, 180), -1)
    return img


def _hues(hues: list[float], n: int = 128) -> np.ndarray:
    idx = np.random.default_rng(1).integers(0, len(hues), (n, n))
    hsv = np.zeros((n, n, 3), np.uint8)
    hsv[..., 0] = (np.asarray(hues)[idx] / 2) % 180
    hsv[..., 1:] = 200
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_tilted_horizon_detected_and_dutch_angle_allowed() -> None:
    level_score, level_tilt = critics.score_horizon(_seascape(0.0))
    tilt_score, tilt = critics.score_horizon(_seascape(3.0))
    assert level_score == 1.0 and abs(level_tilt) < 0.3
    assert tilt_score == 0.0 and abs(abs(tilt) - 3.0) < 0.3
    assert critics.score_horizon(_seascape(3.0), {"composition": "dutch"}) == (None, None)


def test_composition_follows_the_plan() -> None:
    centred, thirds = _disc(0.5, 0.5), _disc(1 / 3, 1 / 3)
    default = critics.score_composition
    assert default(thirds)["composition"] > default(centred)["composition"]
    plan = {"composition": "center"}
    assert default(centred, plan)["composition"] > default(thirds, plan)["composition"]
    assert default(_disc(0.5, 0.0))["crop"] < default(centred)["crop"]


def test_colour_harmony_prefers_template_fit() -> None:
    complementary, _, dev = critics.score_color_harmony(_hues([30, 210]))
    scattered = critics.score_color_harmony(_hues(list(np.random.default_rng(2).uniform(0, 360, 200))))[0]
    assert complementary == 1.0 and dev < 1.0
    assert scattered is not None and scattered < complementary


def test_low_key_plan_rewards_a_dark_frame() -> None:
    dark = np.clip(np.random.default_rng(3).normal(25, 15, (128, 128, 3)), 0, 255).astype(np.uint8)
    assert critics.score_exposure(dark, {"key": "low"})[0] > critics.score_exposure(dark)[0]


def test_requested_teal_orange_grade_is_not_penalised() -> None:
    plan = critics.resolve_plan("night street portrait, teal and orange grade")
    assert plan["grade"] == "teal_orange"
    assert critics.resolve_plan("a portrait", photo_color_grade="teal_orange")["grade"] == "teal_orange"
    graded = np.zeros((64, 64, 3), np.uint8)
    graded[:32] = (20, 170, 215)
    graded[32:] = (95, 45, 12)
    assert test_time_pick.score_color_cast_neutrality(graded, plan) is None
    assert test_time_pick.score_color_cast_neutrality(graded) < 1.0
    neutral = np.full((64, 64, 3), 128, np.uint8)
    neutral[::2] = 90
    # With the grade requested, the cast term drops out instead of preferring the neutral frame.
    ctx = test_time_pick.PickContext(plan=plan)
    assert test_time_pick.TERMS["cast"]([graded, neutral], ctx) is None


def test_every_policy_pick_is_a_known_metric() -> None:
    from config.presets import PROFILES
    from utils.generation.sample_cli_parser import build_sample_parser

    known = set(test_time_pick.PICK_METRICS)
    assert {p.pick_best for p in PROFILES.values() if p.pick_best} <= known
    choices = next(a.choices for a in build_sample_parser()._actions if "--pick-best" in a.option_strings)
    assert set(choices) == known


# ==================================================================================================
# test_art_direction: Art direction planner: explicit cues kept, open fields decided, caption phrased for the medium.
# ==================================================================================================

_CAMERA = re.compile(r"\b(?:\d+mm|f/\d|lens|bokeh|dslr|photo\w*)\b", re.I)


def test_portrait_photo_gets_portrait_defaults():
    ad = plan_art_direction("portrait photo of an old fisherman", seed=1)
    assert (ad.subject, ad.shot, ad.focal_mm, ad.aperture, ad.dof) == ("portrait", "close_up", 85, 2.0, "shallow")
    assert ad.light_pattern == "loop" and ad.key_fill_ratio == 3.0 and ad.has_people
    assert "85mm lens at f/2" in ad.caption()


def test_painting_caption_has_no_camera_terms_and_keeps_explicit_time():
    ad = plan_art_direction("misty mountain lake at dawn, oil painting", seed=3)
    assert ad.medium == "oil_painting" and ad.medium_family == "traditional" and not ad.photographic
    assert ad.time_of_day == "dawn" and "time_of_day" in ad.explicit
    assert ad.horizon is not None and not ad.hard_light
    caption = ad.caption()
    assert not _CAMERA.search(caption)
    assert "dawn" not in caption  # explicit cue is not echoed
    assert ad.critic_targets()["contrast"] == "low"


def test_explicit_cues_are_parsed_not_echoed():
    ad = plan_art_direction("low-key noir portrait of a detective, split lighting, teal and orange grade", seed=0)
    assert (ad.light_pattern, ad.tonal_key, ad.grade, ad.key_fill_ratio) == ("split", "low", "teal_orange", 8.0)
    assert {"light_pattern", "tonal_key", "grade"} <= ad.explicit
    caption = ad.caption().lower()
    assert "split" not in caption and "teal" not in caption
    targets = ad.critic_targets()
    assert targets["key"] == "low" and targets["grade"] == "teal_orange" and targets["harmony"] == "complementary"


def test_lens_and_aperture_numbers_are_explicit():
    ad = plan_art_direction("street portrait, 35mm film, shot at f/1.4", seed=0)
    assert ad.aperture == 1.4 and "aperture" in ad.explicit
    assert "focal_mm" not in ad.explicit  # "35mm film" is a film format, not a focal length
    assert plan_art_direction("cat on a sofa, 135mm", seed=0).focal_mm == 135


def test_seed_explores_open_choices_deterministically():
    plans = [plan_art_direction("a quiet valley with a river", seed=s) for s in range(12)]
    assert len({(p.time_of_day, p.subject_side) for p in plans}) > 2
    assert plan_art_direction("a quiet valley with a river", seed=5) == plans[5]


def test_medium_specific_plans():
    pixel = plan_art_direction("pixel art knight reflected in a puddle", seed=0)
    assert pixel.medium == "pixel_art" and pixel.critic_targets()["pixel_art"]
    assert "limited palette" in pixel.caption() and not _CAMERA.search(pixel.caption())
    ink = plan_art_direction("ink drawing of an old oak tree", seed=0)
    assert ink.harmony == "mono" and "palette" not in ink.caption()
    product = plan_art_direction("product photo of a watch on a white background", seed=0)
    assert product.subject == "product" and product.tonal_key == "high" and product.placement == "center"


def test_gate_tag_drops_contradicting_pack_text():
    lake = plan_art_direction("misty mountain lake at dawn, oil painting")
    for tag in (
        "natural skin texture",
        "correct hands and fingers",
        "lifelike eyes",
        "shot on real camera",
        "true lens bokeh",
    ):
        assert not gate_tag(tag, lake), tag
    assert gate_tag("intentional brushwork", lake)
    assert gate_tag("photorealistic", lake, negative=True)  # camera words stay in a painting's negative
    fox = plan_art_direction("red fox in fresh snow, wildlife photo")
    assert gate_tag("lifelike eyes", fox) and not gate_tag("natural skin texture", fox)
    render = plan_art_direction("3d render of a robot, octane")
    assert gate_tag("85mm lens", render) and not gate_tag("dslr photo", render)
    product = plan_art_direction("product photo of a watch on a white background")
    assert not gate_tag("dead-center framing", product, negative=True)
    assert not gate_tag("rule of thirds", product)
    vista = plan_art_direction("photoreal mountain landscape at golden hour")
    assert vista.dof == "deep" and not gate_tag("true lens bokeh", vista) and gate_tag("deep focus", vista)


def test_apply_to_args_exposes_plan_and_off_switch():
    args = SimpleNamespace(prompt="ignored", _user_prompt="family having dinner in a garden, golden hour", seed=4)
    ad = apply_art_direction_to_args(args)
    plan = args.art_direction_plan
    assert ad is args._art_direction and plan["subject"] == "group" and plan["time_of_day"] == "golden_hour"
    assert plan["caption"] == ad.caption() and plan["grade"] == "warm" and plan["medium"] == "generic"
    assert len(plan["subject_box"]) == 4 and 0.0 <= plan["subject_box"][0] < plan["subject_box"][2] <= 1.0
    off = SimpleNamespace(prompt="a cat", art_direction="off")
    assert apply_art_direction_to_args(off) is None and off.art_direction_plan is None


# ==================================================================================================
# test_art_mediums: Tests for config.styles guidance packs.
# ==================================================================================================


def test_detect_digital_medium():
    ids = detect_medium_ids("digital painting portrait in photoshop", include_photography=True)
    assert "digital_painting" in ids


def test_detect_photo_toggle():
    ids_on = detect_medium_ids("street photography candid portrait photo", include_photography=True)
    assert any(i.endswith("_photo") or "street_documentary_photo" == i for i in ids_on)
    ids_off = detect_medium_ids("street photography candid portrait photo", include_photography=False)
    assert "street_documentary_photo" not in ids_off


def test_guidance_auto_adds_anatomy_lite_for_people():
    pos, neg = guidance_fragments(
        "woman portrait, digital painting",
        "auto",
        include_photography=True,
        anatomy_mode="lite",
    )
    assert "accurate anatomy" in pos.lower()
    assert "bad anatomy" in neg.lower()


def test_guidance_all_contains_multiple_mediums():
    pos, neg = guidance_fragments("landscape", "all", include_photography=False, anatomy_mode="none")
    assert len(pos) > 120
    assert len(neg) > 120


def test_detect_new_3d_and_photo_mediums():
    ids_3d = detect_medium_ids("archviz octane render interior visualization", include_photography=True)
    assert "archviz_3d" in ids_3d
    ids_photo = detect_medium_ids("wedding event photo ceremony portrait", include_photography=True)
    assert "wedding_event_photo" in ids_photo


def test_detect_medium_ids_with_aliases():
    ids = detect_medium_ids("bw film photo and toon-shaded 3d robot scene", include_photography=True)
    assert "film_bw_photo" in ids
    assert "toon_3d" in ids


def test_detect_new_storyboard_and_diorama_aliases():
    ids = detect_medium_ids(
        "animatic storyboard for action beats with miniature photo lighting", include_photography=True
    )
    assert "storyboard_sketch" in ids
    assert "miniature_diorama_photo" in ids


def test_guidance_adds_color_theory_and_render_terms_for_2d():
    pos, neg = guidance_fragments(
        "digital painting concept art character portrait",
        "auto",
        include_photography=False,
        anatomy_mode="none",
    )
    assert "warm-cool balance" in pos.lower()
    assert "blend-mode discipline" in pos.lower()
    assert "banded gradients" in neg.lower()


def test_guidance_adds_pbr_terms_for_3d():
    pos, neg = guidance_fragments(
        "hard surface archviz 3d render",
        "auto",
        include_photography=False,
        anatomy_mode="none",
    )
    assert "physically based shading logic" in pos.lower()
    assert "roughness/metalness" in pos.lower()
    assert "inconsistent brdf" in neg.lower()


# ==================================================================================================
# test_camera_isp: Capture finish: lens, Poisson-Gaussian sensor, ISP, film grain, medium routing, output quantisation.
# ==================================================================================================


NOISELESS = CameraProfile(
    "noiseless",
    lens=NO_LENS,
    full_well_e=1e12,
    read_noise_e=0.0,
    post_read_noise_e=0.0,
    cfa=False,
    tone=0.0,
    sharpen=0.0,
    jpeg_quality=0,
)


def _ramp(h: int = 64, w: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([40 + 150 * xx / w, 60 + 120 * yy / h, 200 - 120 * xx / w], axis=-1)


def test_sensor_noise_variance_grows_linearly_with_signal() -> None:
    rng = np.random.default_rng(0)
    levels = np.array([0.02, 0.1, 0.3, 0.7])
    var = [
        poisson_gaussian_noise(np.full((256, 256), m), full_well_e=5000, read_noise_e=3, rng=rng).var() for m in levels
    ]
    slope, intercept = np.polyfit(levels, var, 1)
    assert np.all(np.diff(var) > 0)
    assert slope == pytest.approx(1 / 5000, rel=0.1)
    assert intercept == pytest.approx((3 / 5000) ** 2, abs=2e-6)


def test_noise_grows_with_iso() -> None:
    flat = np.full((64, 64, 3), 120.0, np.float32)
    base = replace(CAMERA_PRESETS["dslr"], lens=NO_LENS, sharpen=0.0, chroma_nr=0.0)
    low = simulate_capture(flat, replace(base, iso=100), seed=1, jpeg=False).std()
    high = simulate_capture(flat, replace(base, iso=3200), seed=1, jpeg=False).std()
    assert high > 3 * low > 0


def test_unprocess_reprocess_is_identity_without_effects() -> None:
    img = _ramp()
    out = simulate_capture(img, NOISELESS, jpeg=False)
    assert np.abs(out - img).max() < 0.05


def test_malvar_demosaic_reconstructs_flat_colour() -> None:
    rgb = np.broadcast_to(np.array([0.2, 0.5, 0.8], np.float32), (16, 16, 3))
    assert np.allclose(demosaic_malvar(mosaic_rggb(rgb)), rgb, atol=1e-5)


def _centroid(plane: np.ndarray) -> np.ndarray:
    ys, xs = np.mgrid[0 : plane.shape[0], 0 : plane.shape[1]]
    w = plane / plane.sum()
    return np.array([(w * ys).sum(), (w * xs).sum()])


def test_chromatic_aberration_is_radial_and_fixes_the_centre() -> None:
    h = w = 129
    centre, mid, far = (64, 64), (64 + 20, 64 + 15), (64 + 48, 64 + 36)
    shifts = []
    for y, x in (centre, mid, far):
        img = np.zeros((h, w, 3), np.float32)
        img[y - 1 : y + 2, x - 1 : x + 2, :] = 1.0
        out = lateral_chromatic_aberration(img, 0.02, -0.02)
        assert np.array_equal(out[..., 1], img[..., 1])
        red = _centroid(np.clip(out[..., 0], 0, None)) - (y, x)
        blue = _centroid(np.clip(out[..., 2], 0, None)) - (y, x)
        shifts.append((red, blue, np.array([y - 64, x - 64], np.float64)))
    assert np.abs(shifts[0][0]).max() < 1e-4 and np.abs(shifts[0][1]).max() < 1e-4
    for red, blue, radial in shifts[1:]:
        unit = radial / np.linalg.norm(radial)
        assert red @ unit > 0 > blue @ unit
        assert abs(red[0] * unit[1] - red[1] * unit[0]) < 0.02 * np.linalg.norm(red)
    assert np.linalg.norm(shifts[2][0]) > np.linalg.norm(shifts[1][0])


def test_vignette_is_applied_exactly_once() -> None:
    flat = np.full((65, 97, 3), 150.0, np.float32)
    lens = Lens(vignette=0.6, ca_red=0.0, ca_blue=0.0, psf_sigma=0.0)
    out = simulate_capture(flat, replace(NOISELESS, lens=lens), jpeg=False)
    measured = srgb_to_linear(out / 255.0)[..., 1] / srgb_to_linear(np.float32(150 / 255))
    assert np.allclose(measured, vignette_gain(65, 97, lens), atol=2e-3)
    assert measured[32, 48] == pytest.approx(1.0, abs=1e-3)


def test_finish_has_no_seam_or_wraparound_ghost() -> None:
    img = np.zeros((64, 96, 3), np.float32)
    img[:, :8] = 255.0
    plan = replace(CAMERA_PRESETS["dslr"], iso=100)
    out = apply_finish(img, plan, seed=3).astype(np.float32)
    assert out[:, -8:].mean() < 2.0
    ramp = apply_finish(_ramp(), plan, seed=3).astype(np.float32)
    col_step = np.abs(np.diff(ramp.mean(axis=0), axis=0)).max(axis=-1)
    assert col_step[44:52].max() < 3 * np.median(col_step) + 1.0


def test_seed_determinism() -> None:
    img = _ramp()
    plan = plan_finish("street photo at night")
    a = apply_finish(img, plan, seed=5)
    assert np.array_equal(a, apply_finish(img, plan, seed=5))
    assert not np.array_equal(a, apply_finish(img, plan, seed=6))
    film = plan_finish("portra 400 film photo")
    assert np.array_equal(apply_finish(img, film, seed=2), apply_finish(img, film, seed=2))


def test_medium_routing() -> None:
    for prompt in ("anime girl, cel shading", "watercolor painting of a harbor", "octane 3d render", "pixel art"):
        assert plan_finish(prompt) is None
    assert plan_finish("anything", camera="none") is None
    assert isinstance(plan_finish("portrait photo, 85mm"), CameraProfile)
    assert plan_finish("candid iphone photo").name == "phone"
    assert plan_finish("portrait photo").iso < plan_finish("portrait photo at night").iso
    assert plan_finish("photo, ISO 3200").iso == 3200
    film = plan_finish("black and white film photo, tri-x")
    assert isinstance(film, FilmStock) and film.monochrome
    assert isinstance(plan_finish("a cafe", grain_style="heavy_16mm"), FilmStock)


def test_anime_route_adds_no_noise() -> None:
    flat = np.full((32, 32, 3), 123.4, np.float32)
    out = to_uint8(apply_finish(flat, plan_finish("anime girl, cel shading"), seed=1))
    assert np.array_equal(out, np.full_like(out, 123))


def test_film_grain_is_signal_dependent_without_sensor_noise() -> None:
    stock = FilmStock("t", iso=3200, lens=NO_LENS, frame_width_mm=10.0)
    stds = []
    for level in (30.0, 120.0, 230.0):
        out = simulate_film(np.full((96, 96, 3), level, np.float32), stock, seed=0)
        assert out.mean() == pytest.approx(level, abs=0.5)
        stds.append(float(out.std()))
    assert max(stds) > 1.3 * min(stds) > 0
    mono = simulate_film(_ramp(), replace(stock, monochrome=True), seed=0)
    gain = srgb_to_linear(mono / 255.0) / srgb_to_linear(_ramp() / 255.0)
    assert np.allclose(gain[..., 0], gain[..., 1], atol=2e-3)


def test_dithered_quantisation_and_16bit_png(tmp_path) -> None:
    flat = np.full((64, 64, 3), 100.3, np.float32)
    assert to_uint8(flat).mean() == 100
    assert to_uint8(flat, np.random.default_rng(0)).mean() == pytest.approx(100.3, abs=0.05)
    path = tmp_path / "out.png"
    save_rgb(flat, path, sixteen_bit=True)
    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16 and abs(int(back[0, 0, 0]) - round(100.3 * 257)) <= 1


# ==================================================================================================
# test_data_quality_pipeline: Data quality pipeline tests.
# ==================================================================================================


def test_filter_jsonl_caption_length(tmp_path: Path) -> None:
    inp = tmp_path / "m.jsonl"
    rows = [
        {"image_path": "a.png", "caption": "short"},
        {"image_path": "b.png", "caption": "this caption is long enough to pass"},
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    kept, stats = filter_jsonl_file(
        inp,
        config=FilterConfig(min_caption_len=10, dedup=""),
        output_path=out,
    )
    assert stats.kept == 1
    assert kept[0]["caption"].startswith("this caption")


# ==================================================================================================
# test_game_art: Game imagery: prompt detection, exact pixel-art finishing, tiles, icons and the sample-run hooks.
# ==================================================================================================


PAL = np.array([[20, 20, 30], [200, 40, 40], [240, 200, 60], [40, 120, 200], [250, 250, 250], [90, 60, 40]], np.uint8)


def _render(sprite: np.ndarray, scale: int, phase: int = 0, blur: float = 1.0, noise: float = 6.0, seed: int = 0):
    """Upscale a sprite like a diffusion model would draw it: soft edges, noise, off-grid placement."""
    up = np.repeat(np.repeat(PAL[sprite], scale, 0), scale, 1).astype(np.float32)
    if blur:
        up = ndimage.gaussian_filter(up, (blur, blur, 0))
    up += np.random.default_rng(seed).normal(0, noise, up.shape)
    return np.roll(up, (phase, phase), (0, 1))


def _blob_sprite() -> np.ndarray:
    yy, xx = np.mgrid[:32, :32]
    s = np.zeros((32, 32), int)
    s[(yy - 16) ** 2 + (xx - 16) ** 2 < 100] = 1
    s[(yy - 14) ** 2 + (xx - 12) ** 2 < 9] = 2
    s[20:24, 10:22] = 3
    s[13, 12] = 4  # eye glint: a lone high-contrast pixel that must survive
    return s


@pytest.mark.parametrize("scale,phase,blur", [(4, 1, 0.6), (8, 3, 0.0), (12, 0, 1.0), (16, 5, 1.2), (21, 7, 1.5)])
def test_estimate_pixel_size_finds_true_grid(scale, phase, blur):
    busy = np.random.default_rng(scale).integers(0, 6, (48, 48))
    p, ox, oy, _ = estimate_pixel_size(_render(busy, scale, phase, blur))
    assert (p, ox, oy) == (scale, phase % scale, phase % scale)
    p, ox, _, _ = estimate_pixel_size(_render(_blob_sprite(), scale, phase, blur))
    assert (p, ox) == (scale, phase % scale)


def test_estimate_pixel_size_rejects_smooth_images():
    smooth = ndimage.gaussian_filter(np.random.default_rng(1).uniform(0, 255, (512, 512, 3)), (3, 3, 0))
    assert estimate_pixel_size(smooth)[0] == 1
    assert grid_score(smooth) == 0.0


def test_pixel_snap_recovers_sprite_exactly():
    sprite = _blob_sprite()
    img = _render(sprite, 16, blur=1.2)
    full, small = pixel_snap(img, palette_colors=8)
    assert full.dtype == np.uint8 and full.shape == img.shape
    assert small.shape == (32, 32, 3)
    assert np.abs(small.astype(int) - PAL[sprite].astype(int)).max() <= 16
    assert len(np.unique(small.reshape(-1, 3), axis=0)) == len(np.unique(sprite))  # near-duplicate shades merged
    assert tuple(small[13, 12]) == tuple(np.clip(np.rint(small[13, 12]), 0, 255))
    assert small[13, 12].min() > 230  # glint kept
    # Every output pixel belongs to a flat 16x16 cell.
    cells = full.reshape(32, 16, 32, 16, 3)
    assert (cells == cells[:, :1, :, :1]).all()


def test_pixel_snap_palette_limit_and_transparency():
    sprite = _blob_sprite()
    _, few = pixel_snap(_render(sprite, 16), grid=32, palette_colors=3)
    assert len(np.unique(few.reshape(-1, 3), axis=0)) <= 3
    full, small = pixel_snap(_render(sprite, 16), grid=32, palette_colors=8, transparent_bg=True)
    assert small.shape == (32, 32, 4) and full.shape[-1] == 4
    assert np.array_equal(small[..., 3] == 0, sprite == 0)


def test_tileable_and_critics():
    tex = ndimage.gaussian_filter(np.random.default_rng(2).uniform(0, 255, (256, 256, 3)), (6, 6, 0))
    assert seam_score(make_tileable(tex)) > seam_score(tex) + 0.5
    icon, _ = pixel_snap(_render(_blob_sprite(), 16), grid=32, palette_colors=8)
    assert icon_score(icon) > 0.5
    assert icon_score(np.full((64, 64, 3), 200.0)) == 0.0
    assert grid_score(icon) > 0.9


@pytest.mark.parametrize(
    "prompt,asset,pixel",
    [
        ("16-bit pixel art knight sprite sheet, walk cycle", "sprite_sheet", True),
        ("seamless grass texture for a game", "tileset", False),
        ("fantasy potion item icon", "icon", False),
        ("gameplay screenshot of an open-world game, forest at dusk", "in_engine", False),
        ("isometric pixel art tavern", "isometric", True),
        ("32x32 pixel art slime", "pixel_art", True),
        ("portrait photo of a woman, 85mm", None, False),
    ],
)
def test_detect_game_profile(prompt, asset, pixel):
    prof = detect_game_profile(prompt)
    if asset is None:
        assert prof is None
        return
    assert prof.asset == asset and prof.pixel_art == pixel


def test_profile_details_and_overrides():
    prof = detect_game_profile("8-bit 32x32 pixel art slime")
    assert (prof.grid, prof.palette_colors) == (32, 16)
    pos, neg = apply_game_prompt("slime", "", prof)
    assert "crisp hard-edged pixels" in pos and "anti-aliasing" in neg
    args = SimpleNamespace(prompt="a cat", game_asset="icon", palette_colors=-1, transparent_bg="off", pixel_size=0)
    forced = resolve_game_profile(args)
    assert forced.asset == "icon" and not forced.transparent_bg
    assert resolve_game_profile(SimpleNamespace(prompt="pixel art cat", game_asset="off")) is None


def test_game_finish_and_save(tmp_path):
    from PIL import Image
    from utils.generation.sample_run import game_finish, save_game_image

    args = SimpleNamespace(
        prompt="32x32 pixel art slime sprite, transparent background",
        game_asset="auto",
        palette_colors=-1,
        transparent_bg="auto",
        pixel_size=0,
        seed=0,
    )
    report: dict = {}
    finals, natives = game_finish(args, [_render(_blob_sprite(), 16)], report)
    assert finals[0].shape[-1] == 4 and natives[0].shape == (32, 32, 4)
    assert report["game"]["native_size"] == [32, 32]
    out = tmp_path / "slime.png"
    save_game_image(finals[0], natives[0], out, sixteen_bit=False, seed=0)
    assert Image.open(out).mode == "RGBA"
    assert Image.open(tmp_path / "slime_sprite.png").size == (32, 32)
    # Non-game prompts pass through untouched.
    plain = [np.zeros((8, 8, 3), np.float32)]
    assert game_finish(SimpleNamespace(prompt="a cat", seed=0), plain, {})[0] is plain


# ==================================================================================================
# test_glyph_and_contact
# ==================================================================================================


def test_extract_glyph_strings_quoted_and_bracketed():
    strings = plan_ideogram_layout('poster with [text: SALE] and "50% OFF"').glyph_texts
    assert "SALE" in strings
    assert "50% OFF" in strings


def _synthetic_feet_with_contact(*, contact: bool, size: int = 128) -> np.ndarray:
    rgb = np.full((size, size, 3), 190, dtype=np.uint8)
    y0 = int(size * 0.78)
    foot_y = y0 + int(size * 0.10)
    foot_x0 = size // 3
    foot_x1 = 2 * size // 3
    rgb[foot_y : foot_y + 4, foot_x0:foot_x1] = 40
    if contact:
        # Subtle AO just below feet; stays outside subject mask (|175-190| <= 18).
        rgb[foot_y + 4 : foot_y + 6, foot_x0:foot_x1] = 175
    return rgb


def test_contact_shadow_scores_grounded_higher_than_floating():
    grounded = _synthetic_feet_with_contact(contact=True)
    floating = _synthetic_feet_with_contact(contact=False)
    assert critics.score_contact_shadow(grounded) > critics.score_contact_shadow(floating)


def test_apply_contact_shadow_darkens_lower_band():
    rgb = _synthetic_feet_with_contact(contact=False)
    y0 = int(rgb.shape[0] * 0.78)
    before = float(rgb[y0:, :].mean())
    out = critics.apply_contact_shadow(rgb, strength=0.45)
    after = float(out[y0:, :].mean())
    assert after < before


def test_prompt_wants_ground_contact():
    assert critics.prompt_wants_ground_contact("woman standing on wet pavement") is True
    assert critics.prompt_wants_ground_contact("macro of an eye") is False


# ==================================================================================================
# test_human_likeness: Human-art fingerprint scores.
# ==================================================================================================


def test_mirror_scores_lower_than_lived_in() -> None:
    rng = np.random.default_rng(0)
    left = rng.integers(40, 220, (64, 32, 3), dtype=np.uint8)
    doll = np.concatenate([left, np.fliplr(left)], axis=1)
    life = rng.integers(40, 220, (64, 64, 3), dtype=np.uint8)
    assert score_human_likeness(life) > score_human_likeness(doll)


def test_teal_orange_lut_is_penalized() -> None:
    lut = np.zeros((64, 64, 3), dtype=np.uint8)
    lut[:32] = (20, 170, 215)
    lut[32:] = (95, 45, 12)
    plain = np.full((64, 64, 3), 128, dtype=np.uint8)
    plain[:32, :, 1] = 140
    assert score_not_teal_orange(plain) > score_not_teal_orange(lut)
    # Asked-for teal-orange is a look, not a defect.
    assert score_human_likeness(lut, "teal and orange grade") > score_human_likeness(lut)


def test_specular_clone_is_penalized() -> None:
    doll = np.zeros((64, 64, 3), dtype=np.uint8)
    doll[:, :] = 40
    doll[20:24, 18:22] = 255
    doll[20:24, 42:46] = 255
    one = doll.copy()
    one[20:24, 42:46] = 40
    one[38:44, 10:16] = 255
    assert score_specular_life(one) > score_specular_life(doll)


def test_strong_prompt_fragments_name_human_likeness() -> None:
    pos, neg = human_made_prompt_fragments("strong")
    blob = f"{pos} {neg}".lower()
    assert "catchlights" in blob
    assert "teal" in blob


def test_maybe_spatial_bind_not_shadowed_by_human_scores() -> None:
    rng = np.random.default_rng(3)
    imgs = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(2)]
    human = _maybe_human_scores("a portrait", imgs)
    bind = _maybe_spatial_bind_scores("a portrait", imgs)
    assert human is not None and len(human) == 2
    assert all(0.0 <= v <= 1.0 for v in human)
    # Portrait has no spatial color-bind, so the bind helper stays None.
    assert bind is None


# ==================================================================================================
# test_human_made: Human-made anti-AI polish helpers.
# ==================================================================================================


def test_human_made_preset_order() -> None:
    lite = human_made_preset("lite")
    strong = human_made_preset("strong")
    assert lite.strength < strong.strength


def test_prompt_fragments_non_empty() -> None:
    pos, neg = human_made_prompt_fragments("standard")
    assert "plastic" in neg.lower()
    assert len(pos) > 10


def test_reduce_edge_halos_noop_at_zero() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert np.array_equal(reduce_edge_halos(img, strength=0.0), img)


def test_apply_pipeline_identity_when_disabled() -> None:
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert np.array_equal(apply_human_made_pipeline(img, HumanMadeConfig(strength=0.0)), img)


def test_pipeline_is_deterministic_and_adds_no_noise() -> None:
    flat = np.full((48, 48, 3), 120.0, dtype=np.float32)
    out = apply_human_made_pipeline(flat, human_made_preset("strong"))
    assert np.array_equal(out, apply_human_made_pipeline(flat, human_made_preset("strong")))
    assert float(out.std()) < 1e-3


def test_apply_human_made_prompt_flags() -> None:
    class Args:
        human_made = "standard"
        anti_ai_pack = "none"
        human_media_mode = "none"
        naturalize = False
        shortcomings_mitigation = "none"
        less_ai = False

    a = Args()
    apply_human_made_prompt_flags(a)
    assert a.naturalize is True
    assert a.anti_ai_pack != "none"


# ==================================================================================================
# test_scene_physics
# ==================================================================================================


def _fence_rgb(size: int = 128) -> np.ndarray:
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(0, size, 4):
        rgb[:, x : x + 2] = 255
    return rgb


def _uniform_blob_rgb(size: int = 128) -> np.ndarray:
    rgb = np.full((size, size, 3), 24, dtype=np.uint8)
    rgb[12 : size - 12, 12 : size - 12] = 180
    return rgb


def _darker_flip_reflection(size: int = 128) -> np.ndarray:
    rgb = np.full((size, size, 3), 36, dtype=np.uint8)
    half = size // 2
    # Asymmetric blobs near the top of the upper half so a flip is distinctive.
    rgb[6:28, 14:58] = 230
    rgb[44:58, 80:110] = 210
    rgb[half : half + half] = (rgb[:half][::-1].astype(np.float32) * 0.42).astype(np.uint8)
    return rgb


def _two_bright_stacked_blobs(size: int = 128) -> np.ndarray:
    rgb = np.full((size, size, 3), 36, dtype=np.uint8)
    half = size // 2
    rgb[6:28, 14:58] = 230
    rgb[44:58, 80:110] = 210
    rgb[half + 6 : half + 28, 14:58] = 230
    rgb[half + 44 : half + 58, 80:110] = 210
    return rgb


def test_score_occlusion_fence_beats_blob():
    fence = _fence_rgb()
    blob = _uniform_blob_rgb()
    assert score_occlusion(fence) > score_occlusion(blob)


def test_score_reflection_darker_flip_beats_equal_blobs():
    reflected = _darker_flip_reflection()
    stacked = _two_bright_stacked_blobs()
    assert score_reflection(reflected) > score_reflection(stacked)


def test_score_spatial_color_bind_prefers_correct_halves():
    from utils.quality.critics import score_spatial_color_bind

    prompt = "a red cube to the left of a blue sphere"
    good = np.zeros((64, 64, 3), dtype=np.uint8)
    good[:, :32] = (200, 20, 20)
    good[:, 32:] = (20, 30, 200)
    bad = np.zeros((64, 64, 3), dtype=np.uint8)
    bad[:, :32] = (20, 30, 200)
    bad[:, 32:] = (200, 20, 20)
    assert score_spatial_color_bind(good, prompt) > score_spatial_color_bind(bad, prompt)
    assert score_spatial_color_bind(good, "a cat sitting in a garden") is None


# ==================================================================================================
# test_test_time_pick
# ==================================================================================================


def test_infer_expected_people_count_from_prompt_patterns():
    assert test_time_pick.infer_expected_people_count("exactly 3 people in a cafe scene") == 3
    assert test_time_pick.infer_expected_people_count("two characters, dynamic action pose") == 2
    assert test_time_pick.infer_expected_people_count("portrait of 1girl, studio light") == 1
    assert test_time_pick.infer_expected_people_count("beautiful landscape, no people") == 0


def test_infer_expected_object_count_from_prompt_patterns():
    n, obj = test_time_pick.infer_expected_object_count("exactly 7 coins on a wooden table")
    assert n == 7
    assert obj == "coin"
    n2, obj2 = test_time_pick.infer_expected_object_count("two candles in a dark room")
    assert n2 == 2
    assert obj2 == "candle"


def test_score_people_count_match_uses_error_ratio(monkeypatch):
    monkeypatch.setattr(test_time_pick, "estimate_people_count", lambda _im: 3)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert test_time_pick.score_people_count_match(img, 3) == 1.0
    assert abs(test_time_pick.score_people_count_match(img, 2) - 0.5) < 1e-8


def test_combo_count_prefers_matching_candidate(monkeypatch):
    # Make CLIP/edge neutral so count score dominates in this unit test.
    monkeypatch.setattr(
        test_time_pick, "score_clip_similarity", lambda imgs, prompt, device, model_id: [0.5 for _ in imgs]
    )
    monkeypatch.setattr(test_time_pick, "score_edge_sharpness", lambda im: 1.0)
    monkeypatch.setattr(
        test_time_pick,
        "score_people_count_match",
        lambda im, exp: 1.0 if int(im[0, 0, 0]) == int(exp) else 0.0,
    )

    img_wrong = np.zeros((8, 8, 3), dtype=np.uint8)  # marker 0
    img_right = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right[0, 0, 0] = 2  # marker 2

    best, scores = test_time_pick.pick_best_indices(
        [img_wrong, img_right],
        "exactly 2 people",
        "combo_count",
        "cpu",
    )
    assert best == 1
    assert scores[1] > scores[0]


def test_combo_count_object_mode_prefers_matching_candidate(monkeypatch):
    monkeypatch.setattr(
        test_time_pick, "score_clip_similarity", lambda imgs, prompt, device, model_id: [0.5 for _ in imgs]
    )
    monkeypatch.setattr(test_time_pick, "score_edge_sharpness", lambda im: 1.0)
    monkeypatch.setattr(
        test_time_pick,
        "score_object_count_match",
        lambda im, exp, object_hint="": 1.0 if int(im[0, 0, 0]) == int(exp) else 0.0,
    )

    img_wrong = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right[0, 0, 0] = 5

    best, scores = test_time_pick.pick_best_indices(
        [img_wrong, img_right],
        "five candles on a cake",
        "combo_count",
        "cpu",
        expected_count_target="objects",
        expected_count_object="candle",
    )
    assert best == 1
    assert scores[1] > scores[0]


def test_rank_aggregation_is_invariant_to_metric_scale():
    a = [0.1, 0.5, 0.3]
    b = [3.0, 1.0, 2.0]
    base = test_time_pick.rank_aggregate({"a": a, "b": b}, {"a": 0.7, "b": 0.3})
    scaled = test_time_pick.rank_aggregate(
        {"a": [x * 1e6 for x in a], "b": [np.log(x) for x in b]}, {"a": 0.7, "b": 0.3}
    )
    assert scaled == pytest.approx(base)
    assert int(np.argmax(base)) == 1
    # Float noise ties instead of ordering candidates; a missing term drops out.
    assert test_time_pick.rank01([1.0, 1.0 + 1e-12, 0.5]) == pytest.approx([0.75, 0.75, 0.0])
    assert test_time_pick.rank_aggregate({"a": a, "b": None}, {"a": 1.0, "b": 1.0}) == pytest.approx([0.0, 1.0, 0.5])


def test_combo_realism_prefers_candidate_with_better_realism_metrics(monkeypatch):
    monkeypatch.setattr(
        test_time_pick, "score_clip_similarity", lambda imgs, prompt, device, model_id: [0.5 for _ in imgs]
    )
    monkeypatch.setattr(test_time_pick, "score_photographic_detail_balance", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_exposure_balance", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_dynamic_range_headroom", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_color_cast_neutrality", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_skin_tone_naturalness", lambda im: float(im[0, 0, 0]) / 255.0)

    img_low = np.zeros((8, 8, 3), dtype=np.uint8)
    img_high = np.zeros((8, 8, 3), dtype=np.uint8)
    img_high[0, 0, 0] = 255

    best, scores = test_time_pick.pick_best_indices([img_low, img_high], "photo portrait", "combo_realism", "cpu")
    assert best == 1
    assert scores[1] > scores[0]


def test_score_aesthetic_proxy_is_bounded():
    flat = np.full((8, 8, 3), 127, dtype=np.uint8)
    s = test_time_pick.score_aesthetic_proxy(flat)
    assert 0.0 <= s <= 1.0


def test_aesthetic_metric_prefers_higher_proxy(monkeypatch):
    monkeypatch.setattr(test_time_pick, "score_aesthetic_proxy", lambda im: float(im[0, 0, 0]) / 255.0)
    low = np.zeros((4, 4, 3), dtype=np.uint8)
    high = np.full((4, 4, 3), 255, dtype=np.uint8)
    best, scores = test_time_pick.pick_best_indices([low, high], "x", "aesthetic", "cpu")
    assert best == 1
    assert scores[1] > scores[0]


def test_combo_count_vit_still_prefers_count_when_vit_is_neutral(monkeypatch):
    monkeypatch.setattr(
        test_time_pick, "score_clip_similarity", lambda imgs, prompt, device, model_id: [0.5 for _ in imgs]
    )
    monkeypatch.setattr(test_time_pick, "score_edge_sharpness", lambda im: 1.0)
    monkeypatch.setattr(
        test_time_pick,
        "score_people_count_match",
        lambda im, exp: 1.0 if int(im[0, 0, 0]) == int(exp) else 0.0,
    )
    img_wrong = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right[0, 0, 0] = 2
    best, scores = test_time_pick.pick_best_indices(
        [img_wrong, img_right], "exactly 2 people", "combo_count_vit", "cpu"
    )
    assert best == 1
    assert scores[1] > scores[0]


def test_aesthetic_realism_prefers_candidate_with_better_realism_metrics(monkeypatch):
    monkeypatch.setattr(test_time_pick, "score_photographic_detail_balance", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_exposure_balance", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_dynamic_range_headroom", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_color_cast_neutrality", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_skin_tone_naturalness", lambda im: float(im[0, 0, 0]) / 255.0)

    img_low = np.zeros((8, 8, 3), dtype=np.uint8)
    img_high = np.zeros((8, 8, 3), dtype=np.uint8)
    img_high[0, 0, 0] = 255

    best, scores = test_time_pick.pick_best_indices([img_low, img_high], "photo portrait", "aesthetic_realism", "cpu")
    assert best == 1
    assert scores[1] > scores[0]


def test_vit_views_keep_aspect_ratio():
    from PIL import Image

    wide = test_time_pick._vit_views(Image.new("RGB", (600, 400)), 224)
    assert len(wide) == 3 and all(v.size == (224, 224) for v in wide)
    assert len(test_time_pick._vit_views(Image.new("RGB", (300, 300)), 224)) == 1


def test_color_cast_magnitude():
    gray = np.full((64, 64, 3), 128.0, dtype=np.float32)
    warm = np.clip(gray + np.array([30.0, 20.0, -35.0]), 0, 255).astype(np.uint8)
    assert test_time_pick.color_cast_magnitude(warm) > 0.06
    neutral = np.random.default_rng(0).integers(90, 170, size=(64, 64, 3), dtype=np.uint8)
    assert test_time_pick.color_cast_magnitude(neutral) < 0.06
    assert test_time_pick.color_cast_magnitude(np.zeros((16, 16, 3), dtype=np.uint8)) == 0.0
    assert test_time_pick.color_cast_magnitude(np.full((16, 16, 3), 255, dtype=np.uint8)) == 0.0


# ==================================================================================================
# test_variation_guard
# ==================================================================================================


def _rand_image(seed: int) -> np.ndarray:
    # Structurally distinct images (different composition + palette per seed);
    # iid noise would downsample to identical luma and correctly read as similar.
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) / 63.0
    angle = rng.uniform(0, np.pi)
    ramp = np.cos(angle) * xx + np.sin(angle) * yy
    base = rng.uniform(40, 200, size=3)
    for c in range(3):
        img[..., c] = base[c] + 80.0 * ramp
    x0, y0 = rng.integers(0, 40, size=2)
    img[y0 : y0 + 20, x0 : x0 + 20] = rng.uniform(0, 255, size=3)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_identical_batch_collapses():
    img = _rand_image(1)
    report = audit_variation([img.copy() for _ in range(4)])
    assert report.collapse_score > 0.9
    assert report.min_distance < 1e-6
    assert len(report.near_duplicate_pairs) == 6  # all pairs


def test_diverse_batch_is_healthy():
    report = audit_variation([_rand_image(s) for s in range(4)])
    assert report.collapse_score < 0.9
    assert not report.near_duplicate_pairs


def test_reseed_flags_duplicates_not_unique_images():
    base = _rand_image(1)
    batch = [base, base.copy(), _rand_image(2), _rand_image(3)]
    report = audit_variation(batch)
    flagged = reseed_indices(report, max_collapse=0.99)
    assert flagged in ([0], [1])  # one of the duplicate pair, not the unique images


def test_reseed_empty_for_diverse_batch():
    report = audit_variation([_rand_image(s) for s in range(4)])
    assert reseed_indices(report) == []


def test_single_image_batch_is_trivially_ok():
    report = audit_variation([_rand_image(0)])
    assert report.collapse_score == 0.0
    assert report.per_image_uniqueness == [0.0]


# ==================================================================================================
# test_realism_and_hard_prompts: Photoreal shot spec / critics / finishing and hard-prompt plans (utils/prompt/realism.py, utils/quality/realism.py).
# ==================================================================================================


SKIN = (224, 172, 140)


def _skin_image(noise: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((192, 192, 3), SKIN, dtype=np.float32)
    img += rng.normal(0.0, noise, (192, 192, 1))  # luminance texture, as pores / sensor noise
    return np.clip(img, 0, 255)


# --- shot spec ---------------------------------------------------------------------------------------------


def test_shot_spec_agrees_with_light_and_lens() -> None:
    night = plan_shot_spec("85mm f/1.8 portrait photo of a woman at night on a city street")
    assert night.focal_mm == 85 and night.aperture == 1.8
    assert night.shutter_s <= 1 / 85 and night.iso >= 800  # handheld limit, ISO makes up the light
    gold = plan_shot_spec("golden hour landscape photo")
    assert gold.iso == 100 and "clean low-noise detail" in gold.caption()
    dark = plan_shot_spec("night street photo of a market, 35mm")
    assert dark.iso >= 1600 and "luminance grain" in dark.caption()
    action = plan_shot_spec("photo of a kid running on the beach at noon")
    assert action.motion == "action" and action.shutter_s <= 1 / 1000
    astro = plan_shot_spec("photo of the milky way over mountains")
    assert astro.motion == "astro" and astro.aperture <= 2.8 and astro.shutter_s >= 15


def test_shot_spec_keeps_what_the_user_said() -> None:
    s = plan_shot_spec("street photo, ISO 400, 1/500s")
    assert s.iso == 400 and abs(s.shutter_s - 1 / 500) < 1e-9
    assert "ISO" not in s.caption() and "shutter" not in s.caption()
    assert plan_shot_spec("analog film photo of a cafe, portra 400").iso == 0  # the film stock decides


def test_apply_realism_only_for_photo_prompts() -> None:
    photo = SimpleNamespace(realism="auto", prompt="photo of an old fisherman at night, harbor", shot_spec="")
    spec = apply_realism_to_args(photo)
    assert spec is not None and photo._shot_caption and photo.shot_spec_plan["iso"] == spec.iso
    painting = SimpleNamespace(realism="auto", prompt="an oil painting of a harbor at night", shot_spec="")
    assert apply_realism_to_args(painting) is None and painting._shot_caption == ""
    assert apply_realism_to_args(SimpleNamespace(prompt="photo of a cat")) is None  # no flag = off
    quiet = SimpleNamespace(realism="auto", prompt="photo of a cat", shot_spec="off")
    assert apply_realism_to_args(quiet) is not None and quiet._shot_caption == ""
    maxed = SimpleNamespace(realism="max", prompt="photo of a cat", shot_spec="", pick_best="none", num=1)
    apply_realism_to_args(maxed)
    assert maxed.pick_best == "combo_realism" and maxed.num == 4


def test_shot_spec_override_joins_prompt() -> None:
    args = SimpleNamespace(
        shot_spec="85mm f/1.4 ISO 6400", prompt="photo of a dancer", _user_prompt="photo of a dancer"
    )
    apply_shot_spec_override(args)
    assert args.prompt.endswith("ISO 6400") and args._user_prompt.endswith("ISO 6400")
    assert plan_shot_spec(args.prompt).iso == 6400


# --- hard prompts ------------------------------------------------------------------------------------------


def test_count_plan_has_one_region_per_instance() -> None:
    plan = plan_hard_prompt("five apples on a wooden table")
    assert plan.count == (5, "apple") and plan.pick_metric == "combo_count"
    apples = [r for r in plan.layout["regions"] if r["id"].startswith("apple")]
    assert len(apples) == 5 and all(r["prompt"].strip() == "one apple" for r in apples)
    assert "exactly 5 apples" in plan.positive and "6 apples" in plan.negative
    assert plan_hard_prompt("a dragon with two wings").count == (0, "")  # body parts are not instances


def test_spatial_and_binding_layout() -> None:
    plan = plan_hard_prompt("a red cube to the left of a blue sphere")
    assert plan.binding and plan.spatial and plan.pick_metric == "combo_spatial_bind"
    cube, sphere = plan.layout["regions"]
    assert cube["prompt"] == "red cube" and sphere["prompt"] == "blue sphere"
    assert (cube["box"][0] + cube["box"][2]) < (sphere["box"][0] + sphere["box"][2])  # cube centre is left
    assert "blue cube" in cube["negative"] and "red sphere" in sphere["negative"]
    stacked = plan_hard_prompt("a green cup on top of a yellow book")
    cup, book = stacked.layout["regions"]
    assert cup["box"][3] <= book["box"][3] and cup["box"][1] < book["box"][1]  # cup sits higher
    assert plan_hard_prompt("a black and white cat sleeping on a sofa").layout is None  # one two-tone subject


def test_people_hands_and_plain_prompts() -> None:
    assert plan_hard_prompt("close-up photo of hands holding a cup").pick_metric == "combo_hand"
    group = plan_hard_prompt("three women laughing at a cafe")
    assert group.people == 3 and "cloned faces" in group.negative
    plain = plan_hard_prompt("portrait of an old fisherman, golden hour, photo")
    assert not plain.active and plain.layout is None


def test_expected_text_extraction() -> None:
    texts = extract_expected_texts("a coffee shop sign that says OPEN DAILY next to a poster “SALE”")
    assert texts == ["SALE", "OPEN DAILY"]
    assert extract_expected_texts("a bottle labeled 'Cold Brew'") == ["Cold Brew"]
    assert extract_expected_texts("a quiet street at dawn") == []
    from utils.generation.sample_core import _infer_expected_texts_from_prompt

    assert _infer_expected_texts_from_prompt('a sign that says "HI"') == ["HI"]
    assert _infer_expected_texts_from_prompt("street sign reading MAIN ST") == ["MAIN ST"]


def test_apply_hard_prompts_to_namespace(tmp_path) -> None:
    args = SimpleNamespace(
        hard_prompts="auto", prompt='four red apples, a sign that says "FRESH"', negative_prompt="", box_layout=""
    )
    apply_hard_prompts_to_args(args, work_dir=str(tmp_path))
    layout = json.loads(open(args.box_layout, encoding="utf-8").read())
    assert len(layout["regions"]) == 4 and args.anti_bleed
    assert args.expected_count == 4 and args.expected_count_object == "apple"
    assert args.text_in_image and json.loads(args.expected_text) == ["FRESH"]
    assert "exactly 4 apples" in args.prompt and "5 apples" in args.negative_prompt
    user = SimpleNamespace(hard_prompts="auto", prompt="three cats", negative_prompt="", box_layout="mine.json")
    apply_hard_prompts_to_args(user, work_dir=str(tmp_path))
    assert user.box_layout == "mine.json"  # the user's layout wins
    repair = SimpleNamespace(hard_prompts="auto", prompt="three cats", ocr_repair_iter=1)
    assert apply_hard_prompts_to_args(repair) is None
    assert apply_hard_prompts_to_args(SimpleNamespace(prompt="three cats")) is None  # no flag = off


def test_pick_metric_follows_hard_prompt() -> None:
    from utils.generation.sample_prompt import PromptPlan
    from utils.generation.sample_run import pick_metric

    args = SimpleNamespace(pick_best="auto", _hard_pick_metric="combo_count")
    assert pick_metric(args, PromptPlan(prompt="five apples", negative="")) == "combo_count"
    texty = PromptPlan(prompt="x", negative="", expected_texts=["OPEN"])
    assert pick_metric(args, texty) != "combo_count"  # text constraints keep the OCR route
    args.pick_best = "clip"
    assert pick_metric(args, PromptPlan(prompt="x", negative="")) == "clip"


def test_weak_domain_and_hand_inference() -> None:
    from utils.prompt.content_controls import _DOMAIN_POSITIVE, infer_content_controls_from_prompt

    assert infer_content_controls_from_prompt("product photo of a wristwatch")["scene_domain"] == "product"
    assert infer_content_controls_from_prompt("a bowl of ramen on a counter")["scene_domain"] == "food"
    assert infer_content_controls_from_prompt("exploded view of a gearbox")["scene_domain"] == "technical"
    assert infer_content_controls_from_prompt("a cozy living room")["scene_domain"] == "interior"
    assert infer_content_controls_from_prompt("a red car on a road")["scene_domain"] == "vehicles"  # unchanged
    assert infer_content_controls_from_prompt("a chef holding a knife")["hand_mode"] == "grip"
    assert infer_content_controls_from_prompt("five friends at a picnic")["people_layout"] == "group_small"
    assert {"product", "food", "interior", "machinery", "nature_macro", "technical", "diagram"} <= set(_DOMAIN_POSITIVE)


# --- critics -----------------------------------------------------------------------------------------------


def test_plastic_skin_and_waxy_highlights() -> None:
    assert score_plastic_skin(_skin_image(0.0)) < 0.2
    assert score_plastic_skin(_skin_image(3.0)) > 0.9
    assert score_plastic_skin(np.full((192, 192, 3), (40, 90, 200), np.float32)) is None  # no skin
    waxy = _skin_image(0.0)
    cv2.circle(waxy, (96, 96), 30, (255, 255, 255), -1)
    textured = _skin_image(3.0, seed=1)
    cv2.circle(textured, (96, 96), 30, (240, 215, 200), -1)
    textured[66:127, 66:127] += np.random.default_rng(2).normal(0, 3.0, (61, 61, 1))
    w, t = score_waxy_highlights(waxy), score_waxy_highlights(textured)
    assert w is not None and t is not None and w < t


def test_ai_look_pools_skin_parts_for_photos_only() -> None:
    from utils.quality.critics import ai_look_score

    img = _skin_image(0.0)
    assert "plastic_skin" in ai_look_score(img, "photo of a face").parts
    assert "plastic_skin" not in ai_look_score(img, "anime illustration of a face").parts


# --- finishing ---------------------------------------------------------------------------------------------


def test_saturation_tame_only_when_clipped() -> None:
    from utils.quality.finishing import srgb_to_lab

    def chroma(im: np.ndarray) -> float:
        lab = srgb_to_lab(im / 255.0)
        return float(np.hypot(lab[..., 1], lab[..., 2]).max())

    loud = np.zeros((64, 64, 3), np.float32)
    loud[:, :32] = (255, 0, 40)
    loud[:, 32:] = (120, 110, 100)
    out = tame_saturation(loud, 1.0)
    assert chroma(out) < chroma(loud) - 5
    assert np.allclose(out[:, 32:], loud[:, 32:], atol=0.5)  # muted pixels untouched
    calm = np.full((64, 64, 3), (150, 120, 100), np.float32)
    assert np.array_equal(tame_saturation(calm, 1.0), calm)
    assert chroma(tame_saturation(loud, 1.0, vivid_ok=True)) > chroma(out)  # a vivid request keeps more


def test_highlight_rolloff_is_a_gentle_monotone_shoulder() -> None:
    ramp_img = np.repeat(np.linspace(0, 255, 256, dtype=np.float32)[None, :, None], 3, axis=2).repeat(8, 0)
    out = highlight_rolloff(ramp_img, 1.0)
    y = out[0, :, 0]
    assert np.all(np.diff(y) >= -1e-4) and 240 < y[-1] < 255  # compressed, still near white
    assert np.allclose(out[:, :200], ramp_img[:, :200], atol=1e-3)  # below the knee untouched
    dim = np.full((32, 32, 3), 100.0, np.float32)
    assert np.allclose(highlight_rolloff(dim, 1.0), dim)


def test_lens_follows_focal_length_and_user_knobs() -> None:
    from utils.quality.finishing import CAMERA_PRESETS

    base = CAMERA_PRESETS["dslr"].lens
    wide, tele = lens_for_focal(base, 24, 8.0), lens_for_focal(base, 85, 1.8)
    assert wide.half_fov_deg > tele.half_fov_deg and abs(wide.ca_red) > abs(tele.ca_red)
    assert tele.vignette > wide.vignette  # wide open vignettes more
    assert lens_for_focal(base, 85, 1.8, keep_vignette=True).vignette == base.vignette
    phone = SimpleNamespace(camera="phone", focal_mm=85, aperture=1.8)
    finish = CAMERA_PRESETS["phone"]
    assert tune_finish(finish, phone) is finish


def test_capture_finish_uses_shot_spec_iso() -> None:
    from utils.generation.sample_prompt import PromptPlan
    from utils.generation.sample_run import _capture_finish

    spec = plan_shot_spec("night street photo of a market, 35mm")
    args = SimpleNamespace(_raw_prompt_before_compose="night street photo of a market, 35mm", _shot_spec=spec)
    report: dict = {}
    finish, _ = _capture_finish(args, PromptPlan(prompt="", negative=""), report)
    assert finish is not None and finish.iso == spec.iso and report["capture_finish"]["iso"] == spec.iso
    assert abs(finish.lens.half_fov_deg - 31.7) < 0.5  # 35 mm on full frame


# ==================================================================================================
# test_visual_design: Tests for ``utils.visual_design`` prompt packs, presets and ``sample.py`` argv.
# ==================================================================================================


def test_every_domain_has_all_tiers_and_presets_resolve():
    assert set(DOMAIN_POSITIVES) == set(DOMAIN_NEGATIVES)
    for domain in DOMAIN_POSITIVES:
        for tier in ("lite", "standard", "strong"):
            assert DOMAIN_POSITIVES[domain][tier].strip()
            assert DOMAIN_NEGATIVES[domain][tier].strip()
    for domain, _ in VISUAL_DESIGN_PRESETS.values():
        assert domain in DOMAIN_POSITIVES


def test_design_pack_ids_and_cli_choices():
    ids = design_pack_ids()
    for k in (
        "ui_ux",
        "brand",
        "stem",
        "editorial_layout",
        "presentation_slide",
        "technical_blueprint",
        "fashion_flat",
    ):
        assert k in ids
    ch = visual_design_cli_domain_choices()
    assert ch[:2] == ("none", "auto")
    assert "textbook" in ch


def test_merge_visual_fragments_drops_empty():
    assert merge_visual_fragments("a", "", "  b ") == "a, b"


def test_build_pair_appends_fragment_once_and_returns_negative():
    pos, neg = build_visual_design_prompt_pair("hero shot of a soda can", "packaging", intensity="standard")
    assert pos.startswith("hero shot of a soda can, ")
    assert neg
    again, _ = build_visual_design_prompt_pair(pos, "packaging", intensity="standard")
    assert again == pos


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="Unknown"):
        build_visual_design_prompt_pair("x", "not_a_domain")


def test_prompt_suggests_domain():
    assert prompt_suggests_domain("Redesign our app settings screen in Figma style") == "ui_ux"
    assert prompt_suggests_domain("minimal wordmark for a coffee roaster") == "brand"
    assert prompt_suggests_domain("patent drawing exploded isometric view") == "technical_blueprint"
    assert prompt_suggests_domain("random abstract splatter") is None


def test_stage_auto_heuristic():
    logs: list[str] = []
    r = apply_visual_design_stage(
        " keynote slide comparing Q2 revenue ",
        cli_domain="auto",
        intensity="lite",
        use_negative_pack=False,
        emit=logs.append,
    )
    assert r.resolved_domain == "presentation_slide"
    assert "keynote slide" in r.prompt.lower()
    assert r.negative_addon == ""
    assert logs


def test_stage_negative_pack_optional():
    on = apply_visual_design_stage("hero bottle render", cli_domain="packaging", use_negative_pack=True)
    off = apply_visual_design_stage("hero bottle render", cli_domain="packaging", use_negative_pack=False)
    assert on.negative_addon and not off.negative_addon


def test_preset_resolve():
    assert apply_visual_design_preset_to_prompt("logo", "logo_lockup") == ("logo", "brand", "standard")
    assert apply_visual_design_preset_to_prompt("logo", "nope") == ("logo", "none", "standard")


def test_extend_sample_argv_preset_only():
    ns = SimpleNamespace(
        visual_design_preset="saas_ui",
        visual_design_domain="none",
        visual_design_intensity="standard",
        visual_design_negative_pack=True,
    )
    cmd: list[str] = []
    extend_sample_argv_visual_design(cmd, ns)
    assert cmd == ["--visual-design-preset", "saas_ui", "--visual-design-negative-pack"]


def test_extend_sample_argv_domain_path():
    ns = SimpleNamespace(
        visual_design_preset="",
        visual_design_domain="stem",
        visual_design_intensity="strong",
        visual_design_negative_pack=False,
    )
    cmd: list[str] = []
    extend_sample_argv_visual_design(cmd, ns)
    assert cmd == ["--visual-design-domain", "stem", "--visual-design-intensity", "strong"]
