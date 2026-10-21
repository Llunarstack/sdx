"""Art critic on synthetic images, and intent-aware scoring that never penalises a requested look.

Also holds what used to be: test_art_critic, test_art_direction, test_art_mediums.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import cv2
import numpy as np
from config.defaults.art_mediums import detect_medium_ids, guidance_fragments
from utils.prompt.art_direction import apply_art_direction_to_args, gate_tag, plan_art_direction
from utils.quality import art_critic, test_time_pick

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
    level_score, level_tilt = art_critic.score_horizon(_seascape(0.0))
    tilt_score, tilt = art_critic.score_horizon(_seascape(3.0))
    assert level_score == 1.0 and abs(level_tilt) < 0.3
    assert tilt_score == 0.0 and abs(abs(tilt) - 3.0) < 0.3
    assert art_critic.score_horizon(_seascape(3.0), {"composition": "dutch"}) == (None, None)


def test_composition_follows_the_plan() -> None:
    centred, thirds = _disc(0.5, 0.5), _disc(1 / 3, 1 / 3)
    default = art_critic.score_composition
    assert default(thirds)["composition"] > default(centred)["composition"]
    plan = {"composition": "center"}
    assert default(centred, plan)["composition"] > default(thirds, plan)["composition"]
    assert default(_disc(0.5, 0.0))["crop"] < default(centred)["crop"]


def test_colour_harmony_prefers_template_fit() -> None:
    complementary, _, dev = art_critic.score_color_harmony(_hues([30, 210]))
    scattered = art_critic.score_color_harmony(_hues(list(np.random.default_rng(2).uniform(0, 360, 200))))[0]
    assert complementary == 1.0 and dev < 1.0
    assert scattered is not None and scattered < complementary


def test_low_key_plan_rewards_a_dark_frame() -> None:
    dark = np.clip(np.random.default_rng(3).normal(25, 15, (128, 128, 3)), 0, 255).astype(np.uint8)
    assert art_critic.score_exposure(dark, {"key": "low"})[0] > art_critic.score_exposure(dark)[0]


def test_requested_teal_orange_grade_is_not_penalised() -> None:
    plan = art_critic.resolve_plan("night street portrait, teal and orange grade")
    assert plan["grade"] == "teal_orange"
    assert art_critic.resolve_plan("a portrait", photo_color_grade="teal_orange")["grade"] == "teal_orange"
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
    from config.defaults.quality_policy import PROFILES
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
# test_art_mediums: Tests for config.defaults.art_mediums guidance packs.
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
