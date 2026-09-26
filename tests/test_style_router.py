"""Tests for universal style router + motion grammars."""

from __future__ import annotations

from pipelines.video.motion_grammar import grammar_for_engine, grammar_for_prompt
from pipelines.video.process_options import ProcessOptions
from pipelines.video.style_router import apply_style_route_to_options, list_styles, route_style
from pipelines.video.videomax import apply_videomax_to_options, plan_videomax


def test_list_styles_covers_media() -> None:
    styles = set(list_styles())
    for need in (
        "realistic",
        "film",
        "anime_2d",
        "cartoon",
        "spider_verse",
        "pixar_3d",
        "cgi",
        "vfx",
        "product",
        "stop_motion",
        "lego",
        "pixel_art",
        "low_poly",
        "voxel",
        "vector",
        "dream_logic",
        "hybrid",
    ):
        assert need in styles


def test_route_anime_softens_physics() -> None:
    route = route_style("sakuga anime fight, cel shaded")
    assert route.style == "anime_2d"
    assert route.grammar == "anime_2d"
    assert route.videomax_profile.get("physics_gate") is False
    assert route.videomax_profile.get("permanence_repair") is False
    assert float(route.videomax_profile.get("deflicker_strength", 1.0)) < 0.7


def test_route_vfx_tightens_permanence() -> None:
    route = route_style("houdini explosion vfx plate, volumetric fire")
    assert route.style == "vfx"
    assert float(route.videomax_profile.get("min_permanence", 0)) >= 0.65
    assert route.videomax_profile.get("physics_repair") is True


def test_route_stop_motion_keeps_jitter() -> None:
    route = route_style("claymation wallace and gromit stop-motion")
    assert route.style == "stop_motion"
    assert float(route.videomax_profile.get("deflicker_strength", 1.0)) <= 0.45
    assert route.videomax_profile.get("hf_deshimmer") is False


def test_route_force_overrides_prompt() -> None:
    route = route_style("photoreal documentary", force="cartoon")
    assert route.style == "cartoon"
    assert route.grammar == "cartoon"


def test_apply_style_route_to_options() -> None:
    opts = ProcessOptions()
    route = route_style("pixar stylized 3d character")
    out = apply_style_route_to_options(opts, route)
    assert getattr(out, "motion_grammar", "") == "pixar_3d"
    assert getattr(out, "style_route", {}).get("style") == "pixar_3d"


def test_videomax_style_aware_anime() -> None:
    plan = plan_videomax("ghibli anime walk cycle")
    assert plan.style == "anime_2d"
    assert plan.option_overrides.get("physics_gate") is False
    assert plan.option_overrides.get("permanence_repair") is False
    opts = apply_videomax_to_options(ProcessOptions(), prompt="ghibli anime walk cycle")
    assert getattr(opts, "videomax", False) is True
    assert getattr(opts, "motion_grammar", "") == "anime_2d"


def test_videomax_style_aware_realistic() -> None:
    plan = plan_videomax("photoreal handheld documentary street")
    assert plan.style == "realistic"
    assert plan.option_overrides.get("motion_shutter") is True
    assert plan.option_overrides.get("contact_ground") is True


def test_grammars_exist_for_all_engines() -> None:
    for name in (
        "realistic",
        "film",
        "anime_2d",
        "cartoon",
        "spider_verse",
        "vfx",
        "pixar_3d",
        "stop_motion",
        "pixel_art",
        "low_poly",
        "vector",
        "dream_logic",
        "hybrid",
        "product",
    ):
        g = grammar_for_engine(name)
        assert g.name == name


def test_grammar_prompt_detects_pixel_and_lego() -> None:
    assert grammar_for_prompt("16-bit pixel art hero run").name == "pixel_art"
    assert grammar_for_prompt("lego brickfilm chase").name == "stop_motion"
