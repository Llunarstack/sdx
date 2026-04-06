"""Tests for utils.training.ar_curriculum runtime resolution."""

from utils.training.ar_curriculum import parse_ar_order_mix, resolve_ar_for_step


def test_parse_ar_order_mix_filters_and_dedupes():
    out = parse_ar_order_mix("raster,zorder,snake,spiral,zorder,invalid")
    assert out == ["raster", "zorder", "snake", "spiral"]


def test_resolve_ar_for_step_linear_and_mix():
    # linear ramp 0 -> 4 over steps; order cycles deterministic.
    b0, o0 = resolve_ar_for_step(
        0,
        base_blocks=2,
        base_order="raster",
        curriculum_mode="linear",
        ramp_start=0,
        ramp_end=100,
        curriculum_start_blocks=0,
        curriculum_target_blocks=4,
        order_mix="raster,snake",
    )
    b1, o1 = resolve_ar_for_step(
        60,
        base_blocks=2,
        base_order="raster",
        curriculum_mode="linear",
        ramp_start=0,
        ramp_end=100,
        curriculum_start_blocks=0,
        curriculum_target_blocks=4,
        order_mix="raster,snake",
    )
    assert b0 == 0 and o0 in ("raster", "snake")
    assert b1 in (2, 4) and o1 in ("raster", "snake")


def test_resolve_ar_for_step_none_uses_base():
    b, o = resolve_ar_for_step(
        999,
        base_blocks=2,
        base_order="zorder",
        curriculum_mode="none",
    )
    assert b == 2
    assert o == "zorder"
