from types import SimpleNamespace

from diffusion.holy_grail import (
    apply_holy_grail_preset_to_args,
    list_holy_grail_presets,
    recommend_holy_grail_preset,
    sanitize_holy_grail_kwargs,
)


def _args_defaults():
    return SimpleNamespace(
        holy_grail=False,
        holy_grail_cfg_early_ratio=0.72,
        holy_grail_cfg_late_ratio=1.0,
        holy_grail_control_mult=1.0,
        holy_grail_adapter_mult=1.0,
        holy_grail_no_frontload_control=False,
        holy_grail_late_adapter_boost=1.15,
        holy_grail_cads_strength=0.0,
        holy_grail_cads_min_strength=0.0,
        holy_grail_cads_power=1.0,
        holy_grail_unsharp_sigma=0.0,
        holy_grail_unsharp_amount=0.0,
        holy_grail_clamp_quantile=0.0,
        holy_grail_clamp_floor=1.0,
    )


def test_list_has_expected_presets():
    names = list_holy_grail_presets()
    assert "balanced" in names and "anime" in names and "photoreal" in names


def test_apply_preset_enables_holy_grail():
    a = _args_defaults()
    apply_holy_grail_preset_to_args(a, "anime")
    assert a.holy_grail is True
    assert a.holy_grail_adapter_mult >= 1.0
    assert a.holy_grail_unsharp_amount > 0.0


def test_recommend_preset_heuristics():
    assert recommend_holy_grail_preset(prompt="anime girl portrait") == "anime"
    assert recommend_holy_grail_preset(prompt="photoreal cinematic dslr") == "photoreal"
    assert recommend_holy_grail_preset(prompt="storybook illustration") == "illustration"


def test_runtime_guard_sanitizes_ranges():
    kw = dict(
        holy_grail_enable=True,
        holy_grail_cfg_early_ratio=9.0,
        holy_grail_cfg_late_ratio=-2.0,
        holy_grail_control_mult=-4.0,
        holy_grail_adapter_mult=8.0,
        holy_grail_frontload_control=True,
        holy_grail_late_adapter_boost=5.0,
        holy_grail_cads_strength=0.01,
        holy_grail_cads_min_strength=0.05,
        holy_grail_cads_power=99.0,
        holy_grail_unsharp_sigma=4.0,
        holy_grail_unsharp_amount=0.0,
        holy_grail_clamp_quantile=0.2,
        holy_grail_clamp_floor=-1.0,
    )
    out = sanitize_holy_grail_kwargs(kw)
    assert 0.4 <= out["holy_grail_cfg_early_ratio"] <= 1.4
    assert 0.4 <= out["holy_grail_cfg_late_ratio"] <= 1.6
    assert 0.0 <= out["holy_grail_control_mult"] <= 2.5
    assert 0.0 <= out["holy_grail_adapter_mult"] <= 2.5
    assert out["holy_grail_cads_min_strength"] <= out["holy_grail_cads_strength"]
    assert out["holy_grail_unsharp_sigma"] == 0.0
    assert out["holy_grail_clamp_quantile"] == 0.0
