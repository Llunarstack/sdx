"""Tests for OCR repair argv forwarding."""

from __future__ import annotations

import argparse

from utils.generation.sample_cli_passthrough import append_sample_repair_passthrough


def test_append_sample_repair_passthrough_includes_core_flags() -> None:
    args = argparse.Namespace(
        width=512,
        height=768,
        resize_mode="stretch",
        vae_tiling=True,
        boost_quality=True,
        style="anime",
        style_strength=0.8,
        holy_grail=True,
        holy_grail_cfg_early_ratio=0.7,
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
        lora=[],
        control=[],
    )
    cmd: list[str] = ["python", "sample.py"]
    append_sample_repair_passthrough(cmd, args)
    assert "--width" in cmd and "512" in cmd
    assert "--text-in-image" in cmd
    assert "--boost-quality" in cmd
    assert "--style" in cmd
