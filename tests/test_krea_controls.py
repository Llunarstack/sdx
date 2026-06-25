"""Tests for Krea 2–inspired controls (sliders, creativity, style refs)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from utils.generation.krea_controls import (
    CreativityMode,
    aggregate_style_embeddings,
    apply_creativity_mode_to_prompt,
    apply_generative_sliders,
    apply_krea_controls_to_args,
    apply_turbo_preset,
    build_generative_slider_plan,
    parse_style_references,
)


def test_parse_style_references_csv():
    refs = parse_style_references(csv_spec="a.png:0.6,b.png:0.4")
    assert len(refs) == 2
    assert refs[0].path == "a.png"
    assert abs(refs[0].strength - 0.6) < 1e-6


def test_parse_style_references_json(tmp_path: Path):
    p = tmp_path / "refs.json"
    p.write_text(
        json.dumps({"references": [{"path": "x.png", "strength": 0.8}]}),
        encoding="utf-8",
    )
    refs = parse_style_references(json_path=str(p))
    assert len(refs) == 1
    assert refs[0].path == "x.png"


def test_generative_sliders_high_movement():
    plan = build_generative_slider_plan(movement=80.0)
    assert "dynamic" in plan.prompt_additions.lower()
    assert plan.serendipity_offset > 0


def test_generative_sliders_apply_cfg():
    args = SimpleNamespace(
        prompt="cat",
        negative_prompt="",
        cfg_scale=7.5,
        reference_strength=1.0,
        slider_intensity=50.0,
        slider_complexity=0.0,
        slider_movement=0.0,
        creativity_mode="medium",
    )
    plan = apply_generative_sliders(args)
    assert plan.cfg_multiplier > 1.0
    assert float(args.cfg_scale) > 7.5
    assert "bold" in args.prompt.lower()


def test_creativity_mode_raw_skips_expansion():
    args = SimpleNamespace(prompt="sunset", creativity_mode="raw")
    mode = apply_creativity_mode_to_prompt(args)
    assert mode == CreativityMode.RAW
    assert getattr(args, "_skip_prompt_expansion") is True
    assert args.prompt == "sunset"


def test_creativity_mode_high_expands_short_prompt():
    args = SimpleNamespace(prompt="dragon", creativity_mode="high", frontier_serendipity=0.2)
    apply_creativity_mode_to_prompt(args)
    assert len(args.prompt) > len("dragon")
    assert args.frontier_serendipity > 0.2


def test_turbo_preset():
    args = SimpleNamespace(steps=50, cfg_scale=7.5)
    apply_turbo_preset(args)
    assert args.steps == 8
    assert args.cfg_scale == 1.0


def test_apply_krea_controls_bundle():
    args = SimpleNamespace(
        krea_turbo_preset=True,
        creativity_mode="medium",
        slider_intensity=0.0,
        slider_complexity=0.0,
        slider_movement=0.0,
        style_ref="",
        style_references_json="",
        moodboard_json="",
        moodboard_images="",
        prompt="a vase",
        negative_prompt="",
        cfg_scale=7.0,
        steps=50,
    )
    meta = apply_krea_controls_to_args(args)
    assert meta.get("turbo_preset") is True
    assert args.steps == 8


def test_aggregate_style_embeddings(tmp_path: Path):
    pytest.importorskip("transformers")
    for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        img = Image.new("RGB", (64, 64), color)
        img.save(tmp_path / f"ref{i}.png")
    refs = parse_style_references(csv_spec=f"{tmp_path / 'ref0.png'}:0.5,{tmp_path / 'ref1.png'}:0.5")
    emb, dim = aggregate_style_embeddings(
        refs,
        device=torch.device("cpu"),
        model_id="openai/clip-vit-large-patch14",
        dtype=torch.float32,
    )
    assert emb.shape == (1, dim)
    assert torch.isfinite(emb).all()
