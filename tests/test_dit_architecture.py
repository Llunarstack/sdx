"""Tests for utils/dit_architecture.py and utils/nn_inspect.py."""

from __future__ import annotations

import torch
from utils.architecture.dit_architecture import (
    default_dit_profile_kwargs,
    dit_parameter_report,
    instantiate_dit_text,
    latent_side_from_image_size,
    list_dit_text_variant_names,
)
from utils.modeling.nn_inspect import child_parameter_summary, format_module_tree


def test_latent_side() -> None:
    assert latent_side_from_image_size(256) == 32
    assert latent_side_from_image_size(512) == 64


def test_default_kwargs_has_required_keys() -> None:
    kw = default_dit_profile_kwargs()
    assert kw["input_size"] == 32
    assert kw["text_dim"] == 4096


def test_dit_b_instantiate_and_report() -> None:
    m = instantiate_dit_text("DiT-B/2-Text", image_size=256, text_dim=4096)
    r = dit_parameter_report(m)
    assert r["total_parameters"] > 100_000_000
    assert r["size_bf16_gib"] < r["size_fp32_gib"]
    del m


def test_nn_inspect_on_linear() -> None:
    m = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU())
    rows = child_parameter_summary(m)
    assert len(rows) >= 1
    lines = format_module_tree(m, max_depth=3)
    assert any("total_params" in line for line in lines)


def test_list_dit_text_excludes_enhanced() -> None:
    names = list_dit_text_variant_names()
    assert "DiT-XL/2-Text" in names
    assert not any(n.startswith("EnhancedDiT") for n in names)
