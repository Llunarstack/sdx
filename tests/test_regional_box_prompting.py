"""Tests for Ideogram-style box + prompt regional layout."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from utils.generation.regional_box_prompting import (
    build_latent_region_masks,
    layout_text_from_regions,
    load_box_layout_file,
    parse_box_layout,
)


def test_parse_box_layout_minimal():
    data = {
        "global_prompt": "scene",
        "regions": [
            {"name": "a", "box": [0, 0, 0.5, 1], "prompt": "cat"},
            {"name": "b", "x_min": 0.5, "y_min": 0, "x_max": 1, "y_max": 1, "prompt": "dog"},
        ],
    }
    spec = parse_box_layout(data)
    assert spec.global_prompt == "scene"
    assert len(spec.regions) == 2
    assert spec.regions[0].name == "a"
    assert spec.regions[1].x1 == 0.5


def test_layout_text_merges_global_and_regions():
    spec = parse_box_layout(
        {
            "global_prompt": "wide shot",
            "regions": [{"name": "sky", "box": [0, 0, 1, 0.4], "prompt": "blue sky"}],
        }
    )
    text = layout_text_from_regions(spec)
    assert "[layout]" in text
    assert "sky:" in text
    assert text.startswith("wide shot")


def test_build_latent_masks_priority_overlap():
    spec = parse_box_layout(
        {
            "feather": 0,
            "regions": [
                {"name": "hi", "box": [0, 0, 0.6, 1], "prompt": "a", "priority": 10},
                {"name": "lo", "box": [0.4, 0, 1, 1], "prompt": "b", "priority": 1},
            ],
        }
    )
    regions, bg = build_latent_region_masks(spec, 16, 16, device=torch.device("cpu"))
    assert regions.shape[0] == 2
    assert bg.shape == (1, 1, 16, 16)
    # High-priority region owns the overlap column (~0.4-0.6); low-priority should be near zero there.
    overlap_col = int(0.5 * 16)
    assert regions[0, 0, 8, overlap_col] > 0.5
    assert regions[1, 0, 8, overlap_col] < 0.1


def test_example_json_loads():
    path = Path(__file__).resolve().parents[1] / "examples" / "box_layout.example.json"
    spec = load_box_layout_file(path)
    assert len(spec.regions) == 2
    assert spec.regions[0].priority == 10


def test_load_from_dict_keys():
    raw = json.loads((Path(__file__).resolve().parents[1] / "examples" / "box_layout.example.json").read_text())
    spec = parse_box_layout(raw)
    assert spec.feather_px == 12


def test_parse_strokes_in_region():
    spec = parse_box_layout(
        {
            "regions": [
                {
                    "name": "fig",
                    "box": [0, 0, 0.5, 1],
                    "prompt": "a cat sitting",
                    "strokes": [{"points": [[0.2, 0.2], [0.8, 0.8]], "width": 0.03}],
                }
            ]
        }
    )
    assert len(spec.regions[0].strokes) == 1
    assert len(spec.regions[0].strokes[0].points) == 2


def test_rasterize_strokes_nonzero():
    from utils.generation.regional_box_sketch import rasterize_region_sketch

    spec = parse_box_layout(
        {
            "regions": [
                {
                    "name": "a",
                    "box": [0.25, 0.25, 0.75, 0.75],
                    "prompt": "circle",
                    "strokes": [
                        {
                            "points": [[0.1, 0.5], [0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]],
                            "width": 0.05,
                        }
                    ],
                }
            ]
        }
    )
    r = spec.regions[0]
    mask = rasterize_region_sketch(r, 32, 32, device=torch.device("cpu"))
    assert mask.max() > 0.5
    assert mask[:, :, 8:24, 8:24].mean() > 0.1


def test_sketch_example_json_loads():
    path = Path(__file__).resolve().parents[1] / "examples" / "box_layout_sketch.example.json"
    spec = load_box_layout_file(path)
    assert len(spec.regions) == 3
    assert all(r.strokes for r in spec.regions)
    from utils.generation.regional_box_sketch import spec_has_sketches

    assert spec_has_sketches(spec)
