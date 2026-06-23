"""Tests for research-backed frontier modules."""

from __future__ import annotations

import json
import torch

from frontier.registry import list_ideas, idea_by_id
from frontier.guidance import DynamicCFGPicker, GuidanceInterval, cfg_multiplier_for_step
from frontier.layout import (
    OmostCanvas,
    canvas_to_box_layout,
    bind_coordinates_to_prompt,
    parse_loc_tokens,
    RegionFusionSchedule,
    fusion_weight_at_step,
    score_layout_masks,
)
from frontier.attention import build_attention_layout_plan
from frontier.compose import references_from_layout, merge_reference_prompts
from utils.generation.regional_box_prompting import parse_box_layout, build_latent_region_masks


class TestResearchRegistry:
    def test_has_implemented_ideas(self):
        impl = list_ideas(status="implemented")
        assert len(impl) >= 8
        assert idea_by_id("omost_canvas").status == "implemented"


class TestDynamicCFG:
    def test_picks_from_candidates(self):
        x = torch.randn(1, 4, 8, 8)
        cfg = DynamicCFGPicker(candidates=(5.0, 7.5, 10.0)).pick(x)
        assert cfg in (5.0, 7.5, 10.0)


class TestGuidanceInterval:
    def test_skips_early(self):
        gi = GuidanceInterval(skip_early_frac=0.2)
        assert cfg_multiplier_for_step(0, 20, gi) == 0.0
        assert cfg_multiplier_for_step(15, 20, gi) > 0.0


class TestOmostCanvas:
    def test_exports_box_layout(self):
        c = OmostCanvas()
        c.set_global_description("sunset landscape")
        c.add_local_description("red dragon", anchor="left", name="dragon")
        c.add_local_description("castle", box=(0.55, 0.2, 0.95, 0.9), name="castle")
        d = canvas_to_box_layout(c)
        assert d["global_prompt"] == "sunset landscape"
        assert len(d["regions"]) == 2


class TestCoordinateBind:
    def test_roundtrip_token(self):
        p = bind_coordinates_to_prompt("a cat", (0.1, 0.2, 0.5, 0.8))
        boxes = parse_loc_tokens(p)
        assert len(boxes) == 1
        assert boxes[0][0] == 0.1


class TestLamicSchedule:
    def test_early_more_isolated(self):
        s = RegionFusionSchedule()
        early = fusion_weight_at_step(1, 30, s)
        late = fusion_weight_at_step(28, 30, s)
        assert late > early


class TestLayoutMetrics:
    def test_scores_masks(self):
        spec = parse_box_layout(
            {
                "regions": [
                    {"name": "a", "box": [0, 0, 0.5, 1], "prompt": "x"},
                    {"name": "b", "box": [0.5, 0, 1, 1], "prompt": "y"},
                ]
            }
        )
        rm, bg = build_latent_region_masks(spec, 16, 16, device=torch.device("cpu"))
        rep = score_layout_masks(rm, bg)
        assert rep.inclusion_ratio == 1.0


class TestAttentionPlan:
    def test_enforce_early_steps(self):
        spec = parse_box_layout(
            {"regions": [{"name": "a", "box": [0, 0, 1, 1], "prompt": "x"}]}
        )
        plan = build_attention_layout_plan(spec.regions, num_steps=20, inject_frac=0.5)
        assert len(plan.enforce_steps) == 10


class TestMultiReference:
    def test_parse_reference_field(self):
        spec = parse_box_layout(
            {
                "regions": [
                    {
                        "name": "face",
                        "box": [0, 0, 0.5, 1],
                        "prompt": "portrait",
                        "reference": "refs/face.png",
                    }
                ]
            }
        )
        refs = references_from_layout(spec.regions)
        assert len(refs) == 0  # file missing on disk
        merged = merge_reference_prompts("portrait", [])
        assert merged == "portrait"


class TestRegionalInjectConfig:
    def test_parse_inject_keys(self):
        spec = parse_box_layout(
            {
                "mask_inject_steps": 8,
                "base_ratio": 0.2,
                "coordinate_tokens": True,
                "regions": [{"name": "a", "box": [0, 0, 1, 1], "prompt": "sky"}],
            }
        )
        assert spec.inject.mask_inject_steps == 8
        assert spec.inject.base_ratio == 0.2
        assert spec.inject.use_coordinate_tokens is True
