"""Tests for free-text → regional box layout compiler."""

from __future__ import annotations


def test_left_of_two_cubes():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("a red cube to the left of a blue cube")
    assert spec is not None
    assert len(spec.regions) == 2
    left = spec.regions[0]
    right = spec.regions[1]
    assert left.x2 <= 0.5
    assert right.x1 >= 0.5
    assert spec.global_prompt == "a red cube to the left of a blue cube"
    assert spec.feather_px == 10
    assert spec.overlap_mode == "priority"


def test_exactly_three_red_apples():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("exactly three red apples on a wooden table")
    assert spec is not None
    assert len(spec.regions) == 3
    xs = sorted(r.x1 for r in spec.regions)
    assert xs[0] >= 0.04
    assert all("red apples" in r.prompt for r in spec.regions)
    assert all("extra apples" in r.negative for r in spec.regions)


def test_quoted_open_sign_glyph_region():
    from utils.generation.prompt_scene import compile_spatial_layout, quoted_glyph_texts

    prompt = 'storefront sign that says "OPEN" at night'
    assert quoted_glyph_texts(prompt) == ["OPEN"]
    spec = compile_spatial_layout(prompt)
    assert spec is not None
    glyph = next(r for r in spec.regions if r.name == "glyph_text")
    assert "OPEN" in glyph.prompt
    assert glyph.x1 == 0.18
    assert glyph.y2 == 0.42


def test_no_false_positive_sitting_in_garden():
    from utils.generation.prompt_scene import compile_spatial_layout, prompt_needs_spatial_layout

    prompt = "a cat sitting in a garden"
    assert compile_spatial_layout(prompt) is None
    assert prompt_needs_spatial_layout(prompt) is False


def test_cat_on_top_of_microwave():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("cat on top of a microwave in a kitchen")
    assert spec is not None
    assert len(spec.regions) == 2
    cat = next(r for r in spec.regions if "cat" in r.name)
    micro = next(r for r in spec.regions if "microwave" in r.name)
    assert cat.y1 < micro.y1
    assert cat.priority == 10


def test_occlusion_chain_link_fence():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("a person behind a chain-link fence")
    assert spec is not None
    names = {r.name for r in spec.regions}
    assert "occluder" in names
    assert "subject" in names
    occluder = next(r for r in spec.regions if r.name == "occluder")
    subject = next(r for r in spec.regions if r.name == "subject")
    assert occluder.priority > subject.priority
    assert spec.global_negative


def test_chrome_kettle_reflection():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("chrome kettle reflection")
    assert spec is not None
    assert len(spec.regions) == 2
    reflection = next(r for r in spec.regions if r.name == "reflection")
    subject = next(r for r in spec.regions if r.name == "subject")
    assert reflection.y1 > subject.y1
    assert spec.global_negative


def test_lamp_next_to_chair():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("a lamp next to a chair")
    assert spec is not None
    assert len(spec.regions) == 2
    left, right = spec.regions
    assert left.x2 <= 0.5
    assert right.x1 >= 0.5


def test_holding_lantern_regions():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("a sailor holding a brass lantern on wet rocks")
    assert spec is not None
    names = {r.name for r in spec.regions}
    assert "held_object" in names
    assert "holder" in names
    held = next(r for r in spec.regions if r.name == "held_object")
    holder = next(r for r in spec.regions if r.name == "holder")
    assert held.y1 > holder.y1
    assert "lantern" in held.prompt.lower()


def test_lamp_between_chair_and_stool():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("a yellow lamp between a red chair and a blue stool")
    assert spec is not None
    assert len(spec.regions) == 3
    xs = sorted((r.x1 + r.x2) / 2 for r in spec.regions)
    assert xs[0] < xs[1] < xs[2]


def test_foreground_background_planes():
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout("a brass lantern in the foreground and storm clouds in the background")
    assert spec is not None
    names = {r.name for r in spec.regions}
    assert "foreground" in names
    assert "background" in names
    fg = next(r for r in spec.regions if r.name == "foreground")
    bg = next(r for r in spec.regions if r.name == "background")
    assert fg.priority > bg.priority
    assert fg.y1 > bg.y1
