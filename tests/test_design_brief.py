"""Layout-first design brief, Ideogram JSON, hex palette lock."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from utils.generation.design_brief import (
    brief_to_caption,
    compile_design_brief,
    extract_hex_palette,
)
from utils.generation.regional_box_prompting import parse_box_layout
from utils.generation.stills_gaps import apply_design_brief, apply_design_finish
from utils.quality.palette_lock import apply_palette_lock, score_palette_lock


def test_example_brief_file_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "design_brief.example.json"
    spec = parse_box_layout(__import__("json").loads(path.read_text(encoding="utf-8")))
    names = {r.name for r in spec.regions}
    assert "headline" in names
    assert "product" in names
    assert any("NIGHT ROAST" in r.prompt for r in spec.regions)
    pal = extract_hex_palette("poster in #c41e3a and rgb(26, 26, 46)")
    assert "#C41E3A" in pal
    assert "#1A1A2E" in pal


def test_nl_brief_pins_lettering_and_palette() -> None:
    prompt = 'a shop sign that says "OPEN" in #C41E3A on a navy wall'
    brief = compile_design_brief(prompt)
    assert "OPEN" in {el.content for el in brief.elements if el.kind == "text"}
    assert "#C41E3A" in brief.palette
    caption = brief_to_caption(brief, prompt)
    assert "lettering reads exactly" in caption
    assert "#C41E3A" in caption


def test_ideogram_json_loads_as_box_layout() -> None:
    data = {
        "description": "poster",
        "style": {"color_palette": ["#0F172A", "#F8FAFC"]},
        "compositional_deconstruction": {
            "elements": [
                {
                    "type": "text",
                    "text": "SALE",
                    "bbox": [80, 100, 220, 900],
                },
                {
                    "type": "obj",
                    "desc": "product bottle",
                    "bbox": [300, 200, 900, 800],
                },
            ]
        },
    }
    spec = parse_box_layout(data)
    assert len(spec.regions) == 2
    headline = spec.regions[0]
    assert headline.y1 == 0.08
    assert headline.x1 == 0.10
    assert "SALE" in headline.prompt
    assert "#0F172A" in spec.global_prompt


def test_apply_design_brief_rewrites_prompt() -> None:
    args = SimpleNamespace(
        prompt='poster that says "SALE" in #112233',
        design_brief="",
        no_design_brief=False,
        palette="",
        glyph_canvas=False,
        no_glyph_canvas=False,
        _box_layout_spec=None,
    )
    apply_design_brief(args, image_size=64)
    assert "lettering reads exactly" in args.prompt
    assert args._palette_hexes
    assert "#112233" in args._palette_hexes


def test_palette_lock_pulls_toward_swatch() -> None:
    img = np.full((32, 32, 3), 90, dtype=np.uint8)
    img[..., 0] = 140
    img[..., 1] = 40
    img[..., 2] = 40
    out = apply_palette_lock(img, ["#FF0000"], strength=0.8)
    assert out[..., 0].mean() >= img[..., 0].mean() - 1
    scored_red = score_palette_lock(out, ["#FF0000"])
    scored_blue = score_palette_lock(out, ["#0000FF"])
    assert scored_red >= scored_blue - 1e-6


def test_design_finish_opt_out() -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    args = SimpleNamespace(no_palette_lock=True, palette="#FF0000", palette_lock=0.5, _palette_hexes=("#FF0000",))
    out = apply_design_finish(img, args, "#FF0000")
    assert np.array_equal(out, img)
