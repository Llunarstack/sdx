"""Still-image competitor gaps: auto layout, glyph sketches, prompt-ground, contact.

These are the 2026 failure modes closed APIs still share (spatial relations,
readable glyphs, attribute binding, floating feet) and that our DiT sampler
did not run unless the user shipped a JSON box layout.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "apply_prompt_ground",
    "apply_auto_layout",
    "apply_design_brief",
    "apply_contact_post",
    "apply_design_finish",
]


def apply_prompt_ground(args: Any) -> None:
    """Bind attributes/counts/negation in the prompt when 2+ subjects or a negation."""
    if bool(getattr(args, "no_prompt_ground", False)):
        return
    if not bool(getattr(args, "prompt_ground", False)):
        return
    prompt = str(getattr(args, "prompt", "") or "")
    if not prompt.strip():
        return
    try:
        from pipelines.video.prompt_ground_graph import parse_prompt_ground
    except Exception:
        return
    graph = parse_prompt_ground(prompt)
    if len(graph.entities) < 2 and not graph.negations:
        return
    args.prompt = graph.rewritten or prompt
    extra = str(graph.negative_extra or "").strip()
    if extra:
        cur = str(getattr(args, "negative_prompt", "") or "")
        if extra.lower() not in cur.lower():
            args.negative_prompt = f"{cur}, {extra}".strip(", ").strip()
    print(
        f"Prompt ground: {len(graph.entities)} entit(y/ies), negations={len(graph.negations)}.",
        file=sys.stderr,
    )


def apply_auto_layout(args: Any, *, image_size: int = 512) -> None:
    """Compile a regional box spec from the prompt when the user did not pass JSON."""
    if getattr(args, "_box_layout_spec", None) is not None:
        return
    if bool(getattr(args, "no_auto_layout", False)):
        return
    if not bool(getattr(args, "auto_layout", False)):
        return
    prompt = str(getattr(args, "prompt", "") or "")
    from utils.generation.prompt_scene import compile_spatial_layout

    spec = compile_spatial_layout(prompt)
    if spec is None:
        return
    want_glyph = bool(getattr(args, "glyph_canvas", False)) and not bool(getattr(args, "no_glyph_canvas", False))
    if want_glyph:
        try:
            from utils.generation.glyph_canvas import render_glyph_canvas

            canvas = render_glyph_canvas(prompt, size=max(64, int(image_size)))
        except Exception as e:
            print(f"Glyph canvas skipped: {e}", file=sys.stderr)
            canvas = None
        if canvas is not None:
            path = Path(tempfile.gettempdir()) / "sdx_glyph_canvas.png"
            canvas.save(path)
            attached = False
            for region in spec.regions:
                if region.name == "glyph_text":
                    region.sketch_path = str(path)
                    attached = True
                    break
            if not attached and spec.regions:
                spec.regions[0].sketch_path = str(path)
    args._box_layout_spec = spec
    gn = str(getattr(spec, "global_negative", "") or "").strip()
    if gn:
        cur = str(getattr(args, "negative_prompt", "") or "")
        if gn.lower() not in cur.lower():
            args.negative_prompt = f"{cur}, {gn}".strip(", ").strip()
    print(
        f"Auto layout: {len(spec.regions)} region(s) ({', '.join(r.name for r in spec.regions)}).",
        file=sys.stderr,
    )


def _parse_palette_flag(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    return [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]


def apply_design_brief(args: Any, *, image_size: int = 512) -> None:
    """Reve / Ideogram / GPT-Image-2 plan: pin lettering, boxes, and hex palette."""
    if bool(getattr(args, "no_design_brief", False)):
        return
    prompt = str(getattr(args, "prompt", "") or "")
    json_path = str(getattr(args, "design_brief", "") or "").strip()
    extra = _parse_palette_flag(getattr(args, "palette", None))
    if not prompt.strip() and not json_path and not extra:
        return
    try:
        from utils.generation.design_brief import (
            brief_to_box_spec,
            brief_to_caption,
            compile_design_brief,
            glyph_placements_from_brief,
        )
    except Exception:
        return
    brief = compile_design_brief(prompt, palette_extra=extra, json_path=json_path)
    text_els = [el for el in brief.elements if el.kind == "text"]
    if not brief.palette and not text_els and brief.source == "nl":
        # Spatial-only NL already handled by auto_layout; nothing extra to pin.
        if extra:
            args._palette_hexes = tuple(extra)
        return
    args._design_brief = brief
    if brief.palette or extra:
        args._palette_hexes = tuple(dict.fromkeys(list(brief.palette) + extra))
    pinned = brief_to_caption(brief, prompt)
    if pinned and pinned != prompt:
        args.prompt = pinned
    spec = getattr(args, "_box_layout_spec", None)
    compiled = brief_to_box_spec(brief)
    if spec is None and compiled is not None:
        args._box_layout_spec = compiled
        spec = compiled
    elif spec is not None and compiled is not None:
        have = {r.name for r in spec.regions}
        for region in compiled.regions:
            if region.name not in have:
                spec.regions.append(region)
    placements = glyph_placements_from_brief(brief)
    want_glyph = bool(getattr(args, "glyph_canvas", False)) and not bool(getattr(args, "no_glyph_canvas", False))
    if want_glyph and placements and spec is not None:
        try:
            from utils.generation.glyph_canvas import render_glyph_canvas

            canvas = render_glyph_canvas(args.prompt, size=max(64, int(image_size)), placements=placements)
        except Exception as e:
            print(f"Glyph canvas (brief) skipped: {e}", file=sys.stderr)
            canvas = None
        if canvas is not None:
            path = Path(tempfile.gettempdir()) / "sdx_glyph_canvas.png"
            canvas.save(path)
            attached = False
            for region in spec.regions:
                if region.name.startswith("text") or region.name == "glyph_text":
                    region.sketch_path = str(path)
                    attached = True
                    break
            if not attached and spec.regions:
                spec.regions[0].sketch_path = str(path)
    bits = []
    if brief.palette:
        bits.append(f"{len(brief.palette)} swatch(es)")
    if text_els:
        bits.append(f"{len(text_els)} text region(s)")
    if bits:
        print(f"Design brief: {', '.join(bits)}.", file=sys.stderr)


def apply_contact_post(img_np: np.ndarray, args: Any, prompt: str) -> np.ndarray:
    """Optional anti-float contact shadow after RGB finishing."""
    strength = float(getattr(args, "contact_shadow", -1.0) if hasattr(args, "contact_shadow") else -1.0)
    if bool(getattr(args, "no_contact_shadow", False)):
        return img_np
    from utils.quality.contact_shadow import apply_contact_shadow, prompt_wants_ground_contact

    if strength == 0.0:
        return img_np
    if strength < 0.0:
        if not bool(getattr(args, "contact_shadow_auto", False)):
            return img_np
        if not prompt_wants_ground_contact(prompt):
            return img_np
        strength = 0.4
    try:
        return apply_contact_shadow(img_np, strength=float(strength))
    except Exception as e:
        print(f"Contact shadow skipped: {e}", file=sys.stderr)
        return img_np


def apply_design_finish(img_np: np.ndarray, args: Any, prompt: str) -> np.ndarray:
    """Soft hex-palette grade after RGB finishing (Recraft / Ideogram colour lock)."""
    if bool(getattr(args, "no_palette_lock", False)):
        return img_np
    hexes = list(getattr(args, "_palette_hexes", ()) or ())
    if not hexes:
        hexes = _parse_palette_flag(getattr(args, "palette", None))
    strength = float(getattr(args, "palette_lock", -1.0) if hasattr(args, "palette_lock") else -1.0)
    if strength == 0.0:
        return img_np
    from utils.generation.design_brief import extract_hex_palette

    if not hexes:
        hexes = list(extract_hex_palette(prompt or ""))
    if not hexes:
        return img_np
    if strength < 0.0:
        strength = 0.22
    try:
        from utils.quality.palette_lock import apply_palette_lock

        return apply_palette_lock(img_np, hexes, strength=float(strength), prompt=prompt or "")
    except Exception as e:
        print(f"Palette lock skipped: {e}", file=sys.stderr)
        return img_np
