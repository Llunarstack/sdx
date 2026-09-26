"""Layout-first design brief — Reve / Ideogram 4 / GPT Image 2 ideas we lacked.

Closed APIs that win on posters, packaging, and multi-object scenes do three
things SDX used to leave to the raw T5 string:

1. **Plan then render** (Reve, GPT Image 2 thinking): pin exact lettering,
   counts, and spatial boxes *before* denoising.
2. **Typed text vs object elements with bboxes** (Ideogram 4): text is a
   first-class region, not a visual decoration the model might misspell.
3. **Hex palettes as a non-language channel** (Ideogram / Recraft): brand
   colours are listed as ``#RRGGBB``, not hoped-for adjectives.

This module compiles that structure from JSON or from a natural-language
prompt and hands it to the existing regional-CFG / glyph-canvas stack.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompt_scene import compile_spatial_layout, quoted_glyph_texts
from .regional_box_prompting import BoxLayoutSpec, BoxRegion

__all__ = [
    "DesignElement",
    "DesignBrief",
    "extract_hex_palette",
    "compile_design_brief",
    "load_design_brief_file",
    "brief_to_caption",
    "brief_to_box_spec",
    "glyph_placements_from_brief",
]

_HEX_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
_RGB_RE = re.compile(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", re.I)
_JSON_HINT = re.compile(r"^\s*[\{\[]", re.S)


@dataclass(slots=True)
class DesignElement:
    kind: str  # "text" | "obj"
    name: str
    prompt: str
    x1: float
    y1: float
    x2: float
    y2: float
    content: str = ""
    negative: str = ""
    priority: int = 6


@dataclass(slots=True)
class DesignBrief:
    description: str = ""
    palette: tuple[str, ...] = ()
    elements: tuple[DesignElement, ...] = ()
    source: str = "nl"  # nl | json | file

    def has_pins(self) -> bool:
        return bool(self.palette or self.elements or (self.description and self.source != "nl"))


def _expand_short_hex(token: str) -> str:
    t = token.strip()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", t):
        return "#" + "".join(ch * 2 for ch in t[1:])
    return t.upper() if t.startswith("#") else t


def extract_hex_palette(*chunks: str, extra: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Collect up to 16 unique ``#RRGGBB`` colours from text / rgb() / extras."""
    found: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        hx = _expand_short_hex(token)
        if not re.fullmatch(r"#[0-9A-F]{6}", hx):
            return
        if hx in seen:
            return
        seen.add(hx)
        found.append(hx)

    for chunk in chunks:
        if not chunk:
            continue
        for m in _HEX_RE.finditer(chunk):
            _add(m.group(0))
        for m in _RGB_RE.finditer(chunk):
            r, g, b = (max(0, min(255, int(m.group(i)))) for i in (1, 2, 3))
            _add(f"#{r:02X}{g:02X}{b:02X}")
    if extra:
        for token in extra:
            if not str(token).strip():
                continue
            t = str(token).strip()
            if t.startswith("#") or _HEX_RE.fullmatch(t):
                _add(t if t.startswith("#") else f"#{t}")
            else:
                _add(f"#{t}" if re.fullmatch(r"[0-9a-fA-F]{3,6}", t) else t)
    return tuple(found[:16])


def _is_json_brief(text: str) -> bool:
    s = (text or "").strip()
    if not s or not _JSON_HINT.match(s):
        return False
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and bool(
        obj.get("compositional_deconstruction") or obj.get("elements") or obj.get("color_palette") or obj.get("regions")
    )


def _element_from_box_region(region: BoxRegion, *, kind: str = "obj") -> DesignElement:
    is_text = kind == "text" or region.name.startswith("glyph") or region.name.startswith("text")
    content = ""
    if is_text:
        quotes = quoted_glyph_texts(region.prompt)
        content = quotes[0] if quotes else region.prompt
    return DesignElement(
        kind="text" if is_text else "obj",
        name=region.name,
        prompt=region.prompt,
        x1=region.x1,
        y1=region.y1,
        x2=region.x2,
        y2=region.y2,
        content=content,
        negative=region.negative,
        priority=region.priority,
    )


def _brief_from_mapping(data: Mapping[str, Any], *, source: str = "json") -> DesignBrief:
    from .regional_box_prompting import parse_box_layout

    style = data.get("style") if isinstance(data.get("style"), dict) else {}
    palette = extract_hex_palette(
        json.dumps(data, ensure_ascii=False) if not isinstance(data.get("color_palette"), list) else "",
        extra=list(data.get("color_palette") or []) + list((style or {}).get("color_palette") or []),
    )
    description = str(data.get("description", data.get("global_prompt", data.get("prompt", ""))) or "").strip()
    elements: list[DesignElement] = []
    try:
        spec = parse_box_layout(data)
    except (TypeError, ValueError):
        spec = None
    if spec is not None:
        if not description:
            description = spec.global_prompt
        for region in spec.regions:
            kind = "text" if region.name.startswith(("text", "glyph", "letter")) else "obj"
            elements.append(_element_from_box_region(region, kind=kind))
        palette = palette or extract_hex_palette(spec.global_prompt)
    return DesignBrief(description=description, palette=palette, elements=tuple(elements), source=source)


def load_design_brief_file(path: str | Path) -> DesignBrief:
    raw = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(raw, dict):
        raise ValueError("design brief JSON must be an object")
    return _brief_from_mapping(raw, source="file")


def _default_text_box(index: int, n: int) -> tuple[float, float, float, float]:
    """Stack headline → subtitle → body in the upper/center band."""
    if n <= 1:
        return 0.08, 0.32, 0.92, 0.68
    band = 0.50 / max(n, 1)
    gap = 0.03
    y1 = 0.18 + index * (band + gap)
    y2 = min(0.82, y1 + band)
    return 0.08, y1, 0.92, y2


def compile_design_brief(
    prompt: str,
    *,
    palette_extra: list[str] | tuple[str, ...] | None = None,
    json_path: str = "",
) -> DesignBrief:
    """JSON file > inline JSON prompt > natural-language compile."""
    if json_path:
        brief = load_design_brief_file(json_path)
        extra = extract_hex_palette(prompt or "", extra=palette_extra)
        if extra:
            merged = tuple(dict.fromkeys(list(brief.palette) + list(extra)))[:16]
            brief = DesignBrief(
                description=brief.description,
                palette=merged,
                elements=brief.elements,
                source=brief.source,
            )
        return brief
    if _is_json_brief(prompt or ""):
        return _brief_from_mapping(json.loads(prompt), source="json")

    palette = extract_hex_palette(prompt or "", extra=palette_extra)
    texts = quoted_glyph_texts(prompt or "")
    spec = compile_spatial_layout(prompt or "")
    elements: list[DesignElement] = []
    used_text_names: set[str] = set()
    if spec is not None:
        for region in spec.regions:
            el = _element_from_box_region(region)
            elements.append(el)
            if el.kind == "text":
                used_text_names.add(el.content)
    leftover = [t for t in texts if t not in used_text_names]
    if leftover:
        n = len(leftover)
        for i, chunk in enumerate(leftover):
            x1, y1, x2, y2 = _default_text_box(i, n)
            elements.append(
                DesignElement(
                    kind="text",
                    name=f"text_{i + 1}",
                    prompt=f'lettering reads exactly: "{chunk}"',
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    content=chunk,
                    negative="misspelled text, garbled letters, extra words",
                    priority=12,
                )
            )
    return DesignBrief(description=(prompt or "").strip(), palette=palette, elements=tuple(elements), source="nl")


def brief_to_caption(brief: DesignBrief, prompt: str) -> str:
    """GPT-Image-2-style pin: exact strings + hex palette on the T5 string."""
    base = (brief.description or prompt or "").strip()
    pins: list[str] = []
    texts = [el.content.strip() for el in brief.elements if el.kind == "text" and el.content.strip()]
    for chunk in dict.fromkeys(texts):
        already = f'lettering reads exactly: "{chunk}"'
        if already.lower() not in base.lower():
            pins.append(already)
    if brief.palette:
        joined = ", ".join(brief.palette)
        if joined.lower() not in base.lower():
            pins.append(f"color palette locked to {joined}")
    if not pins:
        return base
    return f"{base.rstrip('.')}. {'. '.join(pins)}."


def brief_to_box_spec(brief: DesignBrief, *, global_negative: str = "") -> BoxLayoutSpec | None:
    boxed = [el for el in brief.elements if (el.x2 - el.x1) > 0.04 and (el.y2 - el.y1) > 0.04]
    if not boxed:
        return None
    regions = [
        BoxRegion(
            name=el.name,
            x1=el.x1,
            y1=el.y1,
            x2=el.x2,
            y2=el.y2,
            prompt=el.prompt or el.content or el.kind,
            negative=el.negative,
            priority=el.priority,
        )
        for el in boxed
    ]
    return BoxLayoutSpec(global_prompt=brief.description, global_negative=global_negative, regions=regions)


def glyph_placements_from_brief(brief: DesignBrief) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for el in brief.elements:
        if el.kind != "text" or not el.content.strip():
            continue
        out.append(
            {
                "text": el.content,
                "x1": el.x1,
                "y1": el.y1,
                "x2": el.x2,
                "y2": el.y2,
            }
        )
    return out
