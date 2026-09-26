"""Glyph / Ideogram-class inventions 49–58."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "OcrGlyphLoopPlan",
    "plan_ocr_glyph_loop",
    "FontIdPlan",
    "plan_font_id",
    "BrandKit",
    "load_brand_kit",
    "apply_brand_kit",
    "ui_screenshot_layout",
    "multilingual_glyph_addon",
    "bezier_text_spec",
    "vector_export_stub",
]


@dataclass
class OcrGlyphLoopPlan:
    texts: list[str] = field(default_factory=list)
    max_iters: int = 3
    argv_hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_ocr_glyph_loop(prompt: str, texts: list[str] | None = None) -> OcrGlyphLoopPlan:
    glyphs = list(texts or [])
    if not glyphs:
        glyphs = re.findall(r'"([^"]+)"', prompt) or re.findall(r"'([^']+)'", prompt)
    hints = ["--glyph-canvas", "--text-in-image", "1"]
    return OcrGlyphLoopPlan(
        texts=glyphs,
        argv_hints=hints,
        notes=["render→OCR→inpaint mismatched glyphs until match or max_iters"],
    )


@dataclass
class FontIdPlan:
    font_id: str = "sans_neutral"
    positive: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FONTS = {
    "sans_neutral": "clean grotesque sans-serif letterforms",
    "serif_editorial": "editorial serif display type",
    "mono_code": "monospaced coding font glyphs",
    "script_brush": "brush script lettering",
    "blackletter": "blackletter display type",
}


def plan_font_id(font_id: str = "sans_neutral") -> FontIdPlan:
    desc = _FONTS.get(font_id, font_id)
    return FontIdPlan(font_id=font_id, positive=f"typography style: {desc}, crisp glyph edges")


@dataclass
class BrandKit:
    name: str = ""
    palette_hex: list[str] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    logo_text: str = ""
    avoid: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_brand_kit(path_or_dict: str | Path | dict[str, Any]) -> BrandKit:
    if isinstance(path_or_dict, dict):
        d = path_or_dict
    else:
        d = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    return BrandKit(
        name=str(d.get("name") or ""),
        palette_hex=[str(x) for x in (d.get("palette") or d.get("palette_hex") or [])],
        fonts=[str(x) for x in (d.get("fonts") or [])],
        logo_text=str(d.get("logo_text") or d.get("wordmark") or ""),
        avoid=[str(x) for x in (d.get("avoid") or [])],
    )


def apply_brand_kit(prompt: str, kit: BrandKit) -> tuple[str, str, list[str]]:
    pos = str(prompt or "")
    neg = ", ".join(kit.avoid)
    argv: list[str] = []
    if kit.palette_hex:
        argv.extend(["--palette-lock", ",".join(kit.palette_hex)])
        pos = f"{pos}, brand palette {', '.join(kit.palette_hex)}"
    if kit.logo_text:
        pos = f'{pos}, wordmark "{kit.logo_text}"'
        argv.extend(["--text-in-image", "1", "--glyph-canvas"])
    if kit.fonts:
        pos = f"{pos}, {plan_font_id(kit.fonts[0]).positive}"
    return pos, neg, argv


def ui_screenshot_layout(*, title: str = "Settings", buttons: int = 3) -> dict[str, Any]:
    """UI generator mode (#57): grid of button regions + chrome."""
    regions = [{"id": "title", "prompt": f'UI title text "{title}"', "box": [0.1, 0.08, 0.9, 0.18]}]
    for i in range(max(1, buttons)):
        y0 = 0.25 + i * 0.18
        regions.append(
            {
                "id": f"btn_{i}",
                "prompt": f"rounded UI button {i + 1}",
                "box": [0.15, y0, 0.85, y0 + 0.12],
            }
        )
    return {
        "mode": "ui_screenshot",
        "global_prompt": "clean mobile UI screenshot, consistent spacing, modern design system",
        "regions": regions,
        "anti_bleed": True,
    }


def multilingual_glyph_addon(lang: str = "en") -> tuple[str, str]:
    table = {
        "en": ("correct English spelling", "misspelled English"),
        "ja": ("correct Japanese glyphs, coherent kana/kanji", "gibberish kana"),
        "zh": ("correct Chinese characters", "nonsensical hanzi salad"),
        "ko": ("correct Hangul syllables", "broken Hangul jamo mash"),
        "ar": ("correct Arabic script, right-to-left", "mirrored broken Arabic"),
    }
    return table.get(lang, table["en"])


def bezier_text_spec(text: str, curve: str = "arc") -> dict[str, Any]:
    """Moonshot scaffold (#51): text on Bezier path metadata for a future conditioner."""
    return {"text": text, "curve": curve, "control_points": [[0.1, 0.5], [0.5, 0.2], [0.9, 0.5]]}


def vector_export_stub(latent_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Moonshot scaffold (#58): placeholder SVG export pipeline descriptor."""
    return {
        "format": "svg",
        "status": "stub",
        "steps": ["edge detect", "potrace-like vectorize", "merge glyph paths"],
        "meta": latent_meta or {},
    }
