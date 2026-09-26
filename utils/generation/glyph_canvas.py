"""Rasterize quoted / [text:] strings into a high-contrast control canvas."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from utils.generation.text_rendering import TextRenderingEngine

__all__ = ["extract_glyph_strings", "render_glyph_canvas"]

_ENGINE = TextRenderingEngine()

_FONT_CANDIDATES = (
    "arial.ttf",
    "DejaVuSans.ttf",
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/Arial.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            try:
                return ImageFont.truetype(str(Path(candidate)), size)
            except OSError:
                continue
    return ImageFont.load_default()


def extract_glyph_strings(prompt: str) -> list[str]:
    """Return non-empty quoted / [text:] strings from *prompt*."""
    info = _ENGINE.extract_text_requirements(prompt or "")
    return [s.strip() for s in info.get("text_content", []) if str(s).strip()]


def _wrap_lines(text: str, *, max_chars: int = 12) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(text) <= max_chars and " " not in text:
        return [text]
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) if not current else len(word) + 1
        if current and current_len + extra > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def _placement_box(size: int, typography_type: str) -> tuple[int, int, int, int]:
    margin = int(size * 0.10)
    if typography_type in {"sign", "shop", "storefront"}:
        y0 = margin
        y1 = int(size * 0.38)
    elif typography_type in {"poster", "banner"}:
        y0 = int(size * 0.32)
        y1 = int(size * 0.68)
    else:
        inset = int(size * 0.20)
        y0 = inset
        y1 = size - inset
    return margin, y0, size - margin, y1


def _fit_font(lines: list[str], box_w: int, box_h: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    target_w = max(1, int(box_w * 0.70))
    lo, hi = 8, max(12, box_h)
    best = _load_font(lo)
    probe = ImageDraw.Draw(Image.new("RGB", (box_w, box_h)))
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(mid)
        widths = [probe.textbbox((0, 0), line, font=font)[2] for line in lines]
        line_h = probe.textbbox((0, 0), "Ay", font=font)[3]
        total_h = line_h * max(1, len(lines)) + max(0, len(lines) - 1) * int(line_h * 0.15)
        if max(widths) <= target_w and total_h <= box_h:
            best = font
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_glyph_canvas(
    prompt: str,
    size: int = 512,
    placements: list[dict] | None = None,
) -> Image.Image | None:
    """High-contrast RGB canvas of requested lettering, or None if no text.

    ``placements`` is an optional list of ``{text, x1, y1, x2, y2}`` in 0–1
    (Ideogram-style typed text boxes). Without it, quoted strings share one band.
    """
    placed: list[tuple[str, tuple[int, int, int, int]]] = []
    if placements:
        for item in placements:
            chunk = str(item.get("text", "") or "").strip()
            if not chunk:
                continue
            x1 = int(max(0.0, min(1.0, float(item.get("x1", 0.08)))) * size)
            y1 = int(max(0.0, min(1.0, float(item.get("y1", 0.32)))) * size)
            x2 = int(max(0.0, min(1.0, float(item.get("x2", 0.92)))) * size)
            y2 = int(max(0.0, min(1.0, float(item.get("y2", 0.68)))) * size)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            placed.append((chunk, (x1, y1, x2, y2)))
    if not placed:
        strings = extract_glyph_strings(prompt)
        if not strings:
            return None
        info = _ENGINE.extract_text_requirements(prompt or "")
        typography_type = str(info.get("typography_type", "general") or "general")
        if "shop" in (prompt or "").lower() or "storefront" in (prompt or "").lower():
            typography_type = "sign"
        box = _placement_box(size, typography_type)
        # One stacked block of all quoted strings (legacy behaviour).
        placed.append(("\n".join(strings), box))

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for chunk, (x0, y0, x1, y1) in placed:
        lines = []
        for part in str(chunk).split("\n"):
            lines.extend(_wrap_lines(part) or [part])
        if not lines:
            continue
        box_w = max(1, x1 - x0)
        box_h = max(1, y1 - y0)
        font = _fit_font(lines, box_w, box_h)
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
        spacing = int(max(line_heights) * 0.15) if line_heights else 0
        total_h = sum(line_heights) + spacing * max(0, len(lines) - 1)
        y = y0 + max(0, (box_h - total_h) // 2)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]
            x = x0 + max(0, (box_w - line_w) // 2)
            draw.text((x, y), line, fill=(0, 0, 0), font=font)
            y += line_h + spacing
    return canvas
