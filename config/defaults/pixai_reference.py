# Reference: [PixAI.art](https://pixai.art/en/generator/image) model lineup.
# Our architectures (DiT-XL, DiT-P, etc.) take inspiration from their XL and DiT families
# and named model lines. Use these for presets, docs, or LoRA/naming consistency.

from typing import List, Tuple

# PixAI.art-style model lines (name, family, approximate scale if known from site).
# Family: "XL" = XL-scale backbone, "DiT" = Diffusion Transformer scale.
PIXAI_MODEL_LINES: List[Tuple[str, str, str]] = [
    ("Haruka v2", "PixAI XL", "154.44m"),
    ("Tsubaki.2", "PixAI DiT", "2502.27k"),
    ("Tsubaki", "PixAI DiT", "120.44m"),
    ("Hoshino v2", "PixAI XL", "7.77m"),
    ("Hoshino", "PixAI XL", "51.92m"),
    ("Tsubaki v1.1", "PixAI DiT", "14.95m"),
    ("Nagi", "PixAI XL", "343.59k"),
    ("Crystalize", "PixAI XL", "453.01k"),
    ("Eternal", "PixAI XL", "180.79k"),
    ("Otome v2", "PixAI XL", "7.59m"),
    ("Tsubaki Flash", "PixAI DiT", "1556.60k"),
    ("Hinata v2", "PixAI XL", "2.64m"),
    ("Haruka", "PixAI XL", "34.17m"),
    ("Serin", "PixAI DiT", "13.11m"),
]

# Map our --model names to a PixAI.art-inspired style label (for logs/docs).
# Use when you want to show "Tsubaki-style" or "XL-style" in UI or logs.
SDX_MODEL_TO_PIXAI_STYLE = {
    "DiT-XL/2-Text": "PixAI DiT-style (Tsubaki / Serin family)",
    "DiT-L/2-Text": "PixAI DiT-style smaller",
    "DiT-B/2-Text": "PixAI DiT-style base",
    "DiT-P/2-Text": "PixAI DiT-style large (Tsubaki.2-style)",
    "DiT-P-L/2-Text": "PixAI DiT-style XL with QK-norm/SwiGLU",
}


def get_pixai_style_label(model_name: str) -> str:
    """Return a short label for the given model name, inspired by PixAI.art lineup."""
    return SDX_MODEL_TO_PIXAI_STYLE.get(model_name, "PixAI.art-style")


def list_pixai_model_lines() -> List[Tuple[str, str, str]]:
    """Return the reference list of PixAI.art model lines (name, family, scale)."""
    return list(PIXAI_MODEL_LINES)
