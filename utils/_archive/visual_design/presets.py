"""Named shortcuts: domain + intensity (+ optional prompt prefix)."""

from __future__ import annotations

from functools import cache
from types import MappingProxyType
from typing import Dict, Optional, Tuple

from .sampling import normalize_intensity

_VISUAL_DESIGN_PRESETS_DICT: Dict[str, Tuple[str, str, str]] = {
    "saas_ui": ("ui_ux", "strong", ""),
    "arch_hero": ("architecture", "standard", ""),
    "stem_exam_figure": ("stem", "strong", ""),
    "textbook_spread": ("textbook", "standard", ""),
    "logo_lockup": ("brand", "standard", ""),
    "annual_report": ("editorial_layout", "strong", ""),
    "keynote_slide": ("presentation_slide", "standard", ""),
    "patent_drawing": ("technical_blueprint", "strong", ""),
    "tech_pack_flat": ("fashion_flat", "standard", ""),
}

# Read-only mapping: no accidental mutation at runtime.
VISUAL_DESIGN_PRESETS: MappingProxyType[str, Tuple[str, str, str]] = MappingProxyType(_VISUAL_DESIGN_PRESETS_DICT)


@cache
def preset_ids() -> Tuple[str, ...]:
    return tuple(sorted(_VISUAL_DESIGN_PRESETS_DICT.keys()))


def resolve_visual_design_preset(name: str) -> Optional[Tuple[str, str, str]]:
    n = (name or "").strip().lower()
    if not n:
        return None
    return _VISUAL_DESIGN_PRESETS_DICT.get(n)


def apply_visual_design_preset_to_prompt(prompt: str, preset: str) -> Tuple[str, str, str]:
    base = (prompt or "").strip()
    row = resolve_visual_design_preset(preset)
    if row is None:
        return base, "none", "standard"
    domain, int_raw, prefix = row
    intensity = normalize_intensity(int_raw)
    pst = prefix.strip()
    if pst:
        merged = f"{pst}, {base}".strip().strip(",") if base else pst
    else:
        merged = base
    return merged, domain, intensity


__all__ = [
    "VISUAL_DESIGN_PRESETS",
    "preset_ids",
    "resolve_visual_design_preset",
    "apply_visual_design_preset_to_prompt",
]
