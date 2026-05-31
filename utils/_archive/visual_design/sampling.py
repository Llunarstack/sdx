"""Single entry for applying visual-design packs during sampling (mirrors ``sample.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from .compose import build_visual_design_prompt_pair, prompt_suggests_domain

Intensity = Literal["lite", "standard", "strong"]

_VALID_INTENSITY: frozenset[str] = frozenset({"lite", "standard", "strong"})


@dataclass(frozen=True, slots=True)
class VisualDesignApplyResult:
    prompt: str
    negative_addon: str
    resolved_domain: str


def normalize_intensity(raw: object) -> Intensity:
    s = str(raw or "standard").strip().lower()
    return s if s in _VALID_INTENSITY else "standard"  # type: ignore[return-value]


EmitFn = Optional[Callable[[str], None]]


def apply_visual_design_stage(
    prompt: str,
    *,
    cli_domain: str,
    intensity: str = "standard",
    use_negative_pack: bool,
    emit: EmitFn = None,
) -> VisualDesignApplyResult:
    base = (prompt or "").strip()
    dom = str(cli_domain or "none").strip().lower()

    if dom in {"", "none"} or not base:
        return VisualDesignApplyResult(prompt=base, negative_addon="", resolved_domain="none")

    tier = normalize_intensity(intensity)
    resolved = dom
    if resolved == "auto":
        guess = prompt_suggests_domain(base)
        if guess is None:
            if emit:
                emit("Note: visual-design-domain=auto did not match a domain; skipping pack.")
            return VisualDesignApplyResult(prompt=base, negative_addon="", resolved_domain="none")
        resolved = str(guess)
        if emit:
            emit(f"Visual design: auto-selected domain {resolved}")

    try:
        pos, neg = build_visual_design_prompt_pair(base, resolved, intensity=tier, dedupe_positive=True)
    except ValueError:
        if emit:
            emit(f"Visual design pack skipped: unknown domain {resolved!r}")
        return VisualDesignApplyResult(prompt=base, negative_addon="", resolved_domain="none")

    if use_negative_pack and neg:
        nstrip = neg.strip()
        neg_out = nstrip if nstrip else ""
    else:
        neg_out = ""

    return VisualDesignApplyResult(prompt=pos, negative_addon=neg_out, resolved_domain=resolved)


__all__ = ["VisualDesignApplyResult", "apply_visual_design_stage", "normalize_intensity"]
