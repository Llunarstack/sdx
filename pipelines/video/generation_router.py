"""Generation router — pick the right engine for each scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .style_engines import EnginePreset, RenderEngine, engine_by_id, engine_edit_overrides, match_engine_from_prompt

__all__ = ["RouteDecision", "route_scene"]


@dataclass(slots=True)
class RouteDecision:
    engine: RenderEngine
    preset: EnginePreset
    confidence: float
    reasons: List[str] = field(default_factory=list)
    edit_overrides: Dict[str, Any] = field(default_factory=dict)
    style_positive: str = ""
    style_negative: str = ""
    retrieval_tags: List[str] = field(default_factory=list)


def route_scene(
    prompt: str,
    *,
    style_hint: str = "",
    engine_override: str = "",
    layer_stack: Optional[Mapping[str, str]] = None,
) -> RouteDecision:
    """Analyze prompt + hints → engine + edit overrides."""
    reasons: List[str] = []
    if engine_override and engine_override.lower() not in ("auto", ""):
        preset = engine_by_id(engine_override)
        if preset:
            reasons.append(f"explicit engine={preset.id.value}")
            return RouteDecision(
                engine=preset.id,
                preset=preset,
                confidence=1.0,
                reasons=reasons,
                edit_overrides=engine_edit_overrides(preset),
                style_positive=preset.positive,
                style_negative=preset.negative,
                retrieval_tags=list(preset.retrieval_tags),
            )

    if layer_stack and len(layer_stack) > 1:
        eng = RenderEngine.HYBRID
        preset = engine_by_id("hybrid")
        reasons.append("multi-layer stack → hybrid router")
    else:
        eng = match_engine_from_prompt(prompt, style_hint=style_hint)
        preset = engine_by_id(eng.value)
        reasons.append(f"prompt/style match → {eng.value}")

    assert preset is not None
    return RouteDecision(
        engine=eng,
        preset=preset,
        confidence=0.75,
        reasons=reasons,
        edit_overrides=engine_edit_overrides(preset),
        style_positive=preset.positive,
        style_negative=preset.negative,
        retrieval_tags=list(preset.retrieval_tags),
    )
