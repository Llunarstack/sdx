"""Layer stack schema — regenerate one layer without full rerender."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from .style_engines import engine_by_id

__all__ = ["LayerSpec", "LayerStack", "parse_layer_stack", "stack_prompt_suffix"]


@dataclass(slots=True)
class LayerSpec:
    name: str
    engine: str = "realistic"
    prompt: str = ""
    mask: str = ""
    z_order: int = 0


@dataclass(slots=True)
class LayerStack:
    layers: List[LayerSpec] = field(default_factory=list)

    def engines_by_layer(self) -> Dict[str, str]:
        return {layer.name: layer.engine for layer in self.layers}


def parse_layer_stack(raw: Any) -> LayerStack:
    stack = LayerStack()
    if not raw:
        return stack
    rows = raw if isinstance(raw, list) else (raw.get("layers") if isinstance(raw, Mapping) else [])
    if not isinstance(rows, list):
        return stack
    for i, row in enumerate(rows):
        if isinstance(row, str):
            stack.layers.append(LayerSpec(name=row, z_order=i))
            continue
        if not isinstance(row, Mapping):
            continue
        stack.layers.append(
            LayerSpec(
                name=str(row.get("name") or row.get("id") or f"layer_{i}"),
                engine=str(row.get("engine") or row.get("style") or "realistic"),
                prompt=str(row.get("prompt") or ""),
                mask=str(row.get("mask") or ""),
                z_order=int(row.get("z") or row.get("z_order") or i),
            )
        )
    stack.layers.sort(key=lambda layer: layer.z_order)
    return stack


def stack_prompt_suffix(stack: LayerStack) -> str:
    parts = []
    for layer in stack.layers:
        if layer.prompt:
            parts.append(f"[layer {layer.name}: {layer.prompt}]")
        elif layer.engine:
            preset = engine_by_id(layer.engine)
            if preset:
                parts.append(f"[layer {layer.name}: {preset.positive}]")
    return " ".join(parts)
