from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, List

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "instruction_adherence": 2.0,
    "layout_text_legibility": 1.5,
    "physical_plausibility": 1.25,
    "multi_object_distinctness": 1.25,
    "causal_coherence": 1.25,
    "style_control": 1.0,
    "novelty_without_drift": 0.75,
    "safety_alignment": 2.5,
}


@dataclass(slots=True)
class CapabilityDimensions:
    """Qualities that approximate 'general' image intelligence beyond CLIP-score."""

    instruction_adherence: float = 0.0
    layout_text_legibility: float = 0.0
    physical_plausibility: float = 0.0
    multi_object_distinctness: float = 0.0
    causal_coherence: float = 0.0
    style_control: float = 0.0
    novelty_without_drift: float = 0.0
    safety_alignment: float = 0.0


@dataclass(slots=True)
class CapabilityRubric:
    """Weights + thresholds for CapabilityDimensions."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    min_pass: float = 0.62

    def weighted_sum(self, d: CapabilityDimensions) -> float:
        ws = sum(float(self.weights[k]) * float(getattr(d, k)) for k in self.weights)
        zn = sum(self.weights.values())
        return ws / zn if zn else 0.0


def aggregate_stub(dimensions_list: List[CapabilityDimensions]) -> CapabilityDimensions:
    """Mean aggregate when multiple verifier heads exist."""
    if not dimensions_list:
        return CapabilityDimensions()
    n = float(len(dimensions_list))
    accum = [sum(getattr(x, f.name) for x in dimensions_list) / n for f in fields(CapabilityDimensions)]
    return CapabilityDimensions(*accum)


__all__ = ["CapabilityDimensions", "CapabilityRubric", "aggregate_stub"]
