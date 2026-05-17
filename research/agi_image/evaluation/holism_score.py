from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HolismComponents:
    """
    Separation of aesthetic vs semantic vs structural quality.

    Feeds CapabilityDimensions or external pick-best metrics.
    """

    perceptual_quality: float = 0.0
    semantic_coverage: float = 0.0
    spatial_structure: float = 0.0
    interpersonal_consistency: float = 0.0


def summarise_holism(h: HolismComponents) -> float:
    parts = (
        h.perceptual_quality,
        h.semantic_coverage,
        h.spatial_structure,
        h.interpersonal_consistency,
    )
    return sum(parts) * 0.25


__all__ = ["HolismComponents", "summarise_holism"]
