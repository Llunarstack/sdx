"""Semantic Gravity — narrative weight ranks entities for identity lock and drift repair priority."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "GravityWell",
    "SemanticGravityField",
    "parse_semantic_gravity",
    "build_gravity_field",
    "gravity_edit_overrides",
]


@dataclass(slots=True)
class GravityWell:
    entity_id: str
    weight: float  # 0..1
    lock_priority: int
    drift_resist: float


@dataclass(slots=True)
class SemanticGravityField:
    wells: List[GravityWell] = field(default_factory=list)
    dominant_id: str = ""

    def top_entities(self, n: int = 3) -> List[str]:
        ranked = sorted(self.wells, key=lambda w: (-w.weight, -w.lock_priority))
        return [w.entity_id for w in ranked[:n]]


def parse_semantic_gravity(raw: Any) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if isinstance(raw, Mapping):
        for eid, spec in raw.items():
            if isinstance(spec, (int, float)):
                weights[str(eid)] = float(spec)
            elif isinstance(spec, Mapping):
                weights[str(eid)] = float(spec.get("weight") or spec.get("gravity") or 0.5)
    return weights


def build_gravity_field(
    cast: Mapping[str, Any],
    props: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    shots: Sequence[Any],
) -> SemanticGravityField:
    appearance_count: Dict[str, int] = {}
    for sh in shots:
        for eid in list(getattr(sh, "characters", []) or []) + list(getattr(sh, "objects", []) or []):
            appearance_count[str(eid)] = appearance_count.get(str(eid), 0) + 1

    wells: List[GravityWell] = []
    all_ids = set(cast.keys()) | set(props.keys()) | set(weights.keys())
    for eid in all_ids:
        base = float(weights.get(eid, 0.4))
        ent = cast.get(eid) or props.get(eid)
        if ent is not None and getattr(ent, "lock", False):
            base = max(base, 0.85)
        freq_boost = min(0.3, appearance_count.get(eid, 0) * 0.08)
        w = min(1.0, base + freq_boost)
        wells.append(
            GravityWell(
                entity_id=eid,
                weight=round(w, 3),
                lock_priority=int(w * 100),
                drift_resist=round(0.5 + w * 0.45, 3),
            )
        )
    ranked = sorted(wells, key=lambda x: -x.weight)
    dominant = ranked[0].entity_id if ranked else ""
    return SemanticGravityField(wells=wells, dominant_id=dominant)


def gravity_edit_overrides(field: SemanticGravityField) -> Dict[str, Any]:
    if not field.wells:
        return {}
    top = field.top_entities(2)
    strength = 0.78
    for w in field.wells:
        if w.entity_id in top:
            strength = max(strength, w.drift_resist)
    return {
        "identity_lock": True,
        "identity_lock_strength": min(0.96, strength),
        "semantic_drift_repair": True,
        "drift_threshold": max(0.35, 0.6 - (field.wells[0].weight if field.wells else 0) * 0.2),
        "gravity_priority_entities": top,
    }
