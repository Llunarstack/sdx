from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class StoryBeat:
    """Panels / timestamps / episodic cues for coherent sequences."""

    index: int
    summary: str
    invariant_entity_ids: List[str] = field(default_factory=list)
    allowed_visual_drift: str = "low"  # low | medium | high
    continuity_notes: Dict[str, str] = field(default_factory=dict)
    prior_beat_hint: Optional[str] = None


__all__ = ["StoryBeat"]
