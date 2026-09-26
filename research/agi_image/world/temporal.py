from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StoryBeat:
    """Panels / timestamps / episodic cues for coherent sequences."""

    index: int
    summary: str
    invariant_entity_ids: list[str] = field(default_factory=list)
    allowed_visual_drift: str = "low"  # low | medium | high
    continuity_notes: dict[str, str] = field(default_factory=dict)
    prior_beat_hint: str | None = None


__all__ = ["StoryBeat"]
