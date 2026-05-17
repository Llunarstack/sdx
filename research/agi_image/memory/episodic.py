from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class EpisodicSlot:
    """One saved render + minimal structured recall."""

    step_id: str
    thumbnail_ref: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    latent_digest: Optional[str] = None
    commentary: str = ""


@dataclass(slots=True)
class RollingVisualMemory:
    """Bounded FIFO for iterative agents."""

    slots: List[EpisodicSlot] = field(default_factory=list)
    max_slots: int = 8

    def push(self, slot: EpisodicSlot) -> None:
        self.slots.append(slot)
        overflow = len(self.slots) - self.max_slots
        if overflow > 0:
            del self.slots[:overflow]


__all__ = ["EpisodicSlot", "RollingVisualMemory"]
