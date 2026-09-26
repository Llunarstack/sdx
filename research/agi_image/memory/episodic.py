from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EpisodicSlot:
    """One saved render + minimal structured recall."""

    step_id: str
    thumbnail_ref: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    latent_digest: str | None = None
    commentary: str = ""


@dataclass(slots=True)
class RollingVisualMemory:
    """Bounded FIFO for iterative agents."""

    slots: list[EpisodicSlot] = field(default_factory=list)
    max_slots: int = 8

    def push(self, slot: EpisodicSlot) -> None:
        self.slots.append(slot)
        overflow = len(self.slots) - self.max_slots
        if overflow > 0:
            del self.slots[:overflow]


__all__ = ["EpisodicSlot", "RollingVisualMemory"]
