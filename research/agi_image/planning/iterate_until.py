from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class IterationBudget:
    """Hard caps for verify–repair horizons."""

    max_samples_per_region: int = 6
    max_inpaint_retries: int = 4
    max_full_regenerations: int = 2


@dataclass(slots=True)
class VerificationSnapshot:
    """One scoring pass artefact."""

    iteration: int
    metrics: dict[str, float]
    notes: str | None = None
    raw: dict[str, Any] | None = None


__all__ = ["IterationBudget", "VerificationSnapshot"]
