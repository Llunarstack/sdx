from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


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
    metrics: Dict[str, float]
    notes: Optional[str] = None
    raw: Dict[str, Any] | None = None


__all__ = ["IterationBudget", "VerificationSnapshot"]
