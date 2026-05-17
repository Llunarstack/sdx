from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True, slots=True)
class PolicyGate:
    """Composable safety / rights checks before sampling."""

    disallow_real_person_likeness: bool = False
    disallow_medical_claims_as_fact: bool = True
    require_user_attribution_for_style_clones: bool = False


@dataclass(slots=True)
class GenerationPolicy:
    gates: PolicyGate = field(default_factory=PolicyGate)
    allowed_topics: FrozenSet[str] | None = None
    banned_terms: FrozenSet[str] | None = None


__all__ = ["GenerationPolicy", "PolicyGate"]
