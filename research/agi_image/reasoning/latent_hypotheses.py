from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class InterpretationHypothesis:
    """
    Structured alternative readings of a prompt (helps resolve multimodal ambiguity).
    Executor may score each hypothesis with cheap probes before committing compute.
    """

    name: str
    reading: str
    evidence_phrases: List[str] = field(default_factory=list)
    latent_sketch: Dict[str, Any] = field(default_factory=dict)


__all__ = ["InterpretationHypothesis"]
