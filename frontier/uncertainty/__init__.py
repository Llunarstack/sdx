"""When the model should ask for clarification or crank guidance."""

from .confidence_gate import ConfidenceGate, UncertaintyReport, UncertaintySignal

__all__ = ["ConfidenceGate", "UncertaintyReport", "UncertaintySignal"]
