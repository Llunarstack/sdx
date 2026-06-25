"""Photorealism stack and anti-AI-slop guidance."""

from .anti_slop import AntiSlopPlan, AntiSlopScanner, RealismTier
from .photoreal_stack import PhotorealPlan, PhotorealStack

__all__ = [
    "AntiSlopPlan",
    "AntiSlopScanner",
    "PhotorealPlan",
    "PhotorealStack",
    "RealismTier",
]
