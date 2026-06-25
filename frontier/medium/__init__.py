"""Brush stroke physics and extended medium packs."""

from .brush_planner import BrushPlan, BrushPlanner, StrokeStyle
from .extended_mediums import EXTENDED_MEDIUM_SPECS, detect_extended_medium_ids, extended_guidance_fragments

__all__ = [
    "BrushPlan",
    "BrushPlanner",
    "EXTENDED_MEDIUM_SPECS",
    "StrokeStyle",
    "detect_extended_medium_ids",
    "extended_guidance_fragments",
]
