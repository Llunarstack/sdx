"""Rehearsal pipeline stages — animatic before full GPU burn."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict

__all__ = ["RehearsalStage", "stage_edit_overrides"]


class RehearsalStage(str, Enum):
    ANIMATIC = "animatic"
    LOW_POLY = "low_poly"
    KEYFRAME = "keyframe"
    FULL = "full"


def stage_edit_overrides(stage: str) -> Dict[str, Any]:
    s = (stage or "full").lower()
    if s == RehearsalStage.ANIMATIC.value:
        return {
            "keyframe_interval": 12,
            "edit_strength": 0.35,
            "frame_enhance": False,
            "post_grade": "",
            "quality_retry": False,
        }
    if s == RehearsalStage.LOW_POLY.value:
        return {
            "keyframe_interval": 8,
            "edit_strength": 0.45,
            "frame_enhance": False,
            "post_grade": "muted",
        }
    if s == RehearsalStage.KEYFRAME.value:
        return {"keyframe_interval": 6, "quality_retry": True}
    return {}
