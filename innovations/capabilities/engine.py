"""
Novel capabilities facade — routes to unique edit/generation features (INNOVATION_GUIDE §7).
"""

from typing import List

from .animation import AnimationFromImage
from .dynamic import DynamicQualityAdjustment
from .outpainting import InfiniteOutpainting
from .loop_video import LoopVideoGeneration
from .eraser import MagicEraser
from .remix import ObjectRemixing
from .weights import PromptWeighting
from .inpainting import RealTimeInpainting

__all__ = [
    "AnimationFromImage",
    "DynamicQualityAdjustment",
    "InfiniteOutpainting",
    "LoopVideoGeneration",
    "MagicEraser",
    "NovelCapabilitiesEngine",
    "ObjectRemixing",
    "PromptWeighting",
    "RealTimeInpainting",
]


class NovelCapabilitiesEngine:
    """Unified novel capabilities system."""

    def __init__(self):
        self.outpainting = InfiniteOutpainting()
        self.inpainting = RealTimeInpainting()
        self.eraser = MagicEraser()
        self.animator = AnimationFromImage()
        self.remixer = ObjectRemixing()
        self.weighting = PromptWeighting()
        self.quality_adjuster = DynamicQualityAdjustment()
        self.loop_video = LoopVideoGeneration()

    def get_capabilities(self) -> List[str]:
        """List of novel capabilities."""
        return [
            "Infinite Outpainting (extend images infinitely)",
            "Real-time Inpainting (fill any masked region perfectly)",
            "Magic Eraser (remove objects without traces)",
            "Animation from Single Image (create smooth motion)",
            "Object Remixing (swap objects between images)",
            "Hyper-precise Prompt Weighting (control each word's influence)",
            "Dynamic Quality Adjustment (auto-optimize for prompt)",
            "Perfect Loop Video Generation (seamless looping videos)",
        ]
