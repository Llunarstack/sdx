"""Advanced features: outpainting, inpainting, magic eraser, animation, remix."""

from .animation import AnimationFromImage
from .dynamic import DynamicQualityAdjustment
from .outpainting import InfiniteOutpainting
from .loop_video import LoopVideoGeneration
from .eraser import MagicEraser
from .engine import NovelCapabilitiesEngine
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
