"""Advanced features: outpainting, inpainting, magic eraser, animation, remix."""

from .animation import AnimationFromImage
from .dynamic import DynamicQualityAdjustment
from .engine import NovelCapabilitiesEngine
from .eraser import MagicEraser
from .inpainting import RealTimeInpainting
from .loop_video import LoopVideoGeneration
from .outpainting import InfiniteOutpainting
from .remix import ObjectRemixing
from .weights import PromptWeighting

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
