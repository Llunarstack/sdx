"""
SDX Innovations — research modules + agentic quality stack.

Two layers:
  1. **Core pillars** (7 packages): quality, semantics, control,
     speed, consistency, multimodal, capabilities — lightweight
     ``nn.Module`` building blocks and facade engines.
  2. **Agentic** (``agentic/``): quality control, adherence, refinement, artifact detection.

Unified entry: ``SDXAdvancedPipeline`` in ``pipeline.py``.
Discovery: ``registry.py``.
"""

__version__ = "12.0.0"

__all__ = [
    "ConsistencyEngine",
    "MultimodalFusionEngine",
    "NovelCapabilitiesEngine",
    "PrecisionControlSystem",
    "RealtimeGenerationEngine",
    "SDXAdvancedPipeline",
    "SemanticUnderstandingEngine",
    "UltraQualityEngine",
    "create_advanced_pipeline",
    "describe_package",
    "list_packages",
]


def __getattr__(name: str):
    if name == "UltraQualityEngine":
        from .quality import UltraQualityEngine

        return UltraQualityEngine
    if name == "SemanticUnderstandingEngine":
        from .semantics import SemanticUnderstandingEngine

        return SemanticUnderstandingEngine
    if name == "PrecisionControlSystem":
        from .control import PrecisionControlSystem

        return PrecisionControlSystem
    if name == "RealtimeGenerationEngine":
        from .speed import RealtimeGenerationEngine

        return RealtimeGenerationEngine
    if name == "ConsistencyEngine":
        from .consistency import ConsistencyEngine

        return ConsistencyEngine
    if name == "MultimodalFusionEngine":
        from .multimodal import MultimodalFusionEngine

        return MultimodalFusionEngine
    if name == "NovelCapabilitiesEngine":
        from .capabilities import NovelCapabilitiesEngine

        return NovelCapabilitiesEngine
    if name == "SDXAdvancedPipeline":
        from .pipeline import SDXAdvancedPipeline

        return SDXAdvancedPipeline
    if name == "create_advanced_pipeline":
        from .pipeline import create_advanced_pipeline

        return create_advanced_pipeline
    if name == "list_packages":
        from .registry import list_packages

        return list_packages
    if name == "describe_package":
        from .registry import describe_package

        return describe_package
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
