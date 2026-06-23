"""
Module registry: what each innovations package does and where production code lives.
"""

from __future__ import annotations

from typing import Dict, List

PACKAGE_REGISTRY: Dict[str, Dict[str, object]] = {
    "quality": {
        "purpose": "Material-aware photorealism (skin, metal, cloth, GI) on latents or embeddings.",
        "engine": "UltraQualityEngine",
        "production": [
            "utils/quality/face_region_enhance.py",
            "sample.py --refine-t",
        ],
        "bridge": "innovations.quality.hooks",
    },
    "semantics": {
        "purpose": "Decompose prompts into objects, style, composition, nuance.",
        "engine": "SemanticUnderstandingEngine",
        "production": [
            "utils/prompt/prompt_layout.py",
            "data/caption_utils.py [layout]",
        ],
        "bridge": "innovations.semantics.hooks",
    },
    "control": {
        "purpose": "Spatial, color, lighting, camera, and FX control tensors.",
        "engine": "PrecisionControlSystem",
        "production": [
            "utils/generation/regional_box_prompting.py",
            "utils/generation/spatial_dsl/",
        ],
        "bridge": "innovations.control.hooks",
    },
    "speed": {
        "purpose": "Token pruning, caching, layer skip, tiled/batched inference.",
        "engine": "RealtimeGenerationEngine",
        "production": [
            "diffusion/gaussian_diffusion.py sample_loop caches",
            "sample.py --feature-cache-delta",
        ],
        "bridge": "innovations.speed.hooks",
    },
    "consistency": {
        "purpose": "Deterministic seeds, character/style memory, variation dial.",
        "engine": "ConsistencyEngine",
        "production": [
            "sample.py --seed --character-sheet",
            "models/multi_character.py",
        ],
        "bridge": "innovations.consistency.hooks",
    },
    "multimodal": {
        "purpose": "Fuse text, image, sketch, depth, scene graph, audio.",
        "engine": "MultimodalFusionEngine",
        "production": [
            "utils/generation/engine.py",
            "sample.py --control-image --init-image",
        ],
        "bridge": "innovations.multimodal.hooks",
    },
    "capabilities": {
        "purpose": "Outpaint, inpaint, eraser, animation, remix, prompt weighting.",
        "engine": "NovelCapabilitiesEngine",
        "production": [
            "sample.py --mask --init-image",
            "utils/generation/sample_edit_runner.py",
        ],
        "bridge": "innovations.capabilities.hooks",
    },
    "agentic": {
        "purpose": "Quality agents, adherence, refinement loops, artifact/drift detection.",
        "engine": "QualityControlSystem + IterativeRefinementLoop",
        "production": [
            "innovations/pipeline.py SDXAdvancedPipeline",
        ],
        "bridge": "innovations.pipeline",
    },
}


def list_packages() -> List[str]:
    return sorted(PACKAGE_REGISTRY.keys())


def describe_package(name: str) -> Dict[str, object]:
    key = (name or "").strip().lower()
    if key not in PACKAGE_REGISTRY:
        raise KeyError(f"unknown package: {name!r}; choose from {list_packages()}")
    return dict(PACKAGE_REGISTRY[key])


__all__ = ["PACKAGE_REGISTRY", "describe_package", "list_packages"]
