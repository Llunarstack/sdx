"""Upscale and face-restore HF scaffold helpers."""

from __future__ import annotations

from typing import Dict, List

from utils.modeling import hf_scaffold
from utils.modeling.model_paths import (
    default_codeformer_path,
    default_gfpgan_path,
    default_realesrgan_path,
    default_swinir_classical_path,
)

_UPSCALE_REGISTRY: Dict[str, str] = {
    "Real-ESRGAN": "ai-forever/Real-ESRGAN",
    "SwinIR-classical-x2": "caidas/swin2SR-classical-sr-x2-64",
}

_FACE_REGISTRY: Dict[str, str] = {
    "CodeFormer": "sczhou/CodeFormer",
    "GFPGAN": "TencentARC/GFPGAN",
}


def list_upscale_models() -> List[str]:
    return sorted(_UPSCALE_REGISTRY.keys())


def list_face_restore_models() -> List[str]:
    return sorted(_FACE_REGISTRY.keys())


def upscale_has_weights(name: str) -> bool:
    key = str(name).strip()
    resolvers = {
        "Real-ESRGAN": default_realesrgan_path,
        "SwinIR-classical-x2": default_swinir_classical_path,
    }
    fn = resolvers.get(key)
    if fn is None:
        return False
    return hf_scaffold.has_local_weights(fn())


def face_restore_has_weights(name: str) -> bool:
    key = str(name).strip()
    resolvers = {
        "CodeFormer": default_codeformer_path,
        "GFPGAN": default_gfpgan_path,
    }
    fn = resolvers.get(key)
    if fn is None:
        return False
    return hf_scaffold.has_local_weights(fn())


__all__ = [
    "face_restore_has_weights",
    "list_face_restore_models",
    "list_upscale_models",
    "upscale_has_weights",
]
