"""Canonical ViT quality/adherence package.

Public symbols resolve lazily from sibling modules (``import vit_quality`` does not
import model/loss code until a name is used).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ViTConfig",
    "ViTQualityAdherenceModel",
    "build_vit_model",
    "load_vit_quality_checkpoint",
    "vit_model_parameter_report",
    "pairwise_ranking_loss",
    "binary_focal_loss_with_logits",
    "breakdown_prompt",
    "compose_positive_with_embedded_negative",
    "build_prompt_plan",
]

_ATTR_ORIGIN: dict[str, str] = {
    "ViTConfig": "config",
    "ViTQualityAdherenceModel": "model",
    "build_vit_model": "model",
    "load_vit_quality_checkpoint": "checkpoint_utils",
    "vit_model_parameter_report": "checkpoint_utils",
    "pairwise_ranking_loss": "losses",
    "binary_focal_loss_with_logits": "losses",
    "breakdown_prompt": "prompt_system",
    "compose_positive_with_embedded_negative": "prompt_system",
    "build_prompt_plan": "prompt_system",
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    stem = _ATTR_ORIGIN[name]
    mod = import_module(f".{stem}", __package__)
    val = getattr(mod, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
