"""
Research prototypes (ViT / AR / DiT helpers) — not wired into train/sample by default.

``research.agi_image`` is torch-free scaffolding for agentic / AGI-facing image pipelines.

Torch-backed submodules are **lazy-loaded** (``__getattr__``) so ``import research.agi_image``
does not import PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any, List

from . import agi_image  # noqa: F401 — torch-free

_TORCH_AVAILABLE = find_spec("torch") is not None

_TORCH_LAZY: frozenset[str] = frozenset(
    {
        "autoregressive_plans",
        "creature_character_guidance",
        "diffusion_noise_structures",
        "hybrid_sampling_schedules",
        "latent_agreement",
        "physics_visual_guidance",
        "quality_timestep_weights",
        "visual_quality",
    }
)


def __getattr__(name: str) -> Any:
    if name in _TORCH_LAZY:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                f"research.{name} requires PyTorch; install torch or import research.agi_image (no torch)."
            )
        mod = import_module(f".{name}", __package__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    exposed = set(__all__)
    exposed.update(k for k in globals() if not k.startswith("_"))
    return sorted(exposed)


__all__ = [
    "agi_image",
    "autoregressive_plans",
    "creature_character_guidance",
    "diffusion_noise_structures",
    "hybrid_sampling_schedules",
    "latent_agreement",
    "physics_visual_guidance",
    "quality_timestep_weights",
    "visual_quality",
]
