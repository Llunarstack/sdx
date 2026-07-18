"""
Research prototypes used by agentic / prompt helpers — not on the default train path.

``research.agi_image`` is torch-free scaffolding for agentic image pipelines.
Torch-backed submodules are lazy-loaded so ``import research.agi_image`` does not import PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

from . import agi_image  # noqa: F401 — torch-free

_TORCH_AVAILABLE = find_spec("torch") is not None

_TORCH_LAZY: frozenset[str] = frozenset(
    {
        "creature_character_guidance",
        "physics_visual_guidance",
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


def __dir__() -> list[str]:
    exposed = set(__all__)
    exposed.update(k for k in globals() if not k.startswith("_"))
    return sorted(exposed)


__all__ = [
    "agi_image",
    "creature_character_guidance",
    "physics_visual_guidance",
    "visual_quality",
]
