"""
Probe optional dependencies and return human-readable status (no imports required to use this file).
"""

from __future__ import annotations

from typing import Any, Dict


def _try_version(mod_name: str) -> Dict[str, Any]:
    try:
        m = __import__(mod_name)
        ver = getattr(m, "__version__", "ok")
        return {"installed": True, "version": str(ver)}
    except ImportError:
        return {"installed": False, "version": None}


def describe_optional_libs() -> Dict[str, Any]:
    """
    Hints for faster manifests, prettier CLI, etc. (see ``extras/requirements-suggested.txt``).
    """
    keys = (
        "xxhash",
        "rich",
        "humanize",
        "orjson",
        "psutil",
        "timm",
        "xformers",
        "triton",
        "wandb",
        "tensorboard",
    )
    return {k: _try_version(k) for k in keys}


__all__ = ["describe_optional_libs"]
