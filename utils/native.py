"""
Lazy bridge to ``sdx_native`` (``native/_experimental/python``).

Importing this module only puts ``native/_experimental/python`` on ``sys.path``; each name below is
imported from its ``sdx_native`` module on first attribute access.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

_NP = Path(__file__).resolve().parents[1] / "native" / "_experimental" / "python"
if str(_NP) not in sys.path:
    sys.path.insert(0, str(_NP))

_EXPORTS: dict[str, str] = {
    "dit_patch_size_from_variant_name": "sdx_native.latent_geometry",
    "maybe_count_components_native": "sdx_native.image_metrics_native",
    "maybe_image_luma_stats_cuda": "sdx_native.image_metrics_native",
    "maybe_image_stats_native": "sdx_native.image_metrics_native",
    **dict.fromkeys(
        (
            "file_md5_hex",
            "get_latent_lib",
            "manifest_fingerprint_line",
            "merge_jsonl_files",
            "native_stack_status",
            "run_rust_dup_image_paths",
            "run_rust_image_paths",
            "run_rust_jsonl_stats",
            "run_rust_jsonl_validate",
            "rust_jsonl_tools_exe",
        ),
        "sdx_native.native_tools",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
