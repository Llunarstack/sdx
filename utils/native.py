"""
Lazy bridge to ``sdx_native`` (``native/_experimental/python``).

Importing this module only puts ``native/_experimental/python`` on ``sys.path``; each name in
``_EXPORTS`` is imported from its ``sdx_native`` module on first attribute access. The two helpers
defined here fall back to NumPy when the native build is missing.
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
    **dict.fromkeys(
        ("maybe_count_components_native", "maybe_image_luma_stats_cuda", "maybe_image_stats_native"),
        "sdx_native.ffi",
    ),
    **dict.fromkeys(
        (
            "dit_patch_size_from_variant_name",
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

__all__ = sorted([*_EXPORTS, "native_stack_summary", "quality_highlight_frac"])


def quality_highlight_frac(rgb: Any, thr: float = 245.0) -> float:
    """Fraction of pixels with Rec.601 luma >= ``thr`` (Rust kernel when built)."""
    try:
        from sdx_native.ffi import highlight_frac

        hit = highlight_frac(rgb, thr=thr)
        if hit is not None:
            return float(hit)
    except Exception:
        pass
    import numpy as np

    arr = np.asarray(rgb)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(np.mean(gray >= float(thr)))


def native_stack_summary() -> dict[str, Any]:
    """Which native backends are loadable (for dashboards / copilot notes)."""
    try:
        from sdx_native.ffi import rust_available
        from sdx_native.native_tools import native_stack_status

        return {"quality": rust_available(), "paths": native_stack_status()}
    except Exception as exc:
        return {"error": str(exc)}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
