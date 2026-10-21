"""Facade over optional ``sdx_native`` kernels with NumPy fallbacks (call sites never need a native build)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_SDX_NATIVE_PY = Path(__file__).resolve().parents[2] / "native" / "_experimental" / "python"
if str(_SDX_NATIVE_PY) not in sys.path:
    sys.path.insert(0, str(_SDX_NATIVE_PY))

__all__ = ["native_stack_summary", "quality_highlight_frac"]


def quality_highlight_frac(rgb: np.ndarray, thr: float = 245.0) -> float:
    """Fraction of pixels with Rec.601 luma >= ``thr`` (Rust kernel when built)."""
    try:
        from sdx_native.quality_scorers_native import highlight_frac

        hit = highlight_frac(rgb, thr=thr)
        if hit is not None:
            return float(hit)
    except Exception:
        pass
    arr = np.asarray(rgb)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(np.mean(gray >= float(thr)))


def native_stack_summary() -> dict[str, Any]:
    """Which native backends are loadable (for dashboards / copilot notes)."""
    try:
        from sdx_native import quality_scorers_native
        from sdx_native.native_tools import native_stack_status

        return {"quality": quality_scorers_native.available(), "paths": native_stack_status()}
    except Exception as exc:
        return {"error": str(exc)}
