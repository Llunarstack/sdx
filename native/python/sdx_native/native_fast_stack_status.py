"""Report which pure-Python / numpy fast-path modules are importable."""

from __future__ import annotations

from typing import Any, Dict


def fast_numpy_stack_status() -> Dict[str, Any]:
    """Cheap introspection for diagnostics (no optional DLLs required)."""
    out: Dict[str, Any] = {"numpy": False, "torch": False, "xxhash": False}
    try:
        import numpy as np  # noqa: F401

        out["numpy"] = True
    except Exception:
        pass
    try:
        import torch  # noqa: F401

        out["torch"] = True
    except Exception:
        pass
    try:
        import xxhash  # noqa: F401

        out["xxhash"] = True
    except Exception:
        pass
    out["modules"] = [
        "sdx_native.attention_mask_pack",
        "sdx_native.batching_pad_fast",
        "sdx_native.buffer_scan_fast",
        "sdx_native.c_buffer_stats_native",
        "sdx_native.caption_csv_fast",
        "sdx_native.coord_grid_fast",
        "sdx_native.diffusion_sigma_fast",
        "sdx_native.manifest_line_index",
        "sdx_native.numpy_chw_pack",
        "sdx_native.numpy_latent_ops",
        "sdx_native.prompt_hash_fast",
        "sdx_native.relpath_norm_fast",
        "sdx_native.resize_nearest_np",
        "sdx_native.timestep_grid_fast",
        "sdx_native.torch_contiguous_fast",
        "sdx_native.uint8_histogram_fast",
    ]
    return out
