"""
Unified native bridge for backward-compatible imports.

``native/python`` is placed on ``sys.path`` so ``sdx_native`` resolves. The legacy
``import *`` merge (seven modules, **later** attributes overwrite earlier duplicates) is
built on **first attribute access**, then cached.

Regenerate ``_NATIVE_EXPORTS`` after changing ``sdx_native`` public names::

    python scripts/tools/dev/refresh_native_exports.py --write
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

_NP = Path(__file__).resolve().parents[2] / "native" / "python"
if str(_NP) not in sys.path:
    sys.path.insert(0, str(_NP))

_MERGE_ORDER: tuple[str, ...] = (
    "sdx_native",
    "sdx_native.cuda_image_metrics_native",
    "sdx_native.image_metrics_native",
    "sdx_native.latent_geometry",
    "sdx_native.native_tools",
    "sdx_native.score_ops_native",
    "sdx_native.text_hygiene",
)

_MERGED: dict[str, Any] | None = None

# BEGIN_NATIVE_EXPORTS
_NATIVE_EXPORTS: frozenset[str] = frozenset(
    {
        'Any',
        'CudaImageMetricsLib',
        'Dict',
        'ImageMetricsLib',
        'LatentLib',
        'List',
        'Optional',
        'Path',
        'REPO_ROOT',
        'ScoreOpsLib',
        'Tuple',
        'annotations',
        'append_zero_terminal',
        'batch_latent_rms',
        'batch_normalize_captions',
        'beta_schedules_shared_library_path',
        'blake2b_hex',
        'bool_mask_to_additive',
        'build_line_offset_table',
        'c_buffer_stats_shared_library_path',
        'caption_fingerprint',
        'causal_mask_1d',
        'center_crop_hw',
        'channel_mean_std',
        'chw_f32_to_hwc_u8',
        'clip_histogram_percentiles',
        'clip_snr',
        'cosine_spacing_int',
        'count_newlines_buffer',
        'count_newlines_py',
        'ctypes',
        'cuda_flow_matching_shared_library_path',
        'cuda_gaussian_blur_shared_library_path',
        'cuda_hwc_to_chw_shared_library_path',
        'cuda_image_metrics_shared_library_path',
        'cuda_ml_shared_library_path',
        'cuda_nf4_shared_library_path',
        'cuda_percentile_clamp_shared_library_path',
        'cuda_rmsnorm_shared_library_path',
        'cuda_rope_shared_library_path',
        'cuda_sdpa_online_shared_library_path',
        'cuda_silu_gate_shared_library_path',
        'dedupe_caption_parts_preserve_order',
        'dit_patch_size_from_variant_name',
        'downscale_max_hwc',
        'effective_noise_weight',
        'ensure_contiguous_torch',
        'fast_numpy_stack_status',
        'file_md5_hex',
        'fnv1a64_bytes',
        'fnv1a64_file',
        'fnv1a64_update',
        'fnv64_file_shared_library_path',
        'get_buffer_stats_lib',
        'get_cuda_image_metrics_lib',
        'get_image_metrics_lib',
        'get_latent_lib',
        'get_score_ops_lib',
        'go_sdx_manifest_exe',
        'hashlib',
        'histogram_u8_channels',
        'hwc_u8_to_chw_f32',
        'image_metrics_shared_library_path',
        'inference_timesteps_shared_library_path',
        'iter_jsonl_line_offsets',
        'json',
        'jsonl_row_caption_keys',
        'latent_flat_cosine',
        'latent_mse',
        'latent_numel',
        'latent_shared_library_path',
        'latent_spatial_size',
        'line_stats_shared_library_path',
        'linspace_int_timesteps',
        'luminance_histogram_u8',
        'manifest_fingerprint_line',
        'mask_ops_shared_library_path',
        'match_dims',
        'maybe_apply_rope_interleaved_cuda',
        'maybe_count_components_native',
        'maybe_count_newlines_native',
        'maybe_image_luma_stats_cuda',
        'maybe_image_stats_native',
        'maybe_norm01_native',
        'maybe_rmsnorm_rows_cuda',
        'maybe_rust_file_md5_hex',
        'maybe_silu_gate_cuda',
        'maybe_sum_bytes_native',
        'maybe_weighted_sum_native',
        'merge_caption_csv',
        'merge_jsonl_files',
        'mojo_cli_path',
        'native_stack_status',
        'newline_and_sum',
        'normalize_caption_csv',
        'normalize_caption_for_training',
        'normalize_jsonl_caption_fields',
        'normalize_manifest_jsonl',
        'normalized_caption_key',
        'normalized_grid',
        'np',
        'num_patch_tokens',
        'numpy_to_torch_float32',
        'pad_1d_sequences',
        'pad_2d_hw',
        'patch_grid_dim',
        'pixel_center_grid',
        'pos_neg_token_overlap',
        'py_latent_numel',
        'py_latent_spatial_size',
        'py_num_patch_tokens',
        'py_patch_grid_dim',
        'read_line_at',
        'reflect_pad_hw',
        'relpath_if_under',
        'resize_hwc_nearest',
        'rmsnorm_rows_cpu_shared_library_path',
        'run_rust_dup_image_paths',
        'run_rust_file_fnv',
        'run_rust_file_md5',
        'run_rust_image_paths',
        'run_rust_jsonl_stats',
        'run_rust_jsonl_validate',
        'run_rust_noise_schedule',
        'run_zig_linecrc_file',
        'run_zig_pathstat_list',
        'rust_diffusion_math_shared_library_path',
        'rust_image_metrics_exe',
        'rust_jsonl_tools_exe',
        'rust_noise_schedule_exe',
        'scan_file_chunks',
        'score_ops_shared_library_path',
        'shutil',
        'sigma_from_alpha_cumprod',
        'snr_from_alpha_cumprod_numpy',
        'split_caption_parts',
        'stack_chw_batch',
        'strip_c0_controls',
        'strip_zwsp',
        'subprocess',
        'sum_bytes_py',
        'to_posix_key',
        'token_overlap_ratio',
        'torch_to_numpy_contiguous',
        'try_xxhash_hex',
        'uniform_subsample',
        'unique_preserve_order',
        'zig_linecrc_exe',
        'zig_pathstat_exe',
    }
)
# END_NATIVE_EXPORTS

__all__: list[str] = sorted(_NATIVE_EXPORTS)


def _merged_exports() -> dict[str, Any]:
    global _MERGED
    if _MERGED is None:
        out: dict[str, Any] = {}
        for fqmn in _MERGE_ORDER:
            mod = import_module(fqmn)
            for k in dir(mod):
                if k.startswith("_"):
                    continue
                out[k] = getattr(mod, k)
        _MERGED = out
    return _MERGED


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    m = _merged_exports()
    if name not in m:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    val = m[name]
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    if _MERGED is None:
        return sorted(set(globals()) | set(__all__))
    return sorted(set(globals()) | set(_MERGED.keys()))
