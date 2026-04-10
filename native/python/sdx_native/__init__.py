"""Python bindings / helpers for ``native/`` (Rust, Zig, Go, C++, and pure-Python JSONL tools)."""

from .attention_mask_pack import bool_mask_to_additive, causal_mask_1d
from .batching_pad_fast import pad_1d_sequences, pad_2d_hw
from .buffer_scan_fast import count_newlines_buffer, fnv1a64_update, scan_file_chunks
from .c_buffer_stats_native import (
    count_newlines_py,
    get_buffer_stats_lib,
    maybe_count_newlines_native,
    maybe_sum_bytes_native,
    newline_and_sum,
    sum_bytes_py,
)
from .caption_csv_fast import (
    batch_normalize_captions,
    dedupe_caption_parts_preserve_order,
    merge_caption_csv,
    normalize_caption_csv,
    split_caption_parts,
    token_overlap_ratio,
)
from .coord_grid_fast import normalized_grid, pixel_center_grid
from .cuda_image_metrics_native import maybe_image_luma_stats_cuda
from .diffusion_sigma_fast import (
    clip_snr,
    effective_noise_weight,
    sigma_from_alpha_cumprod,
    snr_from_alpha_cumprod as snr_from_alpha_cumprod_numpy,
)
from .image_metrics_native import maybe_count_components_native, maybe_image_stats_native
from .jsonl_caption_hygiene import normalize_jsonl_caption_fields, normalize_manifest_jsonl
from .latent_geometry import (
    dit_patch_size_from_variant_name,
    latent_numel,
    latent_spatial_size,
    num_patch_tokens,
    patch_grid_dim,
)
from .manifest_line_index import build_line_offset_table, iter_jsonl_line_offsets, read_line_at
from .native_fast_stack_status import fast_numpy_stack_status
from .numpy_chw_pack import channel_mean_std, chw_f32_to_hwc_u8, hwc_u8_to_chw_f32, stack_chw_batch
from .numpy_latent_ops import batch_latent_rms, center_crop_hw, latent_flat_cosine, latent_mse, reflect_pad_hw
from .prompt_hash_fast import blake2b_hex, normalized_caption_key, try_xxhash_hex
from .relpath_norm_fast import relpath_if_under, to_posix_key, unique_preserve_order
from .resize_nearest_np import downscale_max_hwc, resize_hwc_nearest
from .rmsnorm_native import maybe_rmsnorm_rows_cuda
from .rope_apply_native import maybe_apply_rope_interleaved_cuda
from .score_ops_native import maybe_norm01_native, maybe_weighted_sum_native
from .silu_gate_native import maybe_silu_gate_cuda
from .text_hygiene import (
    caption_fingerprint,
    normalize_caption_for_training,
    pos_neg_token_overlap,
)
from .timestep_grid_fast import (
    append_zero_terminal,
    cosine_spacing_int,
    linspace_int_timesteps,
    match_dims,
    uniform_subsample,
)
from .torch_contiguous_fast import ensure_contiguous_torch, numpy_to_torch_float32, torch_to_numpy_contiguous
from .uint8_histogram_fast import clip_histogram_percentiles, histogram_u8_channels, luminance_histogram_u8

__all__ = [
    "append_zero_terminal",
    "batch_latent_rms",
    "batch_normalize_captions",
    "blake2b_hex",
    "bool_mask_to_additive",
    "build_line_offset_table",
    "caption_fingerprint",
    "causal_mask_1d",
    "center_crop_hw",
    "channel_mean_std",
    "chw_f32_to_hwc_u8",
    "clip_histogram_percentiles",
    "clip_snr",
    "cosine_spacing_int",
    "count_newlines_buffer",
    "count_newlines_py",
    "dedupe_caption_parts_preserve_order",
    "dit_patch_size_from_variant_name",
    "downscale_max_hwc",
    "effective_noise_weight",
    "ensure_contiguous_torch",
    "fast_numpy_stack_status",
    "fnv1a64_update",
    "get_buffer_stats_lib",
    "histogram_u8_channels",
    "hwc_u8_to_chw_f32",
    "iter_jsonl_line_offsets",
    "latent_flat_cosine",
    "latent_mse",
    "latent_numel",
    "latent_spatial_size",
    "linspace_int_timesteps",
    "luminance_histogram_u8",
    "match_dims",
    "maybe_apply_rope_interleaved_cuda",
    "maybe_count_components_native",
    "maybe_count_newlines_native",
    "maybe_image_luma_stats_cuda",
    "maybe_image_stats_native",
    "maybe_norm01_native",
    "maybe_rmsnorm_rows_cuda",
    "maybe_silu_gate_cuda",
    "maybe_sum_bytes_native",
    "maybe_weighted_sum_native",
    "merge_caption_csv",
    "newline_and_sum",
    "normalize_caption_csv",
    "normalize_caption_for_training",
    "normalize_jsonl_caption_fields",
    "normalize_manifest_jsonl",
    "normalized_caption_key",
    "normalized_grid",
    "num_patch_tokens",
    "numpy_to_torch_float32",
    "pad_1d_sequences",
    "pad_2d_hw",
    "patch_grid_dim",
    "pixel_center_grid",
    "pos_neg_token_overlap",
    "read_line_at",
    "reflect_pad_hw",
    "relpath_if_under",
    "resize_hwc_nearest",
    "scan_file_chunks",
    "sigma_from_alpha_cumprod",
    "snr_from_alpha_cumprod_numpy",
    "split_caption_parts",
    "stack_chw_batch",
    "sum_bytes_py",
    "to_posix_key",
    "token_overlap_ratio",
    "torch_to_numpy_contiguous",
    "try_xxhash_hex",
    "uniform_subsample",
    "unique_preserve_order",
]
