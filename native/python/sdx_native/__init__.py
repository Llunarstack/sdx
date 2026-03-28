"""Python bindings / helpers for ``native/`` (Rust, Zig, Go, C++, and pure-Python JSONL tools)."""

from .latent_geometry import (
    dit_patch_size_from_variant_name,
    latent_numel,
    latent_spatial_size,
    num_patch_tokens,
    patch_grid_dim,
)
from .text_hygiene import (
    caption_fingerprint,
    normalize_caption_for_training,
    pos_neg_token_overlap,
)
from .rope_apply_native import maybe_apply_rope_interleaved_cuda
from .rmsnorm_native import maybe_rmsnorm_rows_cuda
from .silu_gate_native import maybe_silu_gate_cuda

__all__ = [
    "caption_fingerprint",
    "dit_patch_size_from_variant_name",
    "latent_numel",
    "latent_spatial_size",
    "normalize_caption_for_training",
    "num_patch_tokens",
    "patch_grid_dim",
    "pos_neg_token_overlap",
    "maybe_apply_rope_interleaved_cuda",
    "maybe_rmsnorm_rows_cuda",
    "maybe_silu_gate_cuda",
]
