# DiT Next-Gen Notes (Practical)

This note tracks low-risk upgrades inspired by recent DiT/ViT efficiency papers (U-DiT, adaptive token compression, dynamic patch scheduling ideas), adapted to SDX in a backward-compatible way.

## Implemented in code

- **LayerScale residuals** (`models/vit_next_blocks.py`)
  - Per-channel residual gain for attention/cross-attention/MLP branches.
  - Enable with `--layerscale-init` (e.g. `1e-5`).

- **Stochastic depth / DropPath** (wired in DiT text blocks)
  - Linear depth schedule from block 0 to final block using `--drop-path-rate`.
  - Helps deep stacks regularize while keeping compute steady.

- **Top-k token keep gating** (`models/vit_next_blocks.py`)
  - Compute-preserving adaptive token selection over patch tokens (register tokens never suppressed).
  - Controls:
    - `--token-keep-ratio` (1.0 = disabled)
    - `--token-keep-min-value`
  - Designed as a safe stepping stone before hard token dropping.

## New train flags

- `--token-keep-ratio`
- `--token-keep-min-value`
- `--drop-path-rate`
- `--layerscale-init`

These are propagated through `TrainConfig` and `get_dit_build_kwargs`.

## Why this path

Hard token dropping and dynamic patch remeshing can break tensor shape assumptions across control paths, inpainting, and register tokens. This implementation keeps shape static while still letting the model prioritize salient tokens and improve depth stability.
