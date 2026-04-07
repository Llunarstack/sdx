# AR extensions (macro-block order, layout, curriculum)

This note extends [AR.md](AR.md): same **block-causal** self-attention idea, with extra **visit orders** and helper modules for training / inspection.

## Block visit order (`ar_block_order`)

Training flag (and checkpoint field): **`ar_block_order`**.

| Value | Meaning |
|--------|--------|
| **`raster`** (default) | Macro-blocks ordered row-major: `(0,0)`, `(0,1)`, …, `(1,0)`, … — same as the original implementation. |
| **`zorder`** | Morton / z-order over block indices: tends to preserve **2D locality** between successive blocks (neighbor blocks in space are often closer in the sequence than pure raster). |
| **`snake`** | Boustrophedon row scan: row 0 left→right, row 1 right→left, alternating each row. |
| **`spiral`** | Outside-in spiral over macro-blocks; useful when central composition is important. |

Within each macro-block, patch order stays **raster** (row-major inside the block).

Implementation: `models/ar_masks_extended.py` → `create_block_causal_mask_2d(..., block_order=...)`, re-exported from `models/attention.py` as `create_block_causal_mask_2d`.

CLI:

```bash
python train.py --data-path /path/to/data --num-ar-blocks 2 --ar-block-order zorder
python train.py --data-path /path/to/data --num-ar-blocks 2 --ar-block-order snake
python train.py --data-path /path/to/data --num-ar-blocks 4 --ar-block-order spiral
```

Inference must match training: `get_dit_build_kwargs` passes `ar_block_order` from the saved `TrainConfig`, so checkpoints trained with `zorder` load the same mask.

## Related files

| Path | Role |
|------|------|
| [models/ar_masks_extended.py](../models/ar_masks_extended.py) | Mask construction, sparsity stats, optional mask softening helpers. |
| [utils/architecture/ar_block_layout.py](../utils/architecture/ar_block_layout.py) | Macro-block centers, `block_visit_order`, `patch_block_map` (for tooling / future schedulers). |
| [utils/training/ar_curriculum.py](../utils/training/ar_curriculum.py) | Schedules for ramping `num_ar_blocks` by step (optional; wire in `train.py` if you use it). |
| [utils/generation/ar_latent_ops.py](../utils/generation/ar_latent_ops.py) | Iterate / stack / paste macro-block views on `(B,C,H,W)` latents (raster block grid). |
| [scripts/tools/dev/ar_mask_inspect.py](../scripts/tools/dev/ar_mask_inspect.py) | CLI: mask shape, allowed-pair fraction, visit order; `--compare` raster vs zorder. |

## Runtime AR curriculum (train.py)

You can now vary AR strength during training:

```bash
# Step switch: start full-attn, switch to 2x2 AR after warmup.
python train.py --num-ar-blocks 2 --ar-curriculum-mode step --ar-curriculum-warmup-steps 12000

# Linear ramp to stronger AR.
python train.py --num-ar-blocks 4 --ar-curriculum-mode linear \
  --ar-curriculum-ramp-start 5000 --ar-curriculum-ramp-end 30000
```

Order mixing can be enabled with deterministic per-step cycling:

```bash
python train.py --num-ar-blocks 2 --ar-order-mix raster,zorder,snake,spiral
```

Note: dynamic AR runtime (`--ar-curriculum-mode != none` or `--ar-order-mix`) disables `torch.compile` in training to avoid graph-stability issues while masks mutate.

## ViT alignment

The ViT bridge ([utils/architecture/ar_block_conditioning.py](../utils/architecture/ar_block_conditioning.py)) still keys off **`num_ar_blocks`** (0 / 2 / 4 / …). It does **not** encode `ar_block_order` today; if you train DiT with `zorder` and want ViT conditioning to reflect that, extend the JSONL schema and conditioning vector in a follow-up.
