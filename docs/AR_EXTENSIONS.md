# AR extensions (macro-block order, layout, curriculum)

This note extends [AR.md](AR.md): same **block-causal** self-attention idea, with extra **visit orders** and helper modules for training / inspection.

## Block visit order (`ar_block_order`)

Training flag (and checkpoint field): **`ar_block_order`**.

| Value | Meaning |
|--------|--------|
| **`raster`** (default) | Macro-blocks ordered row-major: `(0,0)`, `(0,1)`, …, `(1,0)`, … — same as the original implementation. |
| **`zorder`** | Morton / z-order over block indices: tends to preserve **2D locality** between successive blocks (neighbor blocks in space are often closer in the sequence than pure raster). |

Within each macro-block, patch order stays **raster** (row-major inside the block).

Implementation: `models/ar_masks_extended.py` → `create_block_causal_mask_2d(..., block_order=...)`, re-exported from `models/attention.py` as `create_block_causal_mask_2d`.

CLI:

```bash
python train.py --data-path /path/to/data --num-ar-blocks 2 --ar-block-order zorder
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

## ViT alignment

The ViT bridge ([utils/architecture/ar_dit_vit.py](../utils/architecture/ar_dit_vit.py)) still keys off **`num_ar_blocks`** (0 / 2 / 4 / …). It does **not** encode `ar_block_order` today; if you train DiT with `zorder` and want ViT conditioning to reflect that, extend the JSONL schema and conditioning vector in a follow-up.
