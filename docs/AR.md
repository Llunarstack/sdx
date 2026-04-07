# Block-wise autoregressive (AR) generation

SDX supports **block-wise autoregressive** self-attention (ACDiT-style): the latent patch grid is split into blocks, and each block can only attend to earlier blocks (and within its own block, in raster order). This combines **diffusion** (noise → clean over time) with **autoregressive structure** (spatial order), which can improve composition and make outputs easier to “fix” by regenerating later blocks.

---

## What it does

- **`num_ar_blocks = 0`** (default): Standard **bidirectional** attention. Every patch can attend to every other patch. Same as typical DiT/FLUX.
- **`num_ar_blocks = 2`**: Grid is split into **2×2 blocks**. Block order is raster (top-left → top-right → bottom-left → bottom-right). Within each block, patches also follow raster order (causal). So the model effectively generates “top-left first”, then “top-right” (seeing top-left), then “bottom-left” (seeing both top blocks), then “bottom-right” (seeing all previous).
- **`num_ar_blocks = 4`**: **4×4 blocks**. Stronger AR bias: 16 blocks in raster order, each block causal within itself. More structure, potentially slower and more constrained.

The **mask** is applied only to **self-attention** over the spatial patches. Cross-attention to the text remains bidirectional (all patches can attend to all text tokens).

---

## Why use AR?

- **Better structure**: The model can commit to a layout (e.g. face in top-left) and then fill in the rest, reducing “averaged” or incoherent compositions.
- **Fixability**: In theory you could regenerate only later blocks (e.g. bottom half) while keeping the top; our current pipeline doesn’t expose block-by-block sampling, but the inductive bias can still help global coherence.
- **Closer to “ChatGPT for images”**: Autoregressive order over space is reminiscent of token-by-token generation in LLMs.

Trade-off: AR adds a **causal constraint**, so the model has less global context when generating later regions. For some datasets and resolutions, **0** (full bidirectional) can still look best; for others, **2** (2×2 blocks) gives a good balance.

---

## How to use

### Training

Set `--num-ar-blocks` when training:

```bash
# No AR (default)
python train.py --data-path /path/to/data

# 2×2 block-wise AR (recommended to try first)
python train.py --data-path /path/to/data --num-ar-blocks 2

# 4×4 block-wise AR (stronger structure)
python train.py --data-path /path/to/data --num-ar-blocks 4

# Same block grid, Morton (z-order) macro-block visit order (see docs/AR_EXTENSIONS.md)
python train.py --data-path /path/to/data --num-ar-blocks 2 --ar-block-order zorder

# Upgraded AR traversals
python train.py --data-path /path/to/data --num-ar-blocks 2 --ar-block-order snake
python train.py --data-path /path/to/data --num-ar-blocks 4 --ar-block-order spiral
```

The value is stored in the checkpoint config. **You must use the same `num_ar_blocks` at inference** as was used at training (sample.py and inference.py read it from the checkpoint). **`ar_block_order`** is saved too; inference builds the DiT via `get_dit_build_kwargs`, which restores it (missing older checkpoints default to **`raster`**).

### Config

In `config/train_config.py`:

- **`num_ar_blocks: int = 0`** — 0 = off, 2 = 2×2 blocks, 4 = 4×4 blocks. Other values (e.g. 3 for 3×3) are supported by the mask code but are less common.
- **`ar_block_order: str = "raster"`** — macro-block sequence: **`raster`** (row-major), **`zorder`** (Morton), **`snake`** (boustrophedon), or **`spiral`** (outside-in). See [AR_EXTENSIONS.md](AR_EXTENSIONS.md).

### Code

- **Mask**: `models/attention.py` → `create_block_causal_mask_2d(h, w, num_ar_blocks, block_order=...)`. Core logic: `models/ar_masks_extended.py`. Returns an (N, N) float mask with `-inf` where attention is disabled.
- **DiT**: `models/dit_text.py` and `models/dit_text_variants.py` (`DiT_Predecessor_Text`, `DiT_Supreme_Text`) register `_ar_mask` when `num_ar_blocks > 0` and pass it into every block’s self-attention.

### Inspect masks

```bash
python -m scripts.tools ar_mask_inspect --h 32 --w 32 --blocks 2 --compare
```

---

## Block order

Default macro-block order is **raster**: block `(0,0)`, then `(0,1)`, …, then `(1,0)`, `(1,1)`, … — top-left toward bottom-right. Within each macro-block, patch order is also raster.

Optional alternatives:
- **`zorder`**: Morton / z-order over block indices (2D locality–friendly sequence)
- **`snake`**: alternating row direction traversal
- **`spiral`**: outside-in traversal over macro blocks

Training flag: `--ar-block-order ...`. Details and extra helpers: **[AR_EXTENSIONS.md](AR_EXTENSIONS.md)**.

### Runtime curriculum and order-mix

`train.py` also supports runtime AR mutation:

- `--ar-curriculum-mode none|step|linear`
- `--ar-curriculum-warmup-steps ...`
- `--ar-curriculum-ramp-start ... --ar-curriculum-ramp-end ...`
- `--ar-curriculum-start-blocks ... --ar-curriculum-target-blocks ...`
- `--ar-order-mix raster,zorder,snake,spiral`

This lets training start with full bidirectional attention and progressively move to stronger AR, while optionally cycling traversal order to improve robustness.

---

## When to prefer 0 vs 2 vs 4

| Setting | Use when |
|--------|----------|
| **0** | You want standard diffusion behavior; maximum context everywhere; no AR bias. |
| **2** | You want a bit of structural bias and possibly better composition; good default to try if 0 feels “messy”. |
| **4** | You want strong left-to-right / top-to-bottom structure; smaller “chunks” per block, more causal steps. |

Start with **0** or **2**; only try **4** if you’re aiming for a very ordered layout and accept the extra constraint.

---

## Technical note: square latent grid

The AR mask is built for a **square** patch grid (e.g. 32×32 patches for 256×256 image with patch_size=2). Non-square latent sizes would require passing separate `h` and `w` into the mask builder; the current DiT path uses `p = sqrt(num_patches)` so both dimensions are equal.

---

## ViT scorer alignment (DiT AR ↔ quality / adherence)

The **`ViT/`** module scores finished images. If your **generator** was trained with `num_ar_blocks` 0, 2, or 4, failure modes differ slightly from full bidirectional DiT. To keep ViT scores **calibrated** to the layout the image came from:

1. **Training ViT** (`python -m vit_quality.train`): **AR conditioning is on by default**. Put the same regime on each JSONL row the model should learn:
   - **`num_ar_blocks`**, **`dit_num_ar_blocks`**, or **`ar_blocks`** — integer **`0`**, **`2`**, or **`4`**. Missing or invalid → **unknown** bucket (4th one-hot).
2. **Older ViT checkpoints** (no `use_ar_conditioning` in saved config) load as **text-only** (8-D caption features only). Use **`--no-ar-conditioning`** when training if you need weights compatible with that layout, or retrain with AR fields in the manifest.
3. **Inference** (`python -m vit_quality.infer`, `python -m vit_quality.export_embeddings`): If the checkpoint has `use_ar_conditioning: true`, each row’s `num_ar_blocks` / `dit_num_ar_blocks` / `ar_blocks` / `generator_num_ar_blocks` is read (including optional nested blobs like `dit_config`) and fused with caption features. Omitted → **unknown** one-hot unless you pass **`--default-num-ar-blocks 0|2|4`** (e.g. whole manifest is from one DiT run).

4. **Facebook / Meta DiT** (`facebookresearch/DiT`): vanilla checkpoints are **full bidirectional** → use **`num_ar_blocks = 0`** for ViT alignment. Only use `2`/`4` if you trained an SDX-compatible fork with the same block-causal mask API.

5. **Tag manifests from a DiT `.pt`**: `python -m scripts.tools ar_tag_manifest --dit-ckpt path/to/best.pt --manifest-jsonl data/in.jsonl --out data/out.jsonl` (or `--num-ar-blocks 2` without reading a file). Sets `num_ar_blocks`, mirrors `dit_num_ar_blocks`, and adds `ar_regime` (`full_bidirectional`, `block_ar_2x2`, …).

Bridge code: **`utils/architecture/ar_block_conditioning.py`** (`ar_conditioning_vector`, `parse_num_ar_blocks_from_row`, `read_num_ar_blocks_from_checkpoint`, `tag_manifest_row_ar`, `batch_ar_conditioning`).
