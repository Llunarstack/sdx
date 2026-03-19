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
```

The value is stored in the checkpoint config. **You must use the same `num_ar_blocks` at inference** as was used at training (sample.py and inference.py read it from the checkpoint).

### Config

In `config/train_config.py`:

- **`num_ar_blocks: int = 0`** — 0 = off, 2 = 2×2 blocks, 4 = 4×4 blocks. Other values (e.g. 3 for 3×3) are supported by the mask code but are less common.

### Code

- **Mask**: `models/attention.py` → `create_block_causal_mask_2d(h, w, num_ar_blocks)`. Returns an (N, N) float mask with `-inf` where attention is disabled.
- **DiT**: `models/dit_text.py` and `models/dit_predecessor.py` register `_ar_mask` when `num_ar_blocks > 0` and pass it into every block’s self-attention.

---

## Block order

Current order is **raster** over blocks:

1. Block (0,0), then (0,1), …, then (1,0), (1,1), …

So “earlier” in AR terms is top-left toward bottom-right. Within a block, patch order is also raster (row by row). This is fixed in the current code; see IMPROVEMENTS.md for ideas (e.g. spiral order, or AR only in early layers).

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
