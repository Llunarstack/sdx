# Local model stack (`pretrained/`)

When folders exist under `pretrained/`, paths resolve automatically (see `utils/modeling/model_paths.py`). For the full runtime picture (repo folders, `ViT/` vs DiT, diagrams), see the **[README § Architecture and pipeline](../README.md#architecture-and-pipeline)** and **[FILES.md](FILES.md)**.

**2026 recommended hub ids (no auto-download):** see **[PRETRAINED_RECOMMENDED.md](#recommended-pretrained-stack-2026)**. Defaults prefer **Qwen3-8B** (T2I text encoder), **Qwen3-14B** (LLM), **Qwen3-VL-8B** (VLM), **DINOv3-L**, **SigLIP2**, **HPSv3**, **moondream3** — while still honoring legacy local folders (`Qwen2.5-14B-Instruct`, `DINOv2-Large`, …).

| Role | Folder / default |
|------|------------------|
| **Qwen3-8B (T2I text encoder)** | `pretrained/Qwen3-8B` or `Qwen/Qwen3-8B` |
| **Qwen3-14B (prompt LLM)** | `pretrained/Qwen3-14B` or `Qwen/Qwen3-14B` |
| **Qwen3-VL-8B (VLM)** | `pretrained/Qwen3-VL-8B-Instruct` or `Qwen/Qwen3-VL-8B-Instruct` |
| T5-XXL (classic DiT TE) | `pretrained/T5-XXL` or `google/t5-v1_1-xxl` |
| CLIP ViT-L/14 | `pretrained/CLIP-ViT-L-14` or `openai/clip-vit-large-patch14` |
| CLIP ViT-bigG/14 | `pretrained/CLIP-ViT-bigG-14` or LAION hub id |
| **DINOv3-L (REPA / critics)** | `pretrained/DINOv3-ViT-L16` or `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| Depth (DA3) | `pretrained/Depth-Anything-V3-Large` or `depth-anything/DA3-LARGE-1.1` |
| FLUX.2 Klein (external) | `pretrained/FLUX.2-klein-9B` or `black-forest-labs/FLUX.2-klein-9B` |
| DC-AE (modern VAE) | `pretrained/DC-AE-f32c32` or `mit-han-lab/dc-ae-f32c32-mix-1.0` |
| DINOv2 (legacy) | `pretrained/DINOv2-Large` or `facebook/dinov2-large` |
| **SigLIP2** | `pretrained/SigLIP2-SO400M` or `google/siglip2-so400m-patch16-384` |
| SigLIP (legacy) | `pretrained/SigLIP-SO400M` |
| Qwen LLM (legacy) | `pretrained/Qwen2.5-14B-Instruct` |
| Stable Cascade | `pretrained/StableCascade-Prior`, `pretrained/StableCascade-Decoder` |
| LongCLIP-L (long prompts) | `pretrained/LongCLIP-L` or `creative-graphic-design/LongCLIP-L` |
| **moondream3** | `pretrained/moondream3-preview` or `moondream/moondream3-preview` |
| moondream2 (legacy) | `pretrained/moondream2` |
| Marigold depth/normals | `pretrained/Marigold-Depth-v1-1`, `pretrained/Marigold-Normals-v1-1` |
| TAESD / TAESDXL (fast preview VAE) | `pretrained/TAESD`, `pretrained/TAESDXL` |
| Consistency Decoder | `pretrained/Consistency-Decoder` or `openai/consistency-decoder` |
| ConvNeXtV2-Large | `pretrained/ConvNeXtV2-Large` |
| LAION Aesthetic v2 | `pretrained/LAION-Aesthetic-v2` |
| CodeFormer (face restore helper) | `pretrained/CodeFormer` |
| AnyDoor reference weights | `pretrained/AnyDoor-Ref` |
| **HPSv3** | `pretrained/HPSv3` or `MizzenAI/HPSv3` |
| HPSv2 (legacy) | `pretrained/HPSv2-hf` |
| BLIP captioning (base) | `pretrained/BLIP-image-captioning-base` or `Salesforce/blip-image-captioning-base` |
| Kosmos-2 VLM | `pretrained/Kosmos-2-patch14-224` or `microsoft/kosmos-2-patch14-224` |
| CRAFT text detector | `pretrained/CRAFT-text-detector` or `boomb0om/CRAFT-text-detector` |
| OwlViT (open-vocab detector) | `pretrained/OwlViT-base-patch32` or `google/owlvit-base-patch32` |
| Donut (doc/layout understanding) | `pretrained/Donut-base` or `naver-clova-ix/donut-base` |
| Florence-2 (caption / vision tasks) | `pretrained/Florence-2-base` or `microsoft/Florence-2-base` |
| BLIP-2 (strong captions) | `pretrained/BLIP2-opt-2.7b` or `Salesforce/blip2-opt-2.7b` |
| Qwen2-VL-2B (compact VLM) | `pretrained/Qwen2-VL-2B-Instruct` or `Qwen/Qwen2-VL-2B-Instruct` |
| Depth-Anything-V2-Small | `pretrained/Depth-Anything-V2-Small` or hub id |
| CLIP ViT-H/14 | `pretrained/CLIP-ViT-H-14` or LAION hub id |
| OWLv2 detector | `pretrained/OWLv2-base-patch16-ensemble` |
| ControlNet Canny | `pretrained/ControlNet-Canny` or `lllyasviel/sd-controlnet-canny` |
| ControlNet Depth / OpenPose / Lineart / Scribble | `pretrained/ControlNet-*` or lllyasviel hub ids |
| SmolVLM-256M (tiny VLM) | `pretrained/SmolVLM-256M-Instruct` |
| Florence-2-large | `pretrained/Florence-2-large` |
| OneAlign (IQA/reward) | `pretrained/OneAlign` or `Q-Future/OneAlign` |
| MetaCLIP / AIMv2 / AltCLIP | vision encoder alternatives for alignment experiments |
| CAFE Aesthetic | `pretrained/CAFE-Aesthetic` |
| GroundingDINO-Tiny | lightweight open-vocab detector scaffold |
| SmolVLM2-2B / Qwen2-VL-7B / Phi-3.5-vision | additional VLM scaffolds |
| GIT-base/large-coco | caption models |
| ControlNet MLSD / SoftEdge / Seg / Normal | extended control scaffolds |
| GroundingDINO-SwinT / OWLv2-Large | stronger detectors |
| MUSIQ / PerceptCLIP / ImageReward | extended HF reward panel |
| EVA02-CLIP-L-14 / CLIP-ViT-L-336 | CLIP alignment variants |
| GOT-OCR2 / TrOCR-small | document OCR scaffolds |
| SAM-ViT-Huge | segmentation scaffold |
| T5-XL / T5-Large / CLIP-ViT-B-32 | lighter text encoder scaffolds |
| SigLIP2 / DINOv2-Base/Small | lighter vision encoder scaffolds |
| LLaVA-1.5-7B / InternVL2-2B / Qwen2.5-VL-3B | additional VLMs |
| ControlNet HED / SDXL canny+depth | extended control scaffolds |
| ZoeDepth / Metric3D / Marigold depth | depth estimation alternatives |
| DETR / Mask2Former / SAM2-Tiny | detection & segmentation |
| LayoutLMv3 / Donut-docvqa | document layout OCR |
| NSFW-Detector | safety gate scaffold |
| sd-vae-ft-mse / sdxl-vae-fp16-fix | VAE decode scaffolds |
| LLaVA-v1.6 / Qwen2.5-VL-7B / PaliGemma2 / InternVL2-1B | previous VLM wave |
| InternVL3 / Gemma-3 / Qwen3-VL | current VLM wave |
| DINOv3 / MobileCLIP-S2 / UMT5-XXL | modern encoders |
| Qwen-Image / FLUX.2 Klein / FLUX.2-dev | external MMDiT foundations (Diffusers) |
| DC-AE / Depth Anything 3 | modern VAE + depth scaffolds |
| ControlNet Union/OpenPose SDXL | SDXL control scaffolds |
| SAM2 Base/Small + GroundingDINO-1.5 | mid-tier seg/detect |
| Pix2Struct / DePlot / vit-gpt2-coco | doc/chart/caption helpers |
| Watermark-Detector / CLIP-IQA | QA + reward scaffolds |
| GFPGAN / SwinIR-classical | face restore + upscale scaffolds |

Registry index and post-process model lists: ``utils/modeling/model_paths.py`` (`registry_summary()`, `role_counts()`, `UPSCALE_MODELS`, `FACE_RESTORE_MODELS`).

## Config-only scaffold (no checkpoint download)

Fetch folder structure + configs/tokenizers without weight files:

```bash
python setup/download.py scaffold --all
python setup/download.py scaffold --role vlm --role reward
python setup/download.py scaffold --list
```

Runtime uses `resolve_model_path_require_weights()` so config-only local folders do **not** block Hugging Face fallback. Lazy loaders: `utils/modeling/hf_loaders.py`, unified reward panel: `utils/modeling/hf_loaders.py`.

## How this maps to the SDX pipeline

- **DiT generation** uses **text** encoders (T5 by default, optionally + CLIP fusion) and **image latents** from VAE/RAE — see the main **[README § Architecture and pipeline](../README.md#architecture-and-pipeline)** for full diagrams (DiT vs `ViT/` QA tools vs REPA). Qwen3-8B is the recommended TE for external FLUX.2 Klein / Qwen-Image stacks (sibling Diffusers), registered under `pretrained/Qwen3-8B`.
- **`ViT/`** in this repo is a **separate** scoring/ranking stack (timm ViT on manifests), not the same module as the DiT generator.
- **REPA** uses a **frozen vision** model (default **DINOv3-L**, legacy DINOv2 / CLIP) to align DiT **internal** features during training.

## Training

- **T5 only (default):** `python train.py ...` — same as before.
- **Triple text encoders:** T5 + CLIP-L + CLIP-bigG with a small trainable fusion:
  ```bash
  python train.py --data-path ... --text-encoder-mode triple
  ```
- **Penta text encoders:** T5 + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L (4 fused CLIP tokens):
  ```bash
  python train.py --data-path ... --text-encoder-mode penta
  ```
  Download weights (or config scaffolds):
  ```bash
  python setup/download.py models --penta-text-encoders
  python setup/download.py scaffold --penta
  python -m scripts.tools.ops.pretrained_status --text-encoder-mode penta
  ```
JSONL fields (optional, for triple/penta training):

- ``prompt_layout``: inline layout object (same schema as ``examples/prompt_layout.example.json``)
- ``prompt_layout_path``: path to layout JSON (relative to manifest / image root)

When set, CLIP encoders receive labeled section captions; LongCLIP (penta) receives the full training caption.

## Sampling

`sample.py` reads `text_encoder_mode` from the checkpoint config and loads the same stack.

**Solvers / schedules (2026):** defaults are `--scheduler ays_dit --solver dpmpp_2m` (VP) and
`--flow-solver dpmpp_2m --flow-schedule ays` (flow). See **[guides/SAMPLING.md](HOW_GENERATION_WORKS.md#sampling-guide-solvers--schedules)**.
Real backends live in `diffusion/solvers/` (DPM-Solver++ 2M/3M, UniPC, flow midpoint/AB2).

## Other

- **Qwen prompt expansion:** `utils/agentic/agent.py` (`load_qwen_causal_lm`, `expand_prompt_qwen`).
- **Stable Cascade (Diffusers):** `python -m scripts.tools cascade_generate --prompt "..."` — separate from DiT; does not share the DiT forward pass.
- **Advanced optional downloads:** `python setup/download.py models --advanced` to fetch LongCLIP / moondream3 / HPSv3 / DINOv3 / SigLIP2 / Qwen3-8B / Marigold / TAESD / Consistency-Decoder / ConvNeXtV2 / LAION Aesthetic v2 / AnyDoor reference.

---

## Shared building blocks

[`models/dit.py`](../models/dit.py) holds the pieces every DiT reuses: `modulate`, `timestep_embedding`, `TimestepEmbedder`, `FinalLayer`, `LayerScale` (an init of `0` means off: `gamma` starts at 1, never 0, so adaLN-Zero branches still train) and `xavier_init_linears`. Norms are `torch.nn.RMSNorm` and stochastic depth is `timm.layers.DropPath`. `DiT-P*` / `DiT-Supreme*` are `DiT_Text` subclasses in [`models/dit_text.py`](../models/dit_text.py) that only change the block options (QK-norm, full-width SwiGLU, RMSNorm).

### `RAELatentBridge`

| Option | Default | Effect |
|--------|---------|--------|
| **`learnable_output_scale`** | `False` | Scalar **`scale_dit`** / **`scale_rae`** on conv outputs (training stability). |

See also [CODEBASE.md](CODEBASE.md).

---

## Block-wise autoregressive (AR) generation

SDX supports **block-wise autoregressive** self-attention (ACDiT-style): the latent patch grid is split into blocks, and each block can only attend to earlier blocks (and within its own block, in raster order). This combines **diffusion** (noise → clean over time) with **autoregressive structure** (spatial order), which can improve composition and make outputs easier to “fix” by regenerating later blocks.

---

### What it does

- **`num_ar_blocks = 0`** (default): Standard **bidirectional** attention. Every patch can attend to every other patch. Same as typical DiT/FLUX.
- **`num_ar_blocks = 2`**: Grid is split into **2×2 blocks**. Block order is raster (top-left → top-right → bottom-left → bottom-right). Within each block, patches also follow raster order (causal). So the model effectively generates “top-left first”, then “top-right” (seeing top-left), then “bottom-left” (seeing both top blocks), then “bottom-right” (seeing all previous).
- **`num_ar_blocks = 4`**: **4×4 blocks**. Stronger AR bias: 16 blocks in raster order, each block causal within itself. More structure, potentially slower and more constrained.

The **mask** is applied only to **self-attention** over the spatial patches. Cross-attention to the text remains bidirectional (all patches can attend to all text tokens).

---

### Why use AR?

- **Better structure**: The model can commit to a layout (e.g. face in top-left) and then fill in the rest, reducing “averaged” or incoherent compositions.
- **Fixability**: In theory you could regenerate only later blocks (e.g. bottom half) while keeping the top; our current pipeline doesn’t expose block-by-block sampling, but the inductive bias can still help global coherence.
- **Closer to “ChatGPT for images”**: Autoregressive order over space is reminiscent of token-by-token generation in LLMs.

Trade-off: AR adds a **causal constraint**, so the model has less global context when generating later regions. For some datasets and resolutions, **0** (full bidirectional) can still look best; for others, **2** (2×2 blocks) gives a good balance.

---

### How to use

#### Training

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

#### Config

In `config/train_config.py`:

- **`num_ar_blocks: int = 0`** — 0 = off, 2 = 2×2 blocks, 4 = 4×4 blocks. Other values (e.g. 3 for 3×3) are supported by the mask code but are less common.
- **`ar_block_order: str = "raster"`** — macro-block sequence: **`raster`** (row-major), **`zorder`** (Morton), **`snake`** (boustrophedon), or **`spiral`** (outside-in). See [AR_EXTENSIONS.md](#block-wise-autoregressive-ar-generation).

#### Code

- **Mask**: `models/attention.py` → `create_block_causal_mask_2d(h, w, num_ar_blocks, block_order=...)`. Returns an (N, N) float mask with `-inf` where attention is disabled.
- **DiT**: `models/dit_text.py` (`DiT_Text` and its `DiT_Predecessor_Text` / `DiT_Supreme_Text` subclasses) register `_ar_mask` when `num_ar_blocks > 0` and pass it into every block’s self-attention.

#### Inspect masks

```bash
python -m scripts.tools ar_mask_inspect --h 32 --w 32 --blocks 2 --compare
```

---

### Block order

Default macro-block order is **raster**: block `(0,0)`, then `(0,1)`, …, then `(1,0)`, `(1,1)`, … — top-left toward bottom-right. Within each macro-block, patch order is also raster.

Optional alternatives:
- **`zorder`**: Morton / z-order over block indices (2D locality–friendly sequence)
- **`snake`**: alternating row direction traversal
- **`spiral`**: outside-in traversal over macro blocks

Training flag: `--ar-block-order ...`. Details and extra helpers: **[AR_EXTENSIONS.md](#block-wise-autoregressive-ar-generation)**.

#### Runtime curriculum and order-mix

`train.py` also supports runtime AR mutation:

- `--ar-curriculum-mode none|step|linear`
- `--ar-curriculum-warmup-steps ...`
- `--ar-curriculum-ramp-start ... --ar-curriculum-ramp-end ...`
- `--ar-curriculum-start-blocks ... --ar-curriculum-target-blocks ...`
- `--ar-order-mix raster,zorder,snake,spiral`

This lets training start with full bidirectional attention and progressively move to stronger AR, while optionally cycling traversal order to improve robustness.

---

### When to prefer 0 vs 2 vs 4

| Setting | Use when |
|--------|----------|
| **0** | You want standard diffusion behavior; maximum context everywhere; no AR bias. |
| **2** | You want a bit of structural bias and possibly better composition; good default to try if 0 feels “messy”. |
| **4** | You want strong left-to-right / top-to-bottom structure; smaller “chunks” per block, more causal steps. |

Start with **0** or **2**; only try **4** if you’re aiming for a very ordered layout and accept the extra constraint.

---

### Technical note: any patch grid

The stored `_ar_mask` covers the native square grid (e.g. 32×32 patches for 256×256 with patch_size=2). Other grids (non-square aspect ratios, hires-fix, dual-stage layout) get the same block order from `models.dit_text.grid_ar_mask(model, h, w)`, which calls `create_block_causal_mask_2d(h, w, ...)` and caches the result.

---

### ViT scorer alignment (DiT AR ↔ quality / adherence)

The **`ViT/`** module scores finished images. If your **generator** was trained with `num_ar_blocks` 0, 2, or 4, failure modes differ slightly from full bidirectional DiT. To keep ViT scores **calibrated** to the layout the image came from:

1. **Training ViT** (`python -m vit_quality.train`): **AR conditioning is on by default**. Put the same regime on each JSONL row the model should learn:
   - **`num_ar_blocks`**, **`dit_num_ar_blocks`**, or **`ar_blocks`** — integer **`0`**, **`2`**, or **`4`**. Missing or invalid → **unknown** bucket (4th one-hot).
2. **Older ViT checkpoints** (no `use_ar_conditioning` in saved config) load as **text-only** (8-D caption features only). Use **`--no-ar-conditioning`** when training if you need weights compatible with that layout, or retrain with AR fields in the manifest.
3. **Inference** (`python -m vit_quality.infer --out scores.jsonl` and/or `--out-npz embeddings.npz`): If the checkpoint has `use_ar_conditioning: true`, each row’s `num_ar_blocks` / `dit_num_ar_blocks` / `ar_blocks` / `generator_num_ar_blocks` is read (including optional nested blobs like `dit_config`) and fused with caption features. Omitted → **unknown** one-hot unless you pass **`--default-num-ar-blocks 0|2|4`** (e.g. whole manifest is from one DiT run).

4. **Facebook / Meta DiT** (`facebookresearch/DiT`): vanilla checkpoints are **full bidirectional** → use **`num_ar_blocks = 0`** for ViT alignment. Only use `2`/`4` if you trained an SDX-compatible fork with the same block-causal mask API.

5. **Tag manifests from a DiT `.pt`**: `python -m scripts.tools ar_tag_manifest --dit-ckpt path/to/best.pt --manifest-jsonl data/in.jsonl --out data/out.jsonl` (or `--num-ar-blocks 2` without reading a file). Sets `num_ar_blocks`, mirrors `dit_num_ar_blocks`, and adds `ar_regime` (`full_bidirectional`, `block_ar_2x2`, …).

Bridge code: **`utils/modeling/architecture.py`** (`ar_conditioning_vector`, `parse_num_ar_blocks_from_row`, `read_num_ar_blocks_from_checkpoint`, `tag_manifest_row_ar`, `batch_ar_conditioning`).

---

### AR extensions (macro-block order, layout, curriculum)

#### Block visit order (`ar_block_order`)

Training flag (and checkpoint field): **`ar_block_order`**.

| Value | Meaning |
|--------|--------|
| **`raster`** (default) | Macro-blocks ordered row-major: `(0,0)`, `(0,1)`, …, `(1,0)`, … — same as the original implementation. |
| **`zorder`** | Morton / z-order over block indices: tends to preserve **2D locality** between successive blocks (neighbor blocks in space are often closer in the sequence than pure raster). |
| **`snake`** | Boustrophedon row scan: row 0 left→right, row 1 right→left, alternating each row. |
| **`spiral`** | Outside-in spiral over macro-blocks; useful when central composition is important. |

Within each macro-block, patch order stays **raster** (row-major inside the block).

Implementation: `models/attention.py` → `create_block_causal_mask_2d(..., block_order=...)`.

CLI:

```bash
python train.py --data-path /path/to/data --num-ar-blocks 2 --ar-block-order zorder
python train.py --data-path /path/to/data --num-ar-blocks 2 --ar-block-order snake
python train.py --data-path /path/to/data --num-ar-blocks 4 --ar-block-order spiral
```

Inference must match training: `get_dit_build_kwargs` passes `ar_block_order` from the saved `TrainConfig`, so checkpoints trained with `zorder` load the same mask.

#### Related files

| Path | Role |
|------|------|
| [models/attention.py](../models/attention.py) | Mask construction, sparsity stats, optional mask softening helpers. |
| [utils/training/objectives.py](../utils/training/objectives.py) | Schedules for ramping `num_ar_blocks` by step (optional; wire in `train.py` if you use it). |
| `python -m scripts.tools ar_mask_inspect` | CLI: mask shape, allowed-pair fraction, visit order; `--compare` raster vs zorder. |

#### Runtime AR curriculum (train.py)

```bash
python train.py --num-ar-blocks 2 --ar-curriculum-mode step --ar-curriculum-warmup-steps 12000
python train.py --num-ar-blocks 4 --ar-curriculum-mode linear \
  --ar-curriculum-ramp-start 5000 --ar-curriculum-ramp-end 30000
python train.py --num-ar-blocks 2 --ar-order-mix raster,zorder,snake,spiral
```

Note: dynamic AR runtime (`--ar-curriculum-mode != none` or `--ar-order-mix`) disables `torch.compile` in training to avoid graph-stability issues while masks mutate.

The ViT bridge ([utils/modeling/architecture.py](../utils/modeling/architecture.py)) keys off **`num_ar_blocks`** (0 / 2 / 4 / …). It does **not** encode `ar_block_order` today.

---

## Recommended pretrained stack (2026)

Metadata-only guide. **No weights are downloaded by this doc.** Point `pretrained/<Folder>` at hub ids below (or run scaffold / download scripts when you choose to fetch).

Resolution order is implemented in `utils/modeling/model_paths.py`: **local folder first** (including legacy names), then hub fallback.

### Primary upgrades (use these)

| Role | Local folder | Hub id | Replaces |
|------|--------------|--------|----------|
| **T2I text encoder (best Qwen)** | `Qwen3-8B` | `Qwen/Qwen3-8B` | T5-only / older LLM TE for modern DiT/FLUX.2 Klein-style stacks |
| **Prompt LLM** | `Qwen3-14B` | `Qwen/Qwen3-14B` | `Qwen2.5-14B-Instruct` |
| **VLM / caption / Qwen-Image TE path** | `Qwen3-VL-8B-Instruct` | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen2.5-VL-7B-Instruct` |
| **REPA / dense vision** | `DINOv3-ViT-L16` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `DINOv2-Large` |
| **Vision–text align** | `SigLIP2-SO400M` | `google/siglip2-so400m-patch16-384` | `SigLIP-SO400M` |
| **Preference reward** | `HPSv3` | `MizzenAI/HPSv3` | `HPSv2-hf` |
| **Light VLM / creative RAG** | `moondream3-preview` | `moondream/moondream3-preview` | `moondream2` |

### Still recommended (keep)

| Role | Folder | Hub id | Notes |
|------|--------|--------|-------|
| Classic DiT TE | `T5-XXL` | `google/t5-v1_1-xxl` | Default SDX DiT train/sample path until a ckpt is Qwen-conditioned |
| Multilingual TE | `UMT5-XXL` | `google/umt5-xxl` | Optional |
| CLIP fusion (triple/penta) | `CLIP-ViT-L-14`, `CLIP-ViT-bigG-14`, `CLIP-ViT-H-14`, `LongCLIP-L` | OpenAI / LAION / LongCLIP | Unchanged |
| Depth | `Depth-Anything-V2-Large` | `depth-anything/Depth-Anything-V2-Large` | Prefer DA3 when available |
| Depth (2025+) | `Depth-Anything-V3-Large` | `depth-anything/DA3-LARGE-1.1` | New default depth scaffold |
| Seg | `SAM2-Hiera-Large` | `facebook/sam2-hiera-large-hf` | Keep |
| Foundation T2I (external) | `Qwen-Image` / `Qwen-Image-2512` | `Qwen/Qwen-Image*` | Sibling Diffusers stack, not the SDX DiT |
| Foundation T2I (external) | `FLUX.2-klein-9B` / `FLUX.2-dev` | `black-forest-labs/FLUX.2*` | Uses Qwen3-8B as TE in Klein |
| Modern DiT VAE | `DC-AE-f32c32` | `mit-han-lab/dc-ae-f32c32-mix-1.0` | Latent experiments |
| Extra VLMs | `InternVL3-8B`, `Gemma-3-4B-IT` | OpenGVLab / Google | Caption / critique alternatives |

### Why these picks

- **Qwen3-8B** is what current FLUX.2 Klein / Comfy text-encoder workflows standardize on for LLM-as-TE.
- **Qwen3-14B** beats Qwen2.5-14B for prompt expansion / agentic text.
- **Qwen3-VL-8B-Instruct** supersedes Qwen2.5-VL for captioning and multimodal critique; Qwen-Image historically used Qwen2.5-VL-7B as TE — Qwen3-VL is the upgrade path for new work.
- **DINOv3** outperforms DINOv2 / prior SSL on dense features (REPA, critics).
- **SigLIP2** is the current SigLIP line.
- **HPSv3** (ICCV 2025) is the wide-spectrum preference scorer above HPSv2.
- **moondream3-preview** is the current Moondream VLM (MoE, longer context).

### Code entry points

```python
from utils.modeling.model_paths import (
    default_qwen_path,  # Qwen3-14B
    default_qwen3_text_encoder_path,  # Qwen3-8B
    default_qwen_vl_path,  # Qwen3-VL-8B
    default_repa_vision_path,  # DINOv3-L
    default_siglip_path,  # SigLIP2
    default_hps_path,  # prefers HPSv3 (alias: default_hpsv2_path)
    default_moondream_path,  # prefers moondream3 (alias: default_moondream2_path)
    default_depth_anything_path,  # prefers DA3
)
```

Scaffold registry: `utils/modeling/model_paths.py`.  
Status: `python -m scripts.tools.ops.pretrained_status`.

### Fetch later (when you want weights)

```bash
# Config/tokenizer scaffolds only (no weight blobs)
python setup/download.py scaffold --name Qwen3-8B --name Qwen3-14B --name Qwen3-VL-8B-Instruct
python setup/download.py scaffold --name DINOv3-ViT-L16 --name SigLIP2-SO400M --name HPSv3 --name moondream3-preview
python setup/download.py scaffold --name FLUX.2-klein-9B --name Depth-Anything-V3-Large --name DC-AE-f32c32

# Full revolutionary stack (includes weights — heavy)
python setup/download.py stack
```

---

## Native & lower-level libraries — quality, training, prompt adherence

This doc ties **systems-level code** (Rust, Zig, C/C++, Go, and common native deps) to SDX goals: **output quality**, **training throughput/reliability**, and **prompt adherence**. For tools already in `native/`, see [../native/README.md](../native/README.md).

---

### How to use this doc

| Your goal | Jump to |
|-----------|---------|
| Faster / safer dataset QA, manifests, captions | [In-repo native tools](#in-repo-native-tools-sdx) |
| Sharper training data (decode, resize, color) | [Image I/O & preprocessing](#image-io--preprocessing) |
| Stricter captions before T5 | [Text, tokenization & adherence](#text-tokenization--prompt-adherence) |
| Cheaper or portable inference | [Export & runtime](#export--inference-runtimes) |
| Don’t fight PyTorch | [What stays in PyTorch](#what-stays-in-pytorch) |

---

### In-repo native tools (SDX)

These are **optional**; Python paths work without them. They help **data quality** and **operational hygiene**, which indirectly improve trained models and adherence (clean captions, no broken paths, deduped manifests).

| Tool | Lang | Helps with |
|------|------|------------|
| `sdx-jsonl-tools` | Rust | **stats**, **validate**, **prompt-lint**, **image-paths**, **dup-image-paths**, **`file-md5`** (streaming MD5 = Python `hashlib`; used by `manifest quality` dedup), **file-fnv** — caption length, structure, duplicates |
| `sdx-noise-schedule` | Rust | **linear** / **cosine** VP-DDPM schedule CSV (analysis vs Python `diffusion/` schedules) |
| `sdx-linecrc` | Zig | Streaming **FNV-1a** over manifest bytes — cache keys, “did the dataset change?” |
| `sdx-pathstat` | Zig | Fast **path → size / missing** for image lists from Rust **image-paths** |
| `libsdx_latent` | C++ | **Latent grid / patch-token math** (C ABI) — consistent geometry with `dit_variant_compare`, ctypes |
| Mojo `native/mojo` | Mojo + Python | Optional **Modular** kernels / `mojopy` CLI launcher |
| `sdx-manifest` | Go | **JSONL merge** + dedupe by image key |
| `data.manifest_utils` / `scripts.tools prompt_lint` | Python | Zero-build manifest **stats** / **prompt-lint** fallbacks for the Rust tools |

**Python bridge:** `native/_experimental/python/sdx_native/` → the lazy shim `utils/native.py` ([native/README.md](../native/README.md)).

**Wiring:** `scripts/tools/manifest.py` (MD5 dedup tries Rust `file-md5` via `sdx_native.native_tools.file_md5_hex` first; `--no-native-md5` forces Python-only), `manifest paths` / `merge`, `captions preflight`, etc. ([native/README.md § Python integration](../native/README.md#python-integration-repo-root-on-pythonpath)).

---

### Image I/O & preprocessing

**Why it matters for quality:** bad decode, wrong gamma, aggressive JPEG, or inconsistent resize/crop **teaches the wrong image–text alignment**. Training speed also depends on decode + resize being cheap.

| Library / stack | Typical use | Lang / binding |
|-----------------|-------------|----------------|
| **libvips** | Fast resize, thumbnail pipelines, many formats | C with `pyvips` |
| **OpenCV** | Augmentations, color transforms, landmark checks | C++ → `cv2` |
| **libjpeg-turbo** / **MozJPEG** | Correct, fast JPEG encode/decode | C |
| **stb_image** / **image** (Rust) | Embedded decode in custom loaders | C / Rust |
| **WebP / AVIF** stacks | Modern lossy formats for storage | Various |

**Prompt adherence angle:** crop/aspect policies should match what you describe in captions (e.g. “full body” vs tight crop). Keep **one policy** in code and document it next to `data/t2i_dataset.py` crop modes.

---

### Text, tokenization & prompt adherence

**Why it matters:** adherence starts **before** DiT — consistent **normalization**, **caption structure**, and **lint** reduce “model ignores tag #17” effects.

| Library / stack | Typical use | Notes |
|-----------------|-------------|--------|
| **Hugging Face `tokenizers`** | Fast BPE/SentencePiece in Rust | Same family as T5/CLIP tokenizers in Python |
| **sentencepiece** | Unigram/BPE training & encode | C++ core; used across many encoders |
| **ICU** | Unicode normalize, casefold, locale-safe text | C/C++/Java; useful for **multilingual** caption cleanup |
| **RE2** / **Rust `regex`** | Bounded-time patterns for tag extraction | Safer than naive regex on untrusted strings |
| **Aho–Corasick / `aho-corasick` (Rust)** | Many keyword triggers (like `infer_content_controls` ideas) | Fast offline lint of huge JSONL |
| **xxHash / BLAKE3** | Per-caption or per-row fingerprints | Dedup near-dupe captions |

**SDX Python side:** `data/caption_utils.py`, `utils/prompt/content_controls.py`, `utils/training/preference.py` (LADD-style distillation + latent D stubs), `scripts/tools/prompts.py` — native tools complement these with **bulk** JSONL checks, not replacement for model conditioning.

**Implemented in-repo (Python, stdlib + optional xxhash):**

| Module / tool | Role |
|----------------|------|
| [`native/_experimental/python/sdx_native/text_hygiene.py`](../native/_experimental/python/sdx_native/text_hygiene.py) | **NFKC** normalization, zero-width strip, comma-segment trim, **SHA256** or **xxhash** caption fingerprints, pos/neg token overlap helper |
| [`utils/prompt/text.py`](../utils/prompt/text.py) | Prompt comma-tag split/merge/dedupe and the pos/neg conflict filter (pure Python) |
| [`scripts/tools/captions.py`](../scripts/tools/captions.py) | JSONL CLI: `--report-dups`, `--report-overlap`, `--normalize-samples` |
| Training | `train.py --caption-unicode-normalize` → `TrainConfig.caption_unicode_normalize` → `Text2ImageDataset(caption_unicode_normalize=True)` |

---

### Hashing, dedup & storage

| Library | Use |
|---------|-----|
| **Zstandard (`zstd`)** | Compress JSONL shards; fast train-side streaming |
| **SQLite / DuckDB** | Index manifests (path → caption hash, labels) for QA dashboards |
| **FNV / xxHash** (see in-repo Zig/Rust) | Cheap change detection |

---

### Export & inference runtimes

| Runtime | Use |
|---------|-----|
| **ONNX Runtime** | Portable inference for **VAE**, small nets, or exported helpers |
| **TensorRT / OpenVINO** | GPU/CPU-optimized inference (ops-specific) |

These do **not** replace the main PyTorch training loop in SDX today; they are options if you export subgraphs for **edge deployment** or **pre/post pipelines** (e.g. fixed resize graph).

---

### What stays in PyTorch

Keep **conv / attention / diffusion steps** in PyTorch (and vendor kernels it already uses: **cuDNN**, **cuBLAS**, **FlashAttention**-class ops where installed). Rewriting the DiT forward pass in raw C for “speed” is rarely the first win versus:

- better **data** (native QA + image stack above),
- **batching**, **compile** (`torch.compile`), **mixed precision**,
- **latent cache** (`python -m scripts.tools precompute_latents`),
- **caption hygiene** (Rust/Node lint + Python controls).

---

### Integration patterns (recommended)

1. **CLI / subprocess** — Rust/Zig/Go binaries called from Python (already used for JSONL tools). Simplest ops story.
2. **C ABI + `ctypes` / `cffi`** — `libsdx_latent` pattern for small numeric APIs without GIL-heavy Python loops.
3. **PyO3 / maturin** (future) — If a hot path must live in Rust **and** be imported as a module; higher maintenance than CLI.

---

### See also

- [native/README.md](../native/README.md) — build commands and examples  
- [PROMPT_STACK.md](PROMPT_STACK.md) — inference text pipeline  
- [CODEBASE.md](CODEBASE.md) — `data/`, `utils/prompt/`, training entrypoints  
- [MODEL_WEAKNESSES.md](QUALITY.md) — what quality problems look like at the model level  
- [HARDWARE.md](GETTING_STARTED.md#pc-specs-for-training-sdx-consumer-hardware-no-datacenter-gpus) — when I/O and CPU preprocessing matter
