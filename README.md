# SDX: DiT + xformers + AR + PixAI — incredible on your dataset

Text-conditioned **DiT** with **xformers**, **block-wise AR** (ACDiT-style), and **[PixAI](https://pixai.art/en/generator/image)-style prompt adherence**: T5 + cross-attention, tag emphasis, structured captions, fast training. **No reference image** — the model excels on the dataset you give it.

## Features

- **Block-wise AR (ACDiT-style)**: Optional **autoregressive** self-attention over spatial blocks (`--num-ar-blocks 2` or `4`). See [docs/AR.md](docs/AR.md) for 0 vs 2 vs 4 and usage.
- **Strong prompt adherence**: Long/complex captions, T5-XXL + full-sequence cross-attention; excels on the dataset you provide.
- **[PixAI](https://pixai.art/en/generator/image)-style**: **Tag-based prompts**, **(tag)** / **((tag))** emphasis and **[tag]** de-emphasis in captions, subject-first tag order for best adherence (inspired by [PixAI.art](https://pixai.art/en/generator/image) AI Art Generator).
- **Negative prompt**: Model **listens to negative prompt** and tries **really hard not to add** those features (positive − weight×negative).
- **Quality tags**: `masterpiece`, `best quality`, `highres` etc. are **boosted** so they improve output quality (10x-style).
- **No character blending / correct count**: Multi-person prompts get **anti-blending** tags and negative terms so **distinct characters** and **right number of people** (not bare minimum).
- **Fix imperfections during generation**: Model is trained to **fix near-clean images** (refinement: small-*t* samples) so at inference you can run an optional **refinement pass** after generation. If the user wants the raw/fucked look, set **allow_imperfect_output=True** (or `--allow-imperfect` at inference) to skip refinement.
- **Training length: passes, not raw steps/epochs**: Say **how many full passes** over the dataset (e.g. `--passes 3`); trainer converts to steps automatically. **Cosine LR** and **save best** checkpoint so quality keeps improving without the usual “too many epochs = weird/overfit” — optional `--max-steps` caps total steps.
- **Speed & scale**: **xformers** memory-efficient attention (self + cross), `torch.compile`, bf16, gradient checkpointing, TF32, multi-worker DataLoader, DDP.
- **Data**: Folder of images + `.txt`/`.caption`, or JSONL manifest; captions are auto-processed with [PixAI](https://pixai.art/en/generator/image)-style emphasis and tag order.

### Inspired by [PixAI.art](https://pixai.art/en/generator/image) models

SDX takes inspiration from the [PixAI.art](https://pixai.art/en/generator/image) generator and its model lineup. Their site offers **PixAI XL** and **PixAI DiT** families plus named model lines (Haruka, Tsubaki, Hoshino, Nagi, Crystalize, Eternal, Otome, Hinata, Serin, etc.). We mirror that spirit:

| Our `--model` | Inspiration |
|---------------|-------------|
| `DiT-XL/2-Text` | PixAI DiT-style (Tsubaki / Serin family) |
| `DiT-P/2-Text` | PixAI DiT-style large (Tsubaki.2-style), QK-norm + SwiGLU |
| `DiT-P-L/2-Text` | PixAI DiT-style XL with QK-norm/SwiGLU |
| `DiT-L/2-Text`, `DiT-B/2-Text` | Smaller DiT-style bases |

Reference model lines from the site (for naming/preset consistency) are listed in `config/pixai_reference.py` (e.g. Haruka v2, Tsubaki.2, Tsubaki, Hoshino v2, Hoshino, Nagi, Crystalize, Eternal, Otome v2, Tsubaki Flash, Hinata v2, Serin). Training logs show the corresponding PixAI.art-style label for the chosen model.

## Setup

Run all commands from the **repo root** (`sdx/`) so that `config`, `data`, `diffusion`, `models`, and `utils` are on the path.

**PC specs for training (no datacenter):** See **[docs/HARDWARE.md](docs/HARDWARE.md)** for a shopping list (VRAM, CPU, PSU tiers) and **storage estimates** for huge booru-style datasets (Rule34, Danbooru, e621, Gelbooru, etc.) plus GIF/video frame extraction—from 100K to millions of images and how many TB to plan for.

```bash
cd sdx
pip install -r requirements.txt
```

**Gated Hugging Face models:** If you use a gated T5/VAE or private repo, copy `.env.example` to `.env` and set `HF_TOKEN=your_token`. See `.env.example` for optional `CUDA_VISIBLE_DEVICES`.

Optional: clone reference repos into `external/`:

```bash
# Windows (PowerShell)
.\scripts\clone_repos.ps1

# Linux / macOS
./scripts/setup/clone_repos.sh
```

This clones **DiT**, **ControlNet**, **flux**, and **Stability-AI/generative-models** (SD3 reference). Runtime dependencies are pip-only; these repos are for reference. See [docs/INSPIRATION.md](docs/INSPIRATION.md) for what we take from each. For a **roadmap of quality and training improvements**, see [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md). For a **map of all project files and external reference links**, see [docs/FILES.md](docs/FILES.md). For **how config, data, and models connect** (train → checkpoint → sample), see [docs/CONNECTIONS.md](docs/CONNECTIONS.md). For **how generation works** (prompt → text encoding → diffusion loop → DiT → VAE → image), see [docs/HOW_GENERATION_WORKS.md](docs/HOW_GENERATION_WORKS.md). **All docs** are indexed in [docs/README.md](docs/README.md).

**Project layout:** Entrypoints at root are `train.py` and `sample.py`; `inference.py` loads checkpoint for programmatic use. Code lives in `config/`, `models/`, `data/`, `diffusion/`, `utils/`. **`model/`** (singular) is where downloaded T5/VAE/LLM weights go (`scripts/download/download_models.py --all`); **`models/`** (plural) is the Python package (DiT, attention, etc.). Scripts are grouped under `scripts/setup/`, `scripts/download/`, `scripts/training/`, `scripts/tools/`.

## Data format

**Option A – Folder layout**

- `data_path` = directory containing subdirs; each subdir has images and captions.
- For each image `img.png`, place `img.txt` (or `img.caption`) with one caption per file.

**Option B – Manifest (JSONL)**

- One JSON object per line: `{"image_path": "/path/to/img.png", "caption": "your caption"}`.
- Use `--manifest-jsonl /path/to/manifest.jsonl` and leave `--data-path` empty.

**Caption tips ([PixAI](https://pixai.art/en/generator/image)-style)**

- **Tag-style**: comma-separated, e.g. `1girl, long hair, outdoors, sunset`. Dataset applies **PixAI.art-style emphasis**: `(tag)` and `((tag))` are expanded for stronger focus; `[tag]` for de-emphasis. Subject (e.g. 1girl) is moved to the front when possible.
- **Quality tags**: Use tags like `masterpiece`, `best quality`, `highres`, `8k` — they are **boosted** (repeated) so the model strongly improves output quality (10x-style).
- **Negative prompt**: In JSONL use `"negative_caption"` or `"negative_prompt"`; in .txt put the negative on the **second line**. The model is trained to **try really hard not to add** those features.
- **No character blending / correct count**: For multi-person prompts (e.g. `2girls`, `room full of people`), the dataset auto-adds **anti-blending** tags and negative terms so the model learns distinct characters and the right number of people (not the bare minimum).
- **Long/complex**: full sentences; T5 handles long context (ReVe-style).
- **Structured**: subject, setting, style in a fixed order for consistent adherence.
- **3D, realistic, interior/exterior**: Include tags like `3d render`, `photorealistic`, `interior design`, `exterior` in captions — the dataset **boosts these domain tags** so the model learns them well (areas many models struggle with). See [docs/DOMAINS.md](docs/DOMAINS.md) and `config/prompt_domains.py` for recommended prompts and negatives per domain. For **hands, faces, double head, text, and other common model failures**, see [docs/MODEL_WEAKNESSES.md](docs/MODEL_WEAKNESSES.md).

**Quick test:** `python scripts/tools/quick_test.py` runs one DiT forward pass (no data) to verify the env. **Inspect a checkpoint:** `python scripts/tools/ckpt_info.py results/.../best.pt` prints config and step count.

## Training

Single GPU:

```bash
python train.py --data-path /path/to/image_folders --results-dir results
```

Multi-GPU (DDP):

```bash
torchrun --nproc_per_node=4 train.py --data-path /path/to/data --global-batch-size 256
```

With **block-wise AR** (ACDiT-style; see [docs/AR.md](docs/AR.md) for 0 vs 2 vs 4):

```bash
python train.py --data-path /path/to/data --num-ar-blocks 2
```

Disable xformers (e.g. if not installed):

```bash
python train.py --data-path /path/to/data --no-xformers
```

**DiT Predecessor (more and better features):** Use `--model DiT-P/2-Text` for a larger, upgraded transformer that acts as DiT’s predecessor: **QK-norm** attention (LLaMA/FLUX-style), **SwiGLU** MLP, **AdaLN-Zero**, and deeper/wider default (depth 32, hidden 1280, 20 heads). Same conditioning as DiT (text, style, control, negative prompt). Use `DiT-P-L/2-Text` for the same improvements at DiT-XL size (28 layers, 1152 hidden).

**DiT Supreme (best-practice architecture):** Use `--model DiT-Supreme/2-Text` for the top-tier variant: **RMSNorm** (Z-Image/Lumina-style), **QK-norm in both self- and cross-attention**, **SwiGLU**, **AdaLN-Zero**, and optional **size embedding** (`--size-embed-dim`) for multi-resolution. Same XL size (28 layers, 1152 hidden). Use `DiT-Supreme-L/2-Text` for the large version (32 layers, 1280 hidden).

**Training length (passes recommended):**

```bash
# N full passes over the dataset — no guessing steps or epochs
python train.py --data-path /path/to/data --passes 3
```

- **`--passes N`**: Train until the model has seen the whole dataset **N times**. Steps = N × (dataset size ÷ batch size). Same idea whether you have 50K or 5M images.
- Optional **`--max-steps 100000`**: Cap total steps (e.g. 3 passes but at most 100k steps).
- Uses **cosine LR** (warmup then decay to `--min-lr`). Best checkpoint is saved when loss improves (`results/.../best.pt`).
- If you omit `--passes`, you can still use **`--max-steps`** for a raw step limit, or **`--epochs`** for the old epoch-based loop.

**Avoiding overtraining (more training = better quality, not worse):**

Training too long can make outputs worse: repetitive, oversaturated, or worse prompt adherence. We keep "more = better" by:

1. **Cosine LR** — Learning rate decays to `--min-lr` so late training doesn’t overshoot; quality can keep improving instead of blowing up.
2. **EMA** — Inference uses EMA weights (saved in checkpoints); they’re smoother and usually generalize better than the raw step weights.
3. **Best checkpoint** — We save the checkpoint with the best loss so you get the best model, not the last step.
4. **Validation + early stopping** — Use a **validation split** so "best" is by **validation loss** (generalization), and stop when val loss stops improving:
   - `--val-split 0.05` — hold out 5% for validation.
   - `--val-every 2000` — evaluate val loss every 2000 steps.
   - `--early-stopping-patience 3` — stop after 3 val checks with no improvement; `best.pt` is the best by val loss.

Example: train with passes but stop as soon as quality (val loss) plateaus:

```bash
python train.py --data-path /path/to/data --passes 5 --val-split 0.05 --val-every 2000 --early-stopping-patience 3
```

Then use **`best.pt`** for inference (it’s the checkpoint that generalized best, not the overtrained tail).

**Refinement (fix imperfections during generation):**

- Training: `--refinement-prob 0.25` (default) trains the model on small-*t* so it learns to fix near-clean images. Use `--refinement-max-t 150` to cap refinement timesteps.
- Inference: By default we **refine output** (one extra denoise pass) to fix artifacts. If the user wants the **raw / imperfect look**, set `allow_imperfect_output=True` in config or use `--allow-imperfect` in `inference.py`.

Use a smaller/faster text encoder for debugging (e.g. T5-Large):

- Set in `config/train_config.py`: `text_encoder = "google/t5-v1_1-xl"` and use `DiT-L/2-Text` (which expects 768-dim; for XL/XXL you’d keep 4096). For a quick test, you can add a `DiT-L/2-Text` variant that uses 768 and change the config to T5-XL (1024) or T5-Large (768). For now the code assumes T5-XXL 4096 for DiT-XL/2-Text.

## Data (JSONL) fields

| Field | Description |
|-------|-------------|
| `caption` | Positive prompt (required) |
| `negative_caption` / `negative_prompt` | What to avoid (model tries hard not to add these) |
| `style` | Style prompt (e.g. "oil painting, vivid"); blended with `style_strength` so output isn’t sloppy |
| `control_image` / `control_path` | Path to control image (depth/edge/pose); use `control_scale` to blend structure |
| `init_image` / `init_image_path` / `source_image` | For img2img training: path to source image (model learns to edit from it with `--img2img-prob`) |

For .txt captions: first line = positive, second line = negative prompt.

## Options

| Flag | Default | Description |
|------|--------|-------------|
| `--data-path` | (required*) | Root folder for image/caption data |
| `--manifest-jsonl` | None | JSONL manifest; overrides data-path |
| `--negative-prompt-weight` | 0.5 | How strongly to subtract negative prompt conditioning |
| `--model` | DiT-XL/2-Text | DiT-XL/2-Text, DiT-L/2-Text, DiT-B/2-Text, DiT-P/2-Text, DiT-P-L/2-Text, **DiT-Supreme/2-Text**, **DiT-Supreme-L/2-Text** |
| `--image-size` | 256 | Train resolution (will be latent_size = size/8) |
| `--global-batch-size` | 128 | Total batch across all GPUs |
| `--passes` | 0 | Number of full passes over the dataset (recommended; replaces guessing steps/epochs) |
| `--max-steps` | 0 | Cap when using passes, or raw step limit when passes=0 |
| `--epochs` | 100 | Epochs (only when passes=0 and max-steps=0) |
| `--lr` | 1e-4 | Learning rate |
| `--num-workers` | 8 | DataLoader workers |
| `--no-bf16` | False | Disable bf16 |
| `--no-compile` | False | Disable torch.compile |
| `--no-grad-checkpoint` | False | Disable gradient checkpointing |
| `--num-ar-blocks` | 0 | Block-wise AR: 0=off, 2 or 4 (ACDiT-style) |
| `--no-xformers` | False | Disable xformers (use PyTorch SDPA fallback) |
| `--min-lr` | 1e-6 | Min LR for cosine schedule |
| `--refinement-prob` | 0.25 | Prob of training on “fix imperfections” (small t) |
| `--refinement-max-t` | 150 | Max t for refinement training |
| `--no-save-best` | False | Disable saving best checkpoint by loss |
| `--beta-schedule` | linear | linear or cosine |
| `--prediction-type` | epsilon | epsilon or v (velocity) |
| `--noise-offset` | 0 | SD-style noise offset (e.g. 0.1) |
| `--min-snr-gamma` | 5 | Min-SNR loss weighting (0=off) |
| `--resume` | None | Resume from checkpoint path |
| `--val-split` | 0 | Fraction for validation (e.g. 0.05); enables best-by-val and early stopping |
| `--val-every` | 2000 | Evaluate val loss every N steps (when val-split > 0) |
| `--early-stopping-patience` | 0 | Stop after N val checks with no improvement; 0 = off |
| `--val-max-batches` | None | Max val batches per eval (default: full val set) |
| `--deterministic` | False | Reproducible training |
| `--latent-cache-dir` | None | Use precomputed latents (faster training) |
| `--img2img-prob` | 0 | When init_image in data, use it as x_start with this prob (img2img training) |
| `--mdm-mask-ratio` | 0 | Enables MDM training: fraction of latent patches to mask (0=off) |
| `--mdm-mask-schedule` | None | Optional state-dependent schedule for MDM mask ratio: `t_step,mask_ratio,...` |
| `--mdm-patch-size` | 2 | Latent patch size for MDM masking (typically 2, matches DiT patch embed) |
| `--mdm-min-mask-patches` | 1 | Ensure at least N patches masked per sample (avoids empty-mask) |
| `--no-mdm-loss-only-masked` | False | If set, includes unmasked regions in the loss (default is masked-only) |
| `--moe-num-experts` | 0 | MoE DiT upgrade: number of FFN experts (0=off) |
| `--moe-top-k` | 2 | MoE routing: top-k experts per token |
| `--moe-balance-loss-weight` | 0 | Router aux loss weight (0=off) |

**Inference:**  
`python sample.py --ckpt .../best.pt --prompt "..." --negative-prompt "..." --steps 50 --width 256 --height 256 --out out.png`  
Optional: `--cfg-scale`, `--cfg-rescale`, `--scheduler ddim|euler`, `--num N` (batch), `--grid`, `--no-cache` (disable T5 cache), `--deterministic`, `--style`, `--auto-style-from-prompt` (extract style/artist from prompt), `--control-image`, `--lora`, `--init-image` (img2img), `--mask` (inpainting), `--inpaint-mode legacy|mdm`, `--sharpen`, `--contrast`. Use `(word)` and `[word]` in the prompt for emphasis (1.2×) / de-emphasis (0.8×). High CFG auto-enables rescale/threshold (see docs). **Prompt:** `--prompt-file path`, `--subject-first` (reorder comma-separated). **Less AI-looking:** `--naturalize` (anti-plastic negative + natural-look prefix + film grain post-process). **Tags:** use `--tags "1girl, long hair, outdoors"` or `--tags-file path/to/tags.txt` (subject-first order); prompt is optional when tags are provided. **LoRAs:** `--lora path.safetensors` or `path.pt:0.6`; use `--lora-trigger "style name"` to prepend the LoRA’s trigger word(s) to the prompt for better activation. Use `--save-prompt` to write a `.txt` sidecar with prompt, seed, steps next to the output.  
Also supports prompt controls in `sample.py`: `--gender-swap`, `--anatomy-scale longer|bigger|wider`, `--object-scale longer|bigger|wider`, `--scene-scale longer|bigger|wider`, and `--character-sheet` (inject a custom character spec).

Optional OCR text repair: `--expected-text "OPEN" --ocr-fix` (uses pytesseract to validate, then runs `--mask` + `--inpaint-mode mdm` until OCR accuracy passes `--ocr-threshold` or `--ocr-iters`).

`--character-sheet` expects a JSON file with optional keys like:
```json
{
  "appearance": ["long black hair", "green eyes"],
  "clothing": ["hoodie", "jacket"],
  "accessories": ["earrings", "necklace"],
  "style_tags": ["manga style", "clean lineart"],
  "negative": ["deformed", "wrong scale"],
  "gender_presentation": "androgynous"
}
```

`python inference.py --ckpt results/.../best.pt` — use `--allow-imperfect` to skip refinement (raw output).

### Tools, repos, and inspiration

See **[docs/INSPIRATION.md](docs/INSPIRATION.md)** for the full list of reference repos and the [PixAI.art](https://pixai.art/en/generator/image) (website) model lineup.

### Features for better images and user control

- **CFG rescale / dynamic threshold** — (Not exposed in CLI; internal sampling uses fixed behavior.)
- **Negative prompt**: Model tries hard not to add unwanted content.
- **Post-process**: `--sharpen`, `--contrast` in sample.py.
- **Quality tags** in data: Boost `masterpiece`, `best quality` so the model learns better output.
- **Aesthetic/sample weighting**: JSONL `weight` or `aesthetic_score` so high-quality samples count more in loss.
- **Refinement**: Train on small-t and optional refinement pass to fix imperfections.

### Styles, ControlNet, LoRA (no sloppy blending)

- **Style**: Train with `--style-embed-dim 4096` (same as T5-XXL) and a `style` field in JSONL. At inference use `--style "your style text" --style-strength 0.7`, or use `--auto-style-from-prompt` to extract style/artist from the prompt (e.g. "by X", "style of X", artist tags from PixAI/Danbooru). See `config/style_artists.py` and [docs/STYLE_ARTIST_TAGS.md](docs/STYLE_ARTIST_TAGS.md). Style is added to text conditioning with a fixed strength so it doesn’t overpower the prompt.
- **ControlNet**: Train with `--control-cond-dim 1` and a `control_image` path in JSONL. At inference use `--control-image path.png --control-scale 0.85`. Control features are added to the latent patch grid with a scale so structure is followed without making the image messy.
- **LoRA**: At inference only: `--lora path.pt path.safetensors path2.pt:0.6`. Supports `.pt` and `.safetensors`; each LoRA can have its own scale (default 0.8). Use `--lora-trigger "trigger word"` to prepend the LoRA’s trigger to the prompt. Use moderate scales (e.g. 0.5–0.8) so multiple LoRAs don’t conflict.
- **Blending**: Keep style_strength and control_scale in the recommended ranges (0.6–0.8 and 0.7–1.0). Don’t stack too many LoRAs at high scale; 0.5–0.8 per LoRA keeps output clean.

### Img2img, inpainting, from-z (FLUX / NoobAI / illust-style)

- **Img2img**: `--init-image path.png --strength 0.75`. Encodes the image to latent, noises it to timestep `t = strength * num_timesteps`, then denoises. Higher strength = more change (1.0 ≈ full denoise from that image).
- **Inpainting**: `--init-image ref.png --mask mask.png`. Mask: white = region to inpaint, black = keep. Default `--inpaint-mode legacy`; use `--inpaint-mode mdm` to freeze the known (black) regions at every denoise step for cleaner structure.
- **From-z**: `--init-latent z.pt --strength 0.8`. Start from a saved latent (e.g. VAE-encoded image). Useful for chaining or custom pipelines.
- **Output size**: `--width 512 --height 768`. Resize the decoded image to this size (model still runs at trained resolution).
- **Training img2img**: In JSONL add `"init_image": "/path/to/source.png"`. Train with `--img2img-prob 0.2` so the model sometimes sees noised *init* image as input and learns to edit (FLUX/NoobAI/illust-style).

### Book / manga workflow (multi-image)

Generate a cover + multiple pages using inpainting coherence (freeze face region across pages) and optional OCR-based text fixing. New: optional `--anchor-speech-bubbles` to freeze approximate speech-bubble outlines so panel text doesn’t drift.
See `scripts/book/generate_book.py` for the CLI.

Quick example (one prompt per page; optional per-page expected text):
```powershell
python scripts/book/generate_book.py --ckpt "C:\path\best.pt" `
  --output-dir out_book `
  --book-type manga --model-preset anime `
  --prompts-file pages.txt `
  --expected-text "OPEN" --ocr-fix --ocr-iters 2 `
  --anchor-face --edge-anchor --anchor-speech-bubbles `
  --speech-bubble-anchor-inner-dilate 2 --speech-bubble-anchor-outer-dilate 18
```

`pages.txt` can optionally override expected OCR text per page:
`your page prompt here|||OPEN`

## SD / SDXL-style features and fixes

Same training and inference tricks used in Stable Diffusion and SDXL are supported:

| Feature | Flag / config | Description |
|--------|----------------|-------------|
| **Offset noise** | `--noise-offset` (e.g. 0.1) | Shifts noise for better light/dark balance (SD/SDXL) |
| **Min-SNR loss weighting** | `--min-snr-gamma` (e.g. 5, 0=off) | Downweights easy timesteps so quality doesn’t degrade |
| **V-prediction** | `--prediction-type v` | Train with velocity target (SD2-style) instead of epsilon |
| **Cosine schedule** | `--beta-schedule cosine` | Cosine beta schedule instead of linear |
Sampling uses a fixed DDIM-style loop with cond/uncond. Use `sample.py --prompt "..." --negative-prompt "..." --steps N --width W --height H`. Optional: **--num N** (batch images), **--vae-tiling** (lower VRAM for large decode), **--cfg-rescale** (e.g. 0.7 to reduce oversaturation).

## Extra features (make it really good)

| Feature | Description |
|--------|-------------|
| **Resume training** | `--resume results/0-DiT-XL-2-Text/best.pt` — loads model, EMA, optimizer, step count and continues. |
| **Aesthetic / sample weighting** | In JSONL add `"weight"` or `"aesthetic_score"` (float). Loss is weighted so high-quality samples count more. |
| **Deterministic training** | `--deterministic` — seeds dataloader workers and RNG for reproducible runs. |
| **Crop mode** | `--crop-mode center|random|largest_center` — training image crop (center; random aug; resize-to-cover then center). |
| **Caption dropout schedule** | `--caption-dropout-schedule 0,0.2,10000,0.05` — decay caption dropout over steps (structure first, then prompt adherence). |
| **Polyak checkpoint** | `--save-polyak N` — running average of last N steps saved as `polyak.pt` every `ckpt-every`. |
| **WandB / TensorBoard** | `--wandb-project NAME`, `--tensorboard-dir DIR` — log loss and LR every `log-every`. |
| **Dry run** | `--dry-run` — run 1 training step and exit (verify setup). |
| **Log sample images** | `--log-images-every N` and `--log-images-prompt "..."` — log a generated sample to WandB/TensorBoard every N steps. |
| **Reproducibility** | `--deterministic` in train.py and sample.py; see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). |
| **Data quality** | `python scripts/tools/data_quality.py manifest.jsonl --dedup phash --min-caption-len 10 --out filtered.jsonl` — dedup, caption length, bad-words, min-weight. |
| **Export ONNX** | `python scripts/tools/export_onnx.py path/to/best.pt [--out dit.onnx] [--dynamic-batch]`. |
| **Latent cache** | Precompute autoencoder latents once, then train without encoding: `python scripts/training/precompute_latents.py --data-path ... --out-dir latent_cache`, then `--latent-cache-dir latent_cache` (add `--autoencoder-type rae` if you are precomputing RAE latents). |
| **ViT-Gen tokens + attention upgrades** | (ViT-style efficiency/features) `--num-register-tokens N` adds learnable register tokens; `--use-rope` enables RoPE; `--kv-merge-factor K` pools KV for faster attention (K>1); `--token-routing-enabled` enables soft per-token gating (tune with `--token-routing-strength`). |
| **ViT-Gen SSM swap** | Replace every Nth self-attention block with an SSM-like token mixer: `--ssm-every-n N` (tune `--ssm-kernel-size`). |
| **REPA (Representation Alignment)** | Optional training auxiliary loss aligning DiT representations to a frozen vision encoder (DINOv2/CLIP): `--repa-weight W` plus `--repa-encoder-model`, `--repa-out-dim`. |
| **AdaGen early exit (sampling)** | Faster sampling by exiting when latent deltas get small: `sample.py --ada-early-exit --ada-exit-delta-threshold ... --ada-exit-patience ... --ada-exit-min-steps ...`. |
| **PBFM edge/high-pass drift (sampling)** | Heuristic edge guidance: `sample.py --pbfm-edge-boost 0.1 --pbfm-edge-kernel 3` (tune boost). |
| **T5 + Autoencoder (VAE/RAE) + CLIP + LLM** | **Better generation and understanding:** `python scripts/download/download_models.py --all` downloads T5-XXL, T5-XL, T5-Large (prompt understanding), SD VAE checkpoints (decode quality), CLIP ViT-L/14 (optional dual-encoder), and SmolLM + Qwen2.5-7B (prompt expansion) into `model/`. Use `--t5`, `--vae`, `--clip`, `--llm`, `--llm-best` to pick. Train/sample with `--text_encoder model/T5-XXL` (best) or `model/T5-XL` / `model/T5-Large` (lighter). For RAE, set `--autoencoder-type rae` and point `--vae-model` to a diffusers RAE checkpoint. |
| **Prompt LLM only** | Or use `python scripts/download/download_llm.py` (360M) / `python scripts/download/download_llm.py --best` (Qwen2.5-7B) with `--local-dir model/SmolLM2-360M-Instruct` etc.; files go under `model/` (in .gitignore). |
| **Runnable sampler** | `python sample.py --ckpt results/.../best.pt --prompt "your prompt" --steps 50 --width 256 --height 256 --out out.png`; optional `--tags "tag1, tag2"` or `--tags-file path` (subject-first), `--lora path.safetensors` / `--lora-trigger "word"`, `--cfg-scale`, `--cfg-rescale`, `--num N`, `--grid`, `--vae-tiling`, `--deterministic`, `--style`, `--control-image`, img2img/inpainting, `--sharpen`, `--contrast`, `--no-neg-filter`, `--boost-quality`, `--preset`, `--op-mode`, `--originality`. By default `sample.py` runs a small **refinement pass**; disable with `--no-refine` (or tune with `--refine-t`). Empty negative uses Civitai-style default. For **text in image**: "sign that says OPEN" and `--text-in-image`. See [docs/CIVITAI_QUALITY_TIPS.md](docs/CIVITAI_QUALITY_TIPS.md). |
| **Export to safetensors** | `python scripts/tools/export_safetensors.py path/to/best.pt [--out model.safetensors] [--metadata]` for ComfyUI / A1111. |
| **Img2img / inpainting** | `--init-image ref.png --strength 0.75` or `--init-image ref.png --mask mask.png` for inpainting |
| **From-z** | `--init-latent z.pt` to start from a saved latent; `--width` / `--height` to resize output |

## Project layout

```
sdx/
├── config/
│   ├── train_config.py   # TrainConfig (incl. num_ar_blocks, use_xformers)
│   └── pixai_reference.py # [PixAI.art](https://pixai.art/en/generator/image) model lines & style labels
├── docs/
│   ├── CIVITAI_QUALITY_TIPS.md  # Oversaturation, blur, bad hands, resolution (Civitai-style)
│   └── INSPIRATION.md    # Tools, repos, and features for better images
├── data/
│   ├── t2i_dataset.py    # Text2ImageDataset + PixAI.art-style emphasis/tag order
│   └── caption_utils.py # apply_pixai_emphasis, normalize_tag_order
├── diffusion/
│   └── gaussian_diffusion.py
├── model/                # Downloaded models (scripts/download/download_models.py --all); in .gitignore
│   ├── T5-XXL/, T5-XL/, T5-Large/   # Text encoders (XXL best; XL/Large lighter)
│   ├── sd-vae-ft-mse/, sd-vae-ft-ema/, sdxl-vae/, sdxl-vae-fp16-fix/  # VAEs (decode quality) (RAEs are different checkpoints)
│   ├── CLIP-ViT-L-14/   # Optional (future T5+CLIP dual-encoder)
│   ├── SmolLM2-360M-Instruct/
│   └── Qwen2.5-7B-Instruct/
├── models/
│   ├── dit.py            # Base DiT (Meta)
│   ├── dit_text.py       # DiT + T5 cross-attn, xformers, block AR
│   └── attention.py      # xformers/SDPA, block-causal mask
├── scripts/
│   ├── setup/            # clone_repos.ps1, clone_repos.sh → external/
│   ├── download/        # download_models.py, download_llm.py → model/
│   ├── training/        # precompute_latents.py, self_improve.py
│   └── tools/           # ckpt_info.py, export_safetensors.py, quick_test.py
├── train.py
├── sample.py
├── inference.py         # Optional refinement pass (separate module); --allow-imperfect to skip (raw output)
├── utils/
│   └── quality.py       # Post-process: sharpen, contrast (ComfyUI/A1111-style)
├── requirements.txt
└── README.md
```

## References

- DiT: [Scalable Diffusion Models with Transformers (Meta)](https://github.com/facebookresearch/DiT) — `external/DiT`
- ControlNet: [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) — structural conditioning; `external/ControlNet`
- FLUX: [black-forest-labs/flux](https://github.com/black-forest-labs/flux) — modern diffusion reference; `external/flux`
- Stability: [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) — SD3 reference; `external/generative-models`
- **PixAI.art**: [PixAI.art](https://pixai.art/en/generator/image) — AI art generator **website** (not a GitHub repo). We take tag-style prompts and emphasis from it; it is unrelated to PixArt-alpha (a separate T5+DiT research repo).
- Strong prompt adherence on your dataset (long/complex captions)
