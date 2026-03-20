# File map: SDX project and external reference repos

All **SDX project files** (what we run and edit) and **key files in external repos** (reference only; clone with `scripts/setup/clone_repos.ps1` or `scripts/setup/clone_repos.sh`). SDX does **not** import from `external/` at runtime.

---

## SDX project files (repo root: `sdx/`)

Run commands from repo root so `config`, `data`, `diffusion`, `models`, `utils` are on the path.

### Root

| File | Description |
|------|-------------|
| [README.md](../README.md) | Project overview, setup, data format, training, options. |
| [requirements.txt](../requirements.txt) | Pip dependencies (torch, transformers, diffusers, xformers, etc.). |
| [train.py](../train.py) | Training script: DiT + T5, passes/epochs/steps, val split, early stopping, DDP; **--crop-mode**, **--caption-dropout-schedule**, **--save-polyak**, **--wandb-project**, **--tensorboard-dir**, **--dry-run**. |
| [inference.py](../inference.py) | Load checkpoint and config for programmatic inference. |
| [sample.py](../sample.py) | CLI sampling: prompt, negative prompt, steps, width, height; **--cfg-scale**, **--num N** (batch), **--grid**, **--vae-tiling**, **--cfg-rescale**, **--deterministic** (reproducible); style, control, lora, img2img, sharpen, contrast. High CFG auto-enables rescale/threshold. |

### Config

| File | Description |
|------|-------------|
| [config/__init__.py](../config/__init__.py) | Exports `TrainConfig`. |
| [config/train_config.py](../config/train_config.py) | TrainConfig + get_dit_build_kwargs(cfg): single place for DiT build args (used by train, sample, inference, self_improve). |
| [config/pixai_reference.py](../config/pixai_reference.py) | PixAI.art-style model labels for logs (Haruka, Tsubaki, etc.). |
| [config/prompt_domains.py](../config/prompt_domains.py) | Recommended prompts/negatives for 3D, realistic, interior, exterior. |

### Data

| File | Description |
|------|-------------|
| [data/__init__.py](../data/__init__.py) | Exports `Text2ImageDataset`, `collate_t2i`. |
| [data/t2i_dataset.py](../data/t2i_dataset.py) | Dataset: folder or JSONL, image + caption, latent cache, PixAI-style emphasis. |
| [data/caption_utils.py](../data/caption_utils.py) | Caption processing: tag order, emphasis, quality boost, anti-blending. |

### Diffusion

| File | Description |
|------|-------------|
| [diffusion/__init__.py](../diffusion/__init__.py) | Exports `create_diffusion`, `GaussianDiffusion`. |
| [diffusion/gaussian_diffusion.py](../diffusion/gaussian_diffusion.py) | Diffusion: beta schedule, training losses (min-SNR, v/epsilon), DDIM sampling, CFG rescale, dynamic threshold. |

### Models

| File | Description |
|------|-------------|
| [models/__init__.py](../models/__init__.py) | Exports DiT models (XL/2-Text, P/2-Text, P-L/2-Text). |
| [models/dit.py](../models/dit.py) | Base DiT (patch embed, timestep embed, adaLN blocks); from Meta DiT. |
| [models/dit_text.py](../models/dit_text.py) | T5-conditioned DiT (cross-attention, caption). |
| [models/dit_predecessor.py](../models/dit_predecessor.py) | DiT-P: QK-norm, SwiGLU, AdaLN-Zero (PixAI-style predecessor). |
| [models/attention.py](../models/attention.py) | Attention with xformers / SDPA fallback. |
| [models/controlnet.py](../models/controlnet.py) | ControlNet conditioning (control image + scale). |
| [models/lora.py](../models/lora.py) | LoRA layers (optional). |

### Utils

| File | Description |
|------|-------------|
| [utils/__init__.py](../utils/__init__.py) | Package init. |
| [utils/quality.py](../utils/quality.py) | Post-process: sharpen (unsharp mask), contrast. |
| [utils/prompt_lint.py](../utils/prompt_lint.py) | Prompt adherence linting for SDX JSONL (pos/neg overlap + caption heuristics). |
| [utils/image_quality_metrics.py](../utils/image_quality_metrics.py) | Pure-PIL/numpy image QC metrics (sharpness + contrast). |

### Scripts

Scripts are grouped by purpose: **setup/** (clone repos), **download/** (T5/VAE/LLM), **training/** (precompute latents, self-improve), **tools/** (inspect, smoke test).

| File | Description |
|------|-------------|
| [scripts/setup/clone_repos.ps1](../scripts/setup/clone_repos.ps1) | Windows: clone DiT, ControlNet, flux, generative-models, PixArt, Z-Image, SiT, Lumina into `external/`. |
| [scripts/setup/clone_repos.sh](../scripts/setup/clone_repos.sh) | Linux/macOS: same clones. |
| [scripts/download/download_models.py](../scripts/download/download_models.py) | Download best HF models: T5-XXL (text encoder), VAEs (sd-vae-ft-mse, sdxl-vae, sdxl-vae-fp16-fix), LLMs (SmolLM, Qwen2.5-7B). Use `--all` or `--t5` / `--vae` / `--llm` / `--llm-best`. |
| [scripts/download/download_llm.py](../scripts/download/download_llm.py) | Download a single LLM for prompt expansion (SmolLM2-360M or Qwen2.5-7B with `--best`). |
| [scripts/training/self_improve.py](../scripts/training/self_improve.py) | Self-improvement loop (8.6): generate images, caption with VLM, write manifest.jsonl. |
| [scripts/training/precompute_latents.py](../scripts/training/precompute_latents.py) | Precompute VAE latents for faster training. |
| [scripts/tools/ckpt_info.py](../scripts/tools/ckpt_info.py) | Inspect checkpoint: print config, steps, best_loss (no full model load). |
| [scripts/tools/data_quality.py](../scripts/tools/data_quality.py) | Filter/dedup JSONL or folder: `--dedup phash|md5`, `--min-caption-len`, `--bad-words`, `--min-weight` (IMPROVEMENTS 1.6). |
| [scripts/tools/prompt_lint.py](../scripts/tools/prompt_lint.py) | Prompt adherence lint for SDX JSONL (empty captions, token heuristics, pos/neg overlap). |
| [scripts/tools/tag_coverage.py](../scripts/tools/tag_coverage.py) | Scan a JSONL manifest for hard-style/person/anatomy/concept-bleed tag coverage. |
| [scripts/tools/spatial_coverage.py](../scripts/tools/spatial_coverage.py) | Scan a JSONL manifest for spatial-wording coverage (`behind`, `next to`, `under`, `left of`, ...). |
| [scripts/tools/op_preflight.py](../scripts/tools/op_preflight.py) | One-shot “coverage + thresholds” gate (PASS/FAIL) before training. |
| [scripts/tools/op_pipeline.ps1](../scripts/tools/op_pipeline.ps1) | Windows wrapper to run preflight + normalize/boost (+ optional train/eval). |
| [scripts/tools/complex_prompt_coverage.py](../scripts/tools/complex_prompt_coverage.py) | Coverage analyzer for tricky categories (clothes/weapons/food/text/foreground/background/weird/NSFW). |
| [scripts/tools/prompt_gap_scout.py](../scripts/tools/prompt_gap_scout.py) | Analyze one prompt and suggest missing tricky category keywords. |
| [scripts/tools/export_onnx.py](../scripts/tools/export_onnx.py) | Export DiT from .pt to ONNX for deployment (optional `--dynamic-batch`). |
| [scripts/tools/export_safetensors.py](../scripts/tools/export_safetensors.py) | Export .pt checkpoint DiT weights to .safetensors (ComfyUI/A1111); optional `--metadata` for config JSON. |
| [scripts/tools/quick_test.py](../scripts/tools/quick_test.py) | Smoke test: one DiT forward pass to verify env. |
| [scripts/tools/image_quality_qc.py](../scripts/tools/image_quality_qc.py) | Image QC for JSONL: Laplacian sharpness + grayscale contrast; optional fail thresholds. |

### Docs

| File | Description |
|------|-------------|
| [docs/CIVITAI_QUALITY_TIPS.md](CIVITAI_QUALITY_TIPS.md) | Civitai-style fixes: oversaturation, blur, bad hands, resolution; sample.py flags and training tips. |
| [docs/STYLE_ARTIST_TAGS.md](STYLE_ARTIST_TAGS.md) | Style/artist tags from PixAI, Danbooru, Gelbooru: extraction, training, `--auto-style-from-prompt`. |
| [docs/GENERATION_DIAGRAM.md](GENERATION_DIAGRAM.md) | Flowchart: text → T5 → diffusion loop (DiT) → VAE → image; optional img2img, control, LoRA (Mermaid + ASCII). |
| [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Same-seed/same-run: `--deterministic`, `--seed`, CUBLAS (sampling and training). |
| [docs/INSPIRATION.md](INSPIRATION.md) | What we take from PixAI, ComfyUI, and cloned repos; optional deps. |
| [docs/IMPROVEMENTS.md](IMPROVEMENTS.md) | Roadmap: quality, fixes, and features from other SD/DiT/FLUX models. |
| [docs/HARDWARE.md](HARDWARE.md) | PC specs, VRAM, storage for training and full booru scrape. |
| [docs/AR.md](AR.md) | Block-wise autoregressive (AR): 0 vs 2 vs 4 blocks, raster order, when to use. |
| [docs/CONNECTIONS.md](CONNECTIONS.md) | How config, data, and models connect: TrainConfig → checkpoint → sample/inference; get_dit_build_kwargs; data flow. |
| [docs/HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) | How generation works: prompt → T5 → diffusion loop → DiT denoising → VAE decode → image. |
| [docs/DOMAINS.md](DOMAINS.md) | 3D, realistic, interior/exterior: how we handle hard-to-generate domains. |
| [docs/FILES.md](FILES.md) | This file: project file map and external reference links. |

---

## External repos (reference only)

Cloned into `external/` by [scripts/setup/clone_repos.ps1](../scripts/setup/clone_repos.ps1) or [scripts/setup/clone_repos.sh](../scripts/setup/clone_repos.sh). **Do not import from these at runtime.**

### 1. DiT (Meta) — base transformer

- **Clone:** `https://github.com/facebookresearch/DiT` → `external/DiT`
- **What we use:** Patch embed, timestep embed, adaLN blocks; architecture reference for our [models/dit.py](../models/dit.py) and [models/dit_text.py](../models/dit_text.py).

| Path under `external/DiT/` | Description |
|----------------------------|-------------|
| [models.py](../external/DiT/models.py) | DiT block, patchify, adaLN. |
| [diffusion/gaussian_diffusion.py](../external/DiT/diffusion/gaussian_diffusion.py) | Diffusion utils. |
| [diffusion/respace.py](../external/DiT/diffusion/respace.py) | Timestep respacing. |
| [train.py](../external/DiT/train.py) | Reference training loop. |
| [sample.py](../external/DiT/sample.py) | Reference sampling. |

---

### 2. ControlNet — structural conditioning

- **Clone:** `https://github.com/lllyasviel/ControlNet` → `external/ControlNet`
- **What we use:** Control image (depth/edge/pose) conditioning; reference for our [models/controlnet.py](../models/controlnet.py) and control scale.

| Path under `external/ControlNet/` | Description |
|-----------------------------------|-------------|
| [cldm/model.py](../external/ControlNet/cldm/model.py) | ControlNet + SD backbone. |
| [ldm/models/diffusion/ddim.py](../external/ControlNet/ldm/models/diffusion/ddim.py) | DDIM sampling. |
| [ldm/models/diffusion/sampling_util.py](../external/ControlNet/ldm/models/diffusion/sampling_util.py) | Dynamic threshold (norm_thresholding). |
| [docs/train.md](../external/ControlNet/docs/train.md) | Training ControlNet. |

---

### 3. FLUX (Black Forest Labs) — modern diffusion

- **Clone:** `https://github.com/black-forest-labs/flux` → `external/flux`
- **What we use:** Sampling, guidance, img2img/fill; structural conditioning; resolution/aspect ideas.

| Path under `external/flux/` | Description |
|------------------------------|-------------|
| [src/flux/model.py](../external/flux/src/flux/model.py) | Transformer, guidance embed. |
| [src/flux/sampling.py](../external/flux/src/flux/sampling.py) | Denoise loop, timesteps, guidance. |
| [src/flux/modules/conditioner.py](../external/flux/src/flux/modules/conditioner.py) | CLIP/T5 conditioning. |
| [src/flux/modules/autoencoder.py](../external/flux/src/flux/modules/autoencoder.py) | VAE. |
| [docs/structural-conditioning.md](../external/flux/docs/structural-conditioning.md) | Control/docs. |
| [docs/fill.md](../external/flux/docs/fill.md) | Inpainting/fill. |

---

### 4. Stability generative-models — SD3 / MM-DiT

- **Clone:** `https://github.com/Stability-AI/generative-models` → `external/generative-models`
- **What we use:** SD3 / MM-DiT architecture and training reference.

| Path under `external/generative-models/` | Description |
|------------------------------------------|-------------|
| (repo structure varies) | Look for diffusion, DiT, or SD3 modules; official Stability reference. |

---

### 5. PixArt-alpha — PixArt-α (T5 + DiT)

- **Clone:** `https://github.com/PixArt-alpha/PixArt-alpha` → `external/PixArt-alpha`
- **What we use:** T5 text encoder + DiT; efficient T2I training reference.

---

### 6. PixArt-sigma — PixArt-Σ (4K T2I)

- **Clone:** `https://github.com/PixArt-alpha/PixArt-sigma` → `external/PixArt-sigma`
- **What we use:** 4K generation, weak-to-strong training, token compression attention.

---

### 7. Z-Image — S3-DiT (single-stream)

- **Clone:** `https://github.com/Tongyi-MAI/Z-Image` → `external/Z-Image`
- **What we use:** Single-stream DiT (text + image in one sequence); efficient scaling reference.

---

### 8. SiT — Scalable Interpolant Transformers

- **Clone:** `https://github.com/willisma/SiT` → `external/SiT`
- **What we use:** Flow matching + DiT backbone; interpolant framework; sampling/convergence ideas.

---

### 9. Lumina-T2X — Lumina-T2I / Next-DiT

- **Clone:** `https://github.com/Alpha-VLLM/Lumina-T2X` → `external/Lumina-T2X`
- **What we use:** Next-DiT scaling; multi-resolution; Rectified Flow; T2I pipeline reference.

---

## Quick links (same doc)

- **Training:** [train.py](../train.py) · [config/train_config.py](../config/train_config.py)
- **Sampling:** [sample.py](../sample.py) · [inference.py](../inference.py)
- **Diffusion:** [diffusion/gaussian_diffusion.py](../diffusion/gaussian_diffusion.py) · [diffusion/respace.py](../diffusion/respace.py) · [diffusion/sampling_utils.py](../diffusion/sampling_utils.py) · [diffusion/loss_weighting.py](../diffusion/loss_weighting.py)
- **Models:** [models/dit_text.py](../models/dit_text.py) · [models/dit_predecessor.py](../models/dit_predecessor.py) · [models/pixart_blocks.py](../models/pixart_blocks.py) (SizeEmbedder, ported from PixArt)
- **Data:** [data/t2i_dataset.py](../data/t2i_dataset.py) · [data/caption_utils.py](../data/caption_utils.py)
- **Docs:** [README](../README.md) · [INSPIRATION](INSPIRATION.md) · [IMPROVEMENTS](IMPROVEMENTS.md) · [HARDWARE](HARDWARE.md)
