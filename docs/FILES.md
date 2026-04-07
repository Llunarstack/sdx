# File map: SDX project and external reference repos

All **SDX project files** (what we run and edit) and **key files in external repos** (reference only; clone with `scripts/setup/clone_repos.ps1` or `scripts/setup/clone_repos.sh`). SDX does **not** import from `external/` at runtime.

---

## SDX project files (repo root: `sdx/`)

Run commands from repo root so `config`, `data`, `diffusion`, `models`, `utils` are on the path.

**Quick orientation:** [CODEBASE.md](CODEBASE.md) — layers, repo tree, entry points, `scripts/` layout, contribution rules.

### How the tree fits together (runtime)

| Area | Role | Consumed by |
|:-----|:-----|:------------|
| **config/** | `TrainConfig`, `get_dit_build_kwargs`, presets | `train.py`, `sample.py`, checkpoints |
| **data/** | `Text2ImageDataset`, captions | `train.py` |
| **diffusion/** | `GaussianDiffusion`, schedules, loss weights | `train.py`, `sample.py` |
| **models/** | DiT, ControlNet, MoE, RAE bridge, optional cascaded / multimodal **scaffolds** | `train.py`, `sample.py` |
| **utils/** | Checkpoint load, text-encoder bundle, REPA helpers, QC, metrics | `train.py`, `sample.py`, scripts |
| **vit_quality/** | Canonical ViT scoring / prompt tools (**not** the DiT generator) | CLI + optional dataset QA |
| **ViT/** | Legacy compatibility package for existing scripts/imports | Backward compatibility |
| **scripts/** | Download, tools, cascade stub | Ops & CI |
| **pipelines/** | **image_gen** vs **book_comic**: per–product-line docs; book script at `pipelines/book_comic/scripts/generate_book.py` | See [pipelines/README.md](../pipelines/README.md) |
| **native/** | Fast JSONL / manifest helpers (Rust, Go, Node, …) | Optional; see `native/README.md`, ecosystem map [NATIVE_AND_SYSTEM_LIBS.md](NATIVE_AND_SYSTEM_LIBS.md) |
| **toolkit/** | QoL Python modules: env report, JSONL digest, seeds, timers | [toolkit/README.md](../toolkit/README.md); `python -m toolkit.training.env_health` |
| **datasets/** | **Your** datasets: `datasets/train/` + sidecar captions (see [datasets/README.md](../datasets/README.md)) | `train.py --data-path datasets/train` |
| **model/** | Downloaded HF weights (gitignored) | Resolved via `utils/modeling/model_paths.py` |

End-to-end flow: **manifest/images → train.py (T5/triple + VAE/RAE + DiT + diffusion) → checkpoint → sample.py → image**. See [README § Architecture and pipeline](../README.md#architecture-and-pipeline).

### Root

| File | Description |
|------|-------------|
| [README.md](../README.md) | Project overview, setup, data format, training, options. |
| [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | **Auto-generated** ASCII tree — run `python -m scripts.tools update_project_structure` to refresh. |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | PR checklist: ruff format/check, manual sanity, docs. |
| [pyproject.toml](../pyproject.toml) | Ruff settings; minimal `[project]` metadata. |
| [requirements.txt](../requirements.txt) | Pip dependencies (torch, transformers, diffusers, xformers, etc.). |
| [requirements-cuda128.txt](../requirements-cuda128.txt) | Optional: reinstall torch/torchvision/xformers from PyTorch’s **cu128** index after `requirements.txt` (avoids CPU-only PyPI torch). |
| [.editorconfig](../.editorconfig) | Editor defaults (indent, UTF-8, final newline). |
| [train.py](../train.py) | Training: DiT + T5 (optional **triple** CLIP fusion via `--text-encoder-mode triple`), VAE/RAE, REPA, passes/epochs, val, DDP; **--crop-mode**, **--caption-dropout-schedule**, **--save-polyak**, **--wandb-project**, **--tensorboard-dir**, **--dry-run**. |
| [inference.py](../inference.py) | Load checkpoint and config for programmatic inference. |
| [sample.py](../sample.py) | CLI sampling: prompt, negative prompt, steps, width, height; **--cfg-scale**, **--num N** (batch), **--grid**, **--vae-tiling**, **--cfg-rescale**, **--deterministic** (reproducible); style, control, lora, img2img, sharpen, contrast. High CFG auto-enables rescale/threshold. |
| [scripts/cli.py](../scripts/cli.py) | Optional CLI entry (analyze dataset, validate config, checkpoints, …). |

### Config

| File | Description |
|------|-------------|
| [config/README.md](../config/README.md) | Folder layout: `train_config` plus canonical catalogs under `config/defaults/`. |
| [config/__init__.py](../config/__init__.py) | Exports `TrainConfig`, `get_dit_build_kwargs`, `DEFAULT_NEGATIVE_PROMPT`. |
| [config/train_config.py](../config/train_config.py) | TrainConfig + `get_dit_build_kwargs(cfg)`: DiT build args; **`text_encoder_mode`**, **`clip_text_encoder_*`**, RAE/REPA fields. |
| [config/defaults/](../config/defaults/) | **Canonical** prompt catalogs, presets, labels (domains, styles, `sample.py` presets, PixAI names). |
| [config/prompt_domains.py](../config/prompt_domains.py) | Shim → `defaults/prompt_domains.py` (stable import path). |
| [config/style_artists.py](../config/style_artists.py) | Shim → `defaults/style_artists.py`. |
| [config/model_presets.py](../config/model_presets.py) | Shim → `defaults/model_presets.py`. |
| [config/pixai_reference.py](../config/pixai_reference.py) | Shim → `defaults/pixai_reference.py`. |

### Data

| File | Description |
|------|-------------|
| [data/__init__.py](../data/__init__.py) | Exports `Text2ImageDataset`, `collate_t2i`. |
| [data/t2i_dataset.py](../data/t2i_dataset.py) | Dataset: folder or JSONL, image + caption, latent cache, PixAI-style emphasis. |
| [data/caption_utils.py](../data/caption_utils.py) | Caption processing: tag order, emphasis, quality boost, anti-blending. |

### Diffusion

| File | Description |
|------|-------------|
| [diffusion/README.md](../diffusion/README.md) | Folder layout: schedules, `losses/`, sampling, utils. |
| [diffusion/__init__.py](../diffusion/__init__.py) | Exports `create_diffusion`, `GaussianDiffusion`, `get_beta_schedule`, `get_timestep_loss_weight`, `sample_training_timesteps`. |
| [diffusion/schedules.py](../diffusion/schedules.py) | VP beta schedules: linear, cosine, sigmoid, squaredcos_cap_v2 (Improved DDPM–style). |
| [diffusion/losses/](../diffusion/losses/) | **Canonical** timestep loss code: `loss_weighting.py`, `timestep_loss_weight.py`. |
| [diffusion/loss_weighting.py](../diffusion/loss_weighting.py) | Shim → `losses/loss_weighting.py`. |
| [diffusion/timestep_loss_weight.py](../diffusion/timestep_loss_weight.py) | Shim → `losses/timestep_loss_weight.py`. |
| [diffusion/snr_utils.py](../diffusion/snr_utils.py) | NumPy SNR / `alpha_cumprod` helpers for analysis. |
| [diffusion/gaussian_diffusion.py](../diffusion/gaussian_diffusion.py) | Diffusion: beta schedule, training losses (min-SNR, v/epsilon), DDIM sampling, CFG rescale, dynamic threshold. |
| [diffusion/respace.py](../diffusion/respace.py) | Timestep respacing for sampling. |
| [diffusion/sampling_utils.py](../diffusion/sampling_utils.py) | Thresholding helpers. |
| [diffusion/timestep_sampling.py](../diffusion/timestep_sampling.py) | Training-time `t` distributions: uniform, logit-normal (SD3-style discrete), high-noise Beta bias. |
| [diffusion/cascaded_multimodal_pipeline.py](../diffusion/cascaded_multimodal_pipeline.py) | Optional **cascaded** stage-1 → stage-2 forward + optional RAE bridge (scaffold; not wired into default `train.py` loop). |

### Models

| File | Description |
|------|-------------|
| [models/__init__.py](../models/__init__.py) | Exports DiT registry, `DiT_XL_2_Text`, EnhancedDiT, **RAELatentBridge**, **NativeMultimodalTransformer**, **CascadedMultimodalDiffusion**. |
| [models/dit.py](../models/dit.py) | Base DiT (patch embed, timestep embed, adaLN blocks); from Meta DiT. |
| [models/dit_text.py](../models/dit_text.py) | T5-conditioned DiT (cross-attention, caption); ViT-Gen options (RoPE, registers, KV-merge). |
| [models/dit_text_variants.py](../models/dit_text_variants.py) | Canonical module for DiT-P / Supreme text variants (QK-norm, SwiGLU, AdaLN-Zero, REPA). |
| [models/dit_predecessor.py](../models/dit_predecessor.py) | Legacy module path kept for compatibility. |
| [models/pixart_blocks.py](../models/pixart_blocks.py) | SizeEmbedder, ZeroInitPatchChannelGate, etc. |
| [models/rae_latent_bridge.py](../models/rae_latent_bridge.py) | RAE ↔ 4ch DiT latent **1×1** bridge + cycle loss. |
| [models/model_enhancements.py](../models/model_enhancements.py) | **RMSNorm**, **DropPath**, **TokenFiLM**, **SE1x1** — shared blocks for fusion / conditioning. |
| [models/native_multimodal_transformer.py](../models/native_multimodal_transformer.py) | Vision + text fusion: self-attn stack, optional **cross-attn**, **RMSNorm** out, **FiLM** on vision; see [MODEL_STACK.md](MODEL_STACK.md). |
| [models/cascaded_multimodal_diffusion.py](../models/cascaded_multimodal_diffusion.py) | Two-stage DiT + optional bridge wrapper. |
| [models/attention.py](../models/attention.py) | Attention with xformers / SDPA fallback. |
| [models/controlnet.py](../models/controlnet.py) | ControlNet conditioning (control image + scale). |
| [models/lora.py](../models/lora.py) | LoRA layers (optional). |
| [models/moe.py](../models/moe.py) | MoE FFN / routing (optional). |
| [models/enhanced_dit.py](../models/enhanced_dit.py) | EnhancedDiT variants (large). |

### Utils

| File | Description |
|------|-------------|
| [utils/__init__.py](../utils/__init__.py) | Re-exports `quality` helpers. |
| [utils/checkpoint/checkpoint_loading.py](../utils/checkpoint/checkpoint_loading.py) | `load_dit_text_checkpoint`: DiT + config + RAE bridge + **`text_encoder_fusion`** state. |
| [utils/modeling/model_paths.py](../utils/modeling/model_paths.py) | Resolve `model/` local paths vs HF ids (`default_t5_path`, CLIP, DINOv2, Qwen, Cascade). |
| [utils/modeling/text_encoder_bundle.py](../utils/modeling/text_encoder_bundle.py) | **Triple** text stack: T5 + CLIP-L + CLIP-bigG + trainable fusion. |
| [utils/analysis/llm_client.py](../utils/analysis/llm_client.py) | Optional Qwen (or HF causal LM) for prompt expansion. |
| [utils/quality/quality.py](../utils/quality/quality.py) | Post-process: sharpen (unsharp mask), contrast; **naturalize** (human-art style). |
| [utils/prompt/prompt_lint.py](../utils/prompt/prompt_lint.py) | Prompt adherence linting for SDX JSONL (pos/neg overlap + caption heuristics). |
| [utils/prompt/prompt_emphasis.py](../utils/prompt/prompt_emphasis.py) | `( )` / `[ ]` prompt emphasis → per-T5-token weights; shared by `sample.py` and **`train.py --train-prompt-emphasis`**. |
| [utils/prompt/originality_augment.py](../utils/prompt/originality_augment.py) | **`--originality`** / **`--train-originality-prob`**: inject `ORIGINALITY_POSITIVE_TOKENS` after subject tags for fresher compositions. |
| [utils/image_quality_metrics.py](../utils/image_quality_metrics.py) | Pure-PIL/numpy image QC metrics (sharpness + contrast). |
| [utils/training/config_validator.py](../utils/training/config_validator.py) | Train config validation. |
| [utils/checkpoint/checkpoint_manager.py](../utils/checkpoint/checkpoint_manager.py) | Checkpoint rotation / save helpers. |
| [utils/training/metrics.py](../utils/training/metrics.py) | FLOPs / logging helpers. |
| [utils/architecture/dit_architecture.py](../utils/architecture/dit_architecture.py) | DiT / EnhancedDiT profiling: param counts, default build kwargs, variant lists. |
| [utils/architecture/ar_block_conditioning.py](../utils/architecture/ar_block_conditioning.py) | Canonical DiT **block-AR** regime ↔ ViT bridge: JSONL parsing + 4-D one-hot (`num_ar_blocks` 0/2/4 / unknown); see [AR.md](AR.md). |
| [native/python/sdx_native/latent_geometry.py](../native/python/sdx_native/latent_geometry.py) | Latent / DiT **patch token** math (pure Python; matches `native/cpp` C ABI). |
| [native/python/sdx_native/text_hygiene.py](../native/python/sdx_native/text_hygiene.py) | Caption **NFKC** + zero-width strip, fingerprints (SHA256 / optional xxhash), pos/neg overlap; training flag `--caption-unicode-normalize`. |
| [native/python/sdx_native/native_tools.py](../native/python/sdx_native/native_tools.py) | Optional **`native/`** tool discovery (Rust/Zig/Go/Node), FNV manifest fingerprints, JSONL merge, ctypes `libsdx_latent`. |
| [native/python/sdx_native/latent_geometry.py](../native/python/sdx_native/latent_geometry.py) | Canonical Python bridge module for latent geometry helpers (`sdx_native.latent_geometry`). |
| [native/python/sdx_native/native_tools.py](../native/python/sdx_native/native_tools.py) | Canonical Python bridge module for native helper discovery/runtime wrappers (`sdx_native.native_tools`). |
| [utils/modeling/nn_inspect.py](../utils/modeling/nn_inspect.py) | Generic module tree + per-child parameter summary for any `nn.Module`. |
| [utils/quality/test_time_pick.py](../utils/quality/test_time_pick.py) | CLIP/edge/OCR best-of-N scoring for sampling. |
| [utils/generation/orchestration.py](../utils/generation/orchestration.py) | Named **Designer / Verifier / Reasoner** pipeline roles (`PipelineRole`, `pipeline_roles`) — docs + future orchestration; see [LANDSCAPE_2026.md](LANDSCAPE_2026.md). |
| [utils/architecture/architecture_map.py](../utils/architecture/architecture_map.py) | **2026 architecture themes** → SDX parity (`THEMES`, `theme_by_id`, `themes_as_dict`); see [LANDSCAPE_2026.md](LANDSCAPE_2026.md), [BLUEPRINTS.md](BLUEPRINTS.md). |
| *(other `utils/*.py`)* | Advanced inference, anatomy, character consistency, multimodal stubs, etc. |

### ViT (`vit_quality/`)

| File | Description |
|------|-------------|
| [vit_quality/__init__.py](../vit_quality/__init__.py) | Canonical ViT quality/adherence package export surface. |
| [vit_quality/train.py](../vit_quality/train.py), [vit_quality/infer.py](../vit_quality/infer.py) | Canonical module paths for train/score tooling. |
| [vit_quality/checkpoint_utils.py](../vit_quality/checkpoint_utils.py) | Canonical import path for checkpoint loading/reporting helpers. |
| [ViT/README.md](../ViT/README.md) | Legacy compatibility docs for `ViT/` shims/launchers; canonical runtime path is `vit_quality/`. |

### Toolkit (`toolkit/`)

| File | Description |
|------|-------------|
| [toolkit/README.md](../toolkit/README.md) | Index: env health, manifest digest, seeds, timing, suggested optional deps. |
| [toolkit/training/env_health.py](../toolkit/training/env_health.py) | `python -m toolkit.training.env_health` — torch/CUDA/cuDNN + optional libs + `sdx_native` status. |
| [toolkit/training/seed_utils.py](../toolkit/training/seed_utils.py) | `seed_everything`, `worker_seed_fn` for DataLoader workers. |
| [toolkit/quality/manifest_digest.py](../toolkit/quality/manifest_digest.py) | `python -m toolkit.quality.manifest_digest` — JSONL line/key stats; optional Rust `stats`. |
| [toolkit/qol/timing.py](../toolkit/qol/timing.py) | `StepTimer`, `@timed` for step/sec and ETA hints. |
| [toolkit/libs/optional_imports.py](../toolkit/libs/optional_imports.py) | `describe_optional_libs()` for install hints. |
| [toolkit/extras/requirements-suggested.txt](../toolkit/extras/requirements-suggested.txt) | Commented optional pip packages (xxhash, rich, …). |

### Native (`native/`)

Optional compiled CLIs (Rust, Go, Zig, C++, Node) for fast JSONL — **not** imported by Python training by default. See [native/README.md](../native/README.md) and [NATIVE_AND_SYSTEM_LIBS.md](NATIVE_AND_SYSTEM_LIBS.md) (quality / training / adherence).

### Pipelines (`pipelines/`)

| Path | Description |
|------|-------------|
| [pipelines/README.md](../pipelines/README.md) | Index: **image_gen** (general T2I) vs **book_comic** (multi-page, OCR). |
| [pipelines/image_gen/README.md](../pipelines/image_gen/README.md) | Training pointers for general image generation. |
| [pipelines/book_comic/README.md](../pipelines/book_comic/README.md) | Book/comic/manga workflow; canonical [generate_book.py](../pipelines/book_comic/scripts/generate_book.py). |
| [pipelines/book_comic/book_helpers.py](../pipelines/book_comic/book_helpers.py) | Presets, pick-best + CFG flags for `sample.py`, post-process (quality.py). |
| [pipelines/book_comic/prompt_lexicon.py](../pipelines/book_comic/prompt_lexicon.py) | Comic/manga **style** snippets, merged negatives, aspect presets, tategaki/SFX hints. |
| [docs/BOOK_COMIC_TECH.md](BOOK_COMIC_TECH.md) | Sequential-art techniques vs SDX + **best output checklist** (data, training, `generate_book`, pick-best, post). |
| [scripts/tools/book_scene_split.py](../scripts/tools/book_scene_split.py) | `## Page` / `---PAGE---` → `pages.txt` for `generate_book.py`. |
| [scripts/book/generate_book.py](../scripts/book/generate_book.py) | Thin launcher → `pipelines/book_comic/scripts/generate_book.py`. |

### Scripts

Index: **[scripts/README.md](../scripts/README.md)**. **Tools:** **[scripts/tools/README.md](../scripts/tools/README.md)** · dispatcher **`python -m scripts.tools <command>`** ([`scripts/tools/__main__.py`](../scripts/tools/__main__.py)). Subdirs: **setup/** (clone repos), **download/** (T5/VAE/LLM, optional stacks), **training/** (precompute latents, self-improve), **tools/** (inspect, smoke test), **book/** (launcher only), **enhanced/** (EnhancedDiT train/sample — [scripts/enhanced/README.md](../scripts/enhanced/README.md)), **root** (e.g. Stable Cascade stub).

| File | Description |
|------|-------------|
| [scripts/enhanced/train_enhanced.py](../scripts/enhanced/train_enhanced.py) | Train **EnhancedDiT** (optional path; main path is `train.py`). |
| [scripts/enhanced/sample_enhanced.py](../scripts/enhanced/sample_enhanced.py) | Sample EnhancedDiT checkpoints when `sample.py` is incompatible. |
| [scripts/enhanced/setup_enhanced.py](../scripts/enhanced/setup_enhanced.py) | Optional enhanced-stack setup / checks. |
| [scripts/enhanced/save_model_checkpoint.py](../scripts/enhanced/save_model_checkpoint.py) | Save initialized Enhanced DiT-XL checkpoint. |
| [scripts/setup/clone_repos.ps1](../scripts/setup/clone_repos.ps1) | Windows: clone DiT, ControlNet, flux, generative-models, PixArt, Z-Image, SiT, Lumina into `external/`. |
| [scripts/setup/clone_repos.sh](../scripts/setup/clone_repos.sh) | Linux/macOS: same clones. |
| [scripts/download/download_models.py](../scripts/download/download_models.py) | Download best HF models: T5-XXL (text encoder), VAEs (sd-vae-ft-mse, sdxl-vae, sdxl-vae-fp16-fix), LLMs (SmolLM, Qwen2.5-7B). Use `--all` or `--t5` / `--vae` / `--llm` / `--llm-best`. |
| [scripts/download/download_llm.py](../scripts/download/download_llm.py) | Download a single LLM for prompt expansion (SmolLM2-360M or Qwen2.5-7B with `--best`). |
| [scripts/download/download_revolutionary_stack.py](../scripts/download/download_revolutionary_stack.py) | Bulk HF snapshot downloads for extended stacks (see `docs/MODEL_STACK.md`). |
| [scripts/cascade_generate.py](../scripts/cascade_generate.py) | **Stable Cascade** (diffusers) sampling — optional path; uses `pretrained/StableCascade-*` via `utils/model_paths`. |
| — | **self_improve.py** (planned, not in `scripts/training/` on this branch) — see [IMPROVEMENTS.md](IMPROVEMENTS.md) §8.6; use `hf_download_and_train.py` for similar loops. |
| [scripts/training/precompute_latents.py](../scripts/training/precompute_latents.py) | Precompute VAE latents for faster training. |
| [scripts/training/hf_export_to_sdx_manifest.py](../scripts/training/hf_export_to_sdx_manifest.py) | HF `datasets` → `manifest.jsonl` + images (Danbooru-style when schema fits). |
| [scripts/training/hf_download_and_train.py](../scripts/training/hf_download_and_train.py) | One-shot: HF export + `DiT-B/2-Text` train; `--demo` for synthetic data only. |
| [docs/DANBOORU_HF.md](DANBOORU_HF.md) | Using Hugging Face Danbooru-related data with SDX. |
| [scripts/tools/dev/ckpt_info.py](../scripts/tools/dev/ckpt_info.py) | Inspect checkpoint: print config, steps, best_loss (no full model load). |
| [scripts/tools/data/data_quality.py](../scripts/tools/data/data_quality.py) | Filter/dedup JSONL or folder: `--dedup phash|md5`, `--min-caption-len`, `--bad-words`, `--min-weight` (IMPROVEMENTS 1.6). |
| [scripts/tools/data/caption_hygiene.py](../scripts/tools/data/caption_hygiene.py) | JSONL caption Unicode hygiene: `--normalize-samples`, `--report-dups`, `--report-overlap` ([`sdx_native.text_hygiene`](../native/python/sdx_native/text_hygiene.py)). |
| [scripts/tools/data/ar_tag_manifest.py](../scripts/tools/data/ar_tag_manifest.py) | Tag JSONL with DiT `num_ar_blocks` + `ar_regime` from `.pt` or explicit flag (ViT AR alignment; [AR.md](AR.md)). |
| [scripts/tools/data/manifest_paths.py](../scripts/tools/data/manifest_paths.py) | List image paths from JSONL (**Rust `image-paths` / `dup-image-paths`** when built); pipe to Zig **`sdx-pathstat`** for file sizes. |
| [scripts/tools/prompt/prompt_lint.py](../scripts/tools/prompt/prompt_lint.py) | Prompt adherence lint for SDX JSONL (empty captions, token heuristics, pos/neg overlap). |
| [scripts/tools/preview_generation_prompt.py](../scripts/tools/preview_generation_prompt.py) | Preview final pos/neg after `content_controls` + neg filter (no model load). |
| [scripts/tools/prompt/tag_coverage.py](../scripts/tools/prompt/tag_coverage.py) | Scan a JSONL manifest for hard-style/person/anatomy/concept-bleed tag coverage. |
| [scripts/tools/spatial_coverage.py](../scripts/tools/spatial_coverage.py) | Scan a JSONL manifest for spatial-wording coverage (`behind`, `next to`, `under`, `left of`, ...). |
| [scripts/tools/ops/op_preflight.py](../scripts/tools/ops/op_preflight.py) | One-shot “coverage + thresholds” gate (PASS/FAIL) before training. |
| [scripts/tools/training_timestep_preview.py](../scripts/tools/training_timestep_preview.py) | ASCII histograms for `timestep_sample_mode` (uniform / logit_normal / high_noise). |
| [scripts/tools/dit_variant_compare.py](../scripts/tools/dit_variant_compare.py) | Table of DiT / EnhancedDiT parameter counts and weight-size estimates (no training). |
| [scripts/tools/make_smoke_dataset.py](../scripts/tools/make_smoke_dataset.py) | Synthetic PNGs + captions for smoke-testing `train.py`. |
| [SMOKE_TRAINING.md](SMOKE_TRAINING.md) | Step-by-step minimal training commands (`--dry-run`, low VRAM tips). |
| [scripts/tools/vit_inspect.py](../scripts/tools/vit_inspect.py) | ViT checkpoint: config, param count, optional module tree. |
| [scripts/tools/op_pipeline.ps1](../scripts/tools/op_pipeline.ps1) | Windows wrapper to run preflight + normalize/boost (+ optional train/eval). |
| [scripts/tools/complex_prompt_coverage.py](../scripts/tools/complex_prompt_coverage.py) | Coverage analyzer for tricky categories (clothes/weapons/food/text/foreground/background/weird/NSFW). |
| [scripts/tools/prompt_gap_scout.py](../scripts/tools/prompt_gap_scout.py) | Analyze one prompt and suggest missing tricky category keywords. |
| [scripts/tools/export/export_onnx.py](../scripts/tools/export/export_onnx.py) | Export DiT from .pt to ONNX for deployment (optional `--dynamic-batch`). |
| [scripts/tools/export/export_safetensors.py](../scripts/tools/export/export_safetensors.py) | Export .pt checkpoint DiT weights to .safetensors (ComfyUI/A1111); optional `--metadata` for config JSON. |
| [scripts/tools/dev/quick_test.py](../scripts/tools/dev/quick_test.py) | Smoke test: one DiT forward pass to verify env. |
| [scripts/tools/image_quality_qc.py](../scripts/tools/image_quality_qc.py) | Image QC for JSONL: Laplacian sharpness + grayscale contrast; optional fail thresholds. |
| [scripts/tools/repo/update_project_structure.py](../scripts/tools/repo/update_project_structure.py) | Regenerate **`PROJECT_STRUCTURE.md`** at repo root. |
| [scripts/tools/repo/verify_doc_links.py](../scripts/tools/repo/verify_doc_links.py) | Verify relative markdown links in key docs (CI-friendly). |

### Docs

| File | Description |
|------|-------------|
| [docs/QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md) | **Merged:** Civitai-style fixes + community issue matrix; sample.py flags and training tips. |
| [docs/STYLE_ARTIST_TAGS.md](STYLE_ARTIST_TAGS.md) | Style/artist tags from PixAI, Danbooru, Gelbooru: extraction, training, `--auto-style-from-prompt`. |
| [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Same-seed/same-run: `--deterministic`, `--seed`, CUBLAS (sampling and training). |
| [docs/INSPIRATION.md](INSPIRATION.md) | What we take from PixAI, ComfyUI, and cloned repos; optional deps. |
| [docs/IMPROVEMENTS.md](IMPROVEMENTS.md) | Roadmap: quality, fixes, and features from other SD/DiT/FLUX models. |
| [docs/HARDWARE.md](HARDWARE.md) | PC specs, VRAM, storage for training and full booru scrape. |
| [docs/AR.md](AR.md) | Block-wise autoregressive (AR): 0 vs 2 vs 4 blocks, raster order, when to use. |
| [docs/TRAINING_TEXT_TO_PIXELS.md](TRAINING_TEXT_TO_PIXELS.md) | Training mental model: text encoder tokens vs DiT patch tokens; alignment, originality levers. |
| [docs/HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) | **Merged:** Mermaid/ASCII pipeline, step-by-step generation, config/checkpoint/data wiring (§13). |
| [docs/PROMPT_STACK.md](PROMPT_STACK.md) | Inference **text** path before T5: `content_controls`, `neg_filter`, key flags, preview CLI. |
| [docs/NATIVE_AND_SYSTEM_LIBS.md](NATIVE_AND_SYSTEM_LIBS.md) | Lower-level libs: in-repo `native/` tools + ecosystem (image I/O, tokenization, QA) vs quality / training / adherence. |
| [docs/MODEL_STACK.md](MODEL_STACK.md) | Local **`model/`** paths (T5, CLIP, DINOv2, Cascade, …) + **model enhancements** (RMSNorm, FiLM, cross-attn, cascade blend, RAE scales). |
| [docs/LANDSCAPE_2026.md](LANDSCAPE_2026.md) | **Merged 2026 hub:** industry snapshot, post-diffusion themes, workflow + disclaimers. |
| [docs/BLUEPRINTS.md](BLUEPRINTS.md) | **Merged** flow/solvers/distillation + prompt-accuracy blueprints. |
| [docs/PROMPT_COOKBOOK.md](PROMPT_COOKBOOK.md) | Copy-paste `sample.py` recipes (presets, quality, book workflows). |
| [docs/DOMAINS.md](DOMAINS.md) | 3D, realistic, interior/exterior: how we handle hard-to-generate domains. |
| [docs/REGION_CAPTIONS.md](REGION_CAPTIONS.md) | JSONL **`parts`** / **`region_captions`**: merge regional labels into T5 text for layout-aware training. |
| [docs/HF_DATASET_SHORTLIST.md](HF_DATASET_SHORTLIST.md) | Curated shortlist from provided Hugging Face dataset links (primary/secondary/optional + suggested mix). |
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
- **Diffusion:** [diffusion/gaussian_diffusion.py](../diffusion/gaussian_diffusion.py) · [diffusion/respace.py](../diffusion/respace.py) · [diffusion/sampling_utils.py](../diffusion/sampling_utils.py) · [diffusion/loss_weighting.py](../diffusion/loss_weighting.py) · [diffusion/timestep_sampling.py](../diffusion/timestep_sampling.py)
- **Models:** [models/dit_text.py](../models/dit_text.py) · [models/dit_text_variants.py](../models/dit_text_variants.py) · [models/pixart_blocks.py](../models/pixart_blocks.py) (SizeEmbedder, ported from PixArt)
- **Data:** [data/t2i_dataset.py](../data/t2i_dataset.py) · [data/caption_utils.py](../data/caption_utils.py)
- **Docs:** [README](../README.md) · [REGION_CAPTIONS](REGION_CAPTIONS.md) · [MODEL_STACK](MODEL_STACK.md) · [INSPIRATION](INSPIRATION.md) · [IMPROVEMENTS](IMPROVEMENTS.md) · [HARDWARE](HARDWARE.md)
- **Weights / paths:** [docs/MODEL_STACK.md](MODEL_STACK.md) · [utils/modeling/model_paths.py](../utils/modeling/model_paths.py)
