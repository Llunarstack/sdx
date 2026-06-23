# SDX codebase guide — files, roles, and how they connect

This document explains **what lives where**, **what each area does**, and **how modules call each other** across the SDX repository. It is written for onboarding, debugging, and planning changes.

For complementary references:

| Resource | Use when you need… |
|----------|-------------------|
| **[CODEBASE.md](CODEBASE.md)** | Short layers table, contribution rules, “where to change what” |
| **[FILES.md](FILES.md)** | Per-file one-liners for most Python modules |
| **[../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)** | Machine-generated full directory tree (`python -m scripts.tools update_project_structure`) |
| **[HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md)** | Training/sampling pipeline diagrams and checkpoint wiring |
| **[PROMPT_STACK.md](PROMPT_STACK.md)** | Text path before T5 (inference + training caption parity) |

---

## 1. Mental model: three runtime stories

SDX is one **text-to-image diffusion transformer (DiT)** stack with optional product workflows on top. Almost everything funnels through two programs:

| Story | Entry | Frozen / trained | Output |
|-------|--------|------------------|--------|
| **Train** | `train.py` | Trains **DiT** (+ optional fusion layers); **T5/VAE** usually frozen | Checkpoints under `results/` |
| **Sample** | `sample.py` | Loads checkpoint; runs **diffusion** denoising | PNG(s) |
| **Ops** | `scripts/`, `python -m scripts.tools` | Does not train the core loop by default | Data, latents, lint, exports |

A fourth line, **ViT quality** (`vit_quality/`), scores images and captions — it is **not** the generator.

```mermaid
flowchart TB
  subgraph data_layer [Data]
    JSONL[manifest.jsonl / image folders]
    DS[Text2ImageDataset]
    JSONL --> DS
  end

  subgraph encoders [Frozen encoders]
    T5[T5 / triple text bundle]
    VAE[VAE or RAE]
    DS --> VAE
    DS --> T5
  end

  subgraph core [Trained core]
    DiT[DiT + optional ControlNet / LoRA]
    DIFF[GaussianDiffusion]
    VAE --> DIFF
    T5 --> DiT
    DiT --> DIFF
  end

  subgraph train_path [train.py]
    DS --> train_path
    train_path --> encoders
    train_path --> core
    train_path --> CKPT[(checkpoint .pt)]
  end

  subgraph sample_path [sample.py]
    PROMPT[Prompt stack utils/prompt]
    PROMPT --> T5
    CKPT --> DiT
    sample_path --> PROMPT
    sample_path --> core
    sample_path --> IMG[output.png]
  end
```

---

## 2. Repository map (top level)

```
sdx/
├── train.py, sample.py, inference.py, demo.py   # Main programs (stay at root for imports)
├── config/          # Hyperparameters + prompt/domain catalogs
├── data/            # Datasets → batches
├── diffusion/       # Noise schedules, losses, samplers
├── models/          # DiT, ControlNet, bridges, optional scaffolds
├── utils/           # Checkpoints, encoders, prompt pipeline, quality, training helpers
├── training/        # CLI arg builders for train.py (not the loop itself)
├── vit_quality/     # Separate ViT QA / adherence model
├── pipelines/       # Product docs + book/comic orchestration
├── scripts/         # Downloads, tools, enhanced DiT path (not imported by train at startup)
├── native/          # Optional Rust/Go/C++ fast paths
├── toolkit/         # Small QoL CLIs (env report, seeds, timing)
├── tests/           # Pytest suite
├── docs/            # Human documentation (you are here)
├── examples/        # Small runnable examples
├── assets/          # JSON presets (characters, scenes, styles)
├── datasets/        # Your training images (gitignored content typical)
└── pretrained/      # Downloaded HF weights (gitignored)
```

**Rule:** Run commands from **repo root** so `config`, `data`, `models`, `utils` import without extra `PYTHONPATH`. `pyproject.toml` adds `native/python` for `sdx_native`.

---

## 3. Root entry points (how they relate)

| File | Role | Calls into | Used by |
|------|------|------------|---------|
| **`train.py`** | Main training loop: DDP, AMP, VAE encode, T5 encode, diffusion loss, checkpoints, optional REPA/AR curriculum | `config`, `data`, `diffusion`, `models`, `utils/*` | You, CI, `scripts/training/hf_download_and_train.py` |
| **`sample.py`** | Full inference CLI: prompt stack, schedulers, CFG, ControlNet, LoRA, post-process | Same packages + `utils/prompt/*`, `diffusion/sampling` | Users, `pipelines/book_comic/*` |
| **`inference.py`** | Thin loader + optional refine smoke test | `utils/checkpoint/checkpoint_loading` | Scripts wanting programmatic load without full `sample.py` |
| **`demo.py`** | Minimal demo / class-label smoke | `train.py` patterns | Quick sanity |
| **`scripts/cli.py`** | Utility CLI: dataset analysis, config validation, checkpoint tools | `utils/*` | Ops, not hot training path |

**`training/`** (package) only builds `TrainConfig` from argparse — the actual loop is **`train.py`**, not `training/enhanced_trainer.py` (that path is for EnhancedDiT in `scripts/enhanced/`).

---

## 4. `config/` — single source of training truth + prompt catalogs

| Path | Purpose | Consumed by |
|------|---------|-------------|
| **`train_config.py`** | `TrainConfig` dataclass: batch size, LR, DiT flags, REPA, AR, compile, dataloader perf | `train.py`, `training/train_args.py`, checkpoint snapshots |
| **`get_dit_build_kwargs()`** | Maps config → DiT constructor kwargs | `train.py`, `sample.py`, checkpoint load |
| **`defaults/prompt_domains.py`** | Default negatives, domain tips, anti-AI strings, originality tokens | `utils/prompt/stack`, `sample.py`, `data/caption_utils` |
| **`defaults/model_presets.py`** | `--preset sdxl|flux|anime` style bundles | `sample.py` |
| **`defaults/style_*.py`**, **`art_mediums.py`**, **`ai_image_shortcomings.py`** | Training/inference guidance tag sets | Prompt stack guidance stage, caption boosting |
| **`prompt_domains.py`** (root shim) | Re-exports `defaults.prompt_domains` | Legacy imports (`DEPRECATIONS.md`) |
| **`reference/prompt_domains.py`** | Same shim for old doc paths | Docs / external notebooks |

**Relationship:** Training reads **`TrainConfig`** once at startup. Inference rebuilds a compatible config object from the **checkpoint** plus CLI overrides in `sample.py`.

---

## 5. `data/` — from disk to `collate_t2i` batches

| File | Purpose | Feeds |
|------|---------|-------|
| **`t2i_dataset.py`** | `Text2ImageDataset`: folder or JSONL, crops, latent cache, control/init images, grounding masks | `train.py` DataLoader |
| **`caption_utils.py`** | Tag order, quality boost, domain tags, part-aware captions | Dataset + prompt stack training parity |
| **`caption_truncate.py`** | Comma-safe caption length limits | Dataset |
| **`bucket_batch_sampler.py`** | Multi-resolution bucket batches | `train.py` when `--resolution-buckets` set |
| **`enhanced_dataset.py`** | Extended fields for EnhancedDiT training path | `scripts/enhanced/train_enhanced.py` |
| **`vector_index_sampler.py`** | Vector-index based sampling (specialized) | Optional training setups |
| **`civitai/`, `danbooru/`, `prompt_tags/`** | Reference CSVs / tag lists | Caption boosting, docs, tools — not imported at train import time |

**Batch shape:** `collate_t2i` in `t2i_dataset.py` stacks `pixel_values` or `latent_values`, lists of `captions` / `negative_captions`, optional `control_image`, `style`, etc.

**Downstream:** `train.py` moves tensors to GPU (optionally via `utils/training/fast_dataloader.PrefetchDataLoader`), encodes with VAE/T5 inside the step.

---

## 6. `diffusion/` — mathematics between latents and noise

| File | Purpose | Used by |
|------|---------|---------|
| **`gaussian_diffusion.py`** | Core `GaussianDiffusion`: `q_sample`, training losses, DDIM/DDPM steps, CFG | `train.py`, `sample.py` |
| **`respace.py`** | Fewer inference timesteps than training | `create_diffusion()` |
| **`schedules.py`** | Beta schedules (linear, cosine, …) | Diffusion construction |
| **`timestep_sampling.py`** | How training picks `t` (uniform, logit-normal, …) | `train.py` |
| **`losses/timestep_loss_weight.py`** | Min-SNR and related loss weights | Training loss |
| **`sampling_utils.py`** | Thresholding, dynamic CFG helpers | Sampling |
| **`sampling_extras.py`** | Holy Grail presets, extra sampler hooks | `sample.py` |
| **`snr_utils.py`** | SNR analysis (NumPy) | Tools, research |
| **`holy_grail/`** | Adaptive per-step CFG/control plans | `sample.py --holy-grail` |
| **`flow_matching.py`**, **`bridge_training.py`**, … | Research / optional objectives | Flags in `train.py` when enabled |

**Relationship:** `diffusion/__init__.py` exports `create_diffusion()` — both train and sample build the **same** diffusion object type; checkpoints store model weights, not the diffusion class state (betas come from config).

---

## 7. `models/` — DiT and attachments

| File | Purpose | Trained in default `train.py`? |
|------|---------|--------------------------------|
| **`dit.py`** | Base patch DiT blocks (Meta-style) | Yes (backbone) |
| **`dit_text.py`** | T5 cross-attention, caption dropout, ViT options (RoPE, registers) | Yes |
| **`dit_text_variants.py`** | DiT-P / variant constructors | When `model_name` selects them |
| **`attention.py`** | Memory-efficient attention (xformers → SDPA fallback) | Used inside DiT blocks |
| **`controlnet.py`** | Control image conditioning | If control data + flags |
| **`lora.py`** | Low-rank adapters | `sample.py` inference; optional train |
| **`rae_latent_bridge.py`** | RAE latent channel bridge | When `--autoencoder-type rae` |
| **`moe.py`**, **`model_enhancements.py`** | MoE FFN, RMSNorm, FiLM, SE blocks | Optional flags |
| **`enhanced_dit.py`** | Larger “enhanced” architecture | **`scripts/enhanced/`** path only |
| **`native_multimodal_transformer.py`**, **`cascaded_multimodal_diffusion.py`** | Research scaffolds | Not default loop |

**Registry:** `models/__init__.py` maps names like `DiT-XL/2-Text` to builder functions.

**Relationship to encoders:** DiT expects **`encoder_hidden_states`** from T5 (or triple bundle). Patch grid size follows **latent H×W** (image_size // 8 for standard VAE).

---

## 8. `utils/` — cross-cutting services

Utils are grouped by concern. Nothing in `utils/` should be required at `import train` time except what `train.py` already imports.

### 8.1 `utils/checkpoint/`

| File | Role |
|------|------|
| **`checkpoint_loading.py`** | Load DiT + config + RAE bridge + text fusion from `.pt` |
| **`checkpoint_manager.py`** | Rotate saves, best/EMA paths during training |

**Flow:** `train.py` saves → `sample.py` / `inference.py` load via `load_sampler_checkpoint` / `load_dit_text_checkpoint`.

### 8.2 `utils/modeling/`

| File | Role |
|------|------|
| **`text_encoder_bundle.py`** | Triple encoder: T5 + CLIP-L + CLIP-bigG + fusion layers |
| **`t5_segmented_encode.py`** | Long-caption segmented T5 |
| **`model_paths.py`** | Resolve `pretrained/` vs Hugging Face IDs |

**Flow:** `train.py` `get_t5_and_vae()` + optional `load_text_encoder_bundle()` → `encode_text()` in `train.py` (shared with sampling).

### 8.3 `utils/prompt/` — text before the encoder

| Path | Role |
|------|------|
| **`stack/`** | **PromptStack v2**: staged pipeline (`runner.py` → `stages/*`) |
| **`stack/tokens.py`** | CSV tag helpers (`append_unique`, `split_tags`, …) → `fast_paths.py` |
| **`fast_paths.py`** | Hot-path caption merge / neg filter; tries `sdx_native` first |
| **`content_controls.py`** | Quality packs, adherence, anti-AI, Civitai-style controls |
| **`neg_filter.py`** | Remove neg tokens that duplicate positive (delegates to `fast_paths`) |
| **`prompt_emphasis.py`** | `(word)` / `[word]` → T5 token weights |
| **`originality_augment.py`** | Inject originality tokens (train + sample) |
| **`style_native.py`**, **`style_genome/`** | Style Genome v8 exploration |

**Inference chain:** `sample.py` → `apply_sample_prompt_stack()` / `run_prompt_stack()` → stages → final strings → T5.

**Training chain:** `data/caption_utils` + `merge_guidance_for_training_caption()` mirror guidance stage for JSONL captions.

See **[PROMPT_STACK.md](PROMPT_STACK.md)** for stage order.

### 8.4 `utils/generation/`

Sampling helpers used by `sample.py` and research: latent edits, speculative denoise, multimodal stubs, **`run_artifacts.py`**, **`inference_stages.py`**, **`eval_prompt_pack.py`**, etc. These wrap or extend the core loop without replacing `sample.py`.

### 8.5 `utils/training/`

| File | Role |
|------|------|
| **`fast_dataloader.py`** | `PrefetchDataLoader`, `build_fast_dataloader`, worker heuristics |
| **`device_perf.py`** | TF32, cuDNN benchmark, SDPA backends |
| **`config_validator.py`** | Validate `TrainConfig`, memory estimates |
| **`ar_curriculum.py`** | Dynamic AR block schedules |
| **`metrics.py`**, **`error_handling.py`** | Logging, GPU memory, FLOPs |

### 8.6 `utils/quality/`

Post-process and ranking: sharpen, test-time pick, face enhance, artistic post — applied **after** denoising in `sample.py`, not inside DiT.

### 8.7 Other `utils/` packages

| Package | Role |
|---------|------|
| **`utils/consistency/`** | Character consistency DB / losses (experiments, book workflow) |
| **`utils/architecture/`** | AR block layout, DiT profiling, 2026 theme map |
| **`utils/analysis/`** | Dataset analyzer, optional LLM prompt expansion |
| **`utils/visual_design/`** | Design-system / layout helpers for structured prompts |

---

## 9. `training/` — CLI glue only

| File | Role |
|------|------|
| **`train_cli_parser.py`** | All `train.py` argparse flags |
| **`train_args.py`** | `build_train_config_from_args()` → `TrainConfig` |
| **`book_train_preset.py`**, **`enhanced_trainer.py`** | Presets / EnhancedDiT trainer (secondary paths) |

**Relationship:** `train.py` ends with `if __name__ == "__main__":` parsing via these modules — keep new training flags in **both** `train_config.py` and `train_cli_parser.py` / `train_args.py`.

---

## 10. `vit_quality/` — separate quality model

| Module | Role |
|--------|------|
| **`model.py`**, **`dataset.py`**, **`losses.py`** | Train a ViT to score quality / prompt adherence |
| **`train.py`**, **`infer.py`** | CLI entrypoints |
| **`export_embeddings.py`** | Export vectors for dataset QA |

**Not wired into default `train.py` loss.** Use to filter JSONL, rank samples, or gate book pipelines.

---

## 11. `pipelines/` — product workflows (shared engine)

| Path | Role |
|------|------|
| **`image_gen/README.md`** | Docs for general T2I |
| **`book_comic/README.md`** | Multi-page comic/book |
| **`book_comic/scripts/generate_book.py`** | Calls `sample.py` per page with presets |
| **`book_comic/book_helpers.py`** | CFG/post-process presets for books |

**Relationship:** Pipelines do **not** duplicate DiT code — they orchestrate **`train.py` / `sample.py`** with different defaults and manifests.

---

## 12. `scripts/` — operations (import-safe separation)

| Area | Examples | Touches core? |
|------|----------|----------------|
| **`download/`** | `download_models.py` → `pretrained/` | No |
| **`setup/`** | `clone_repos.sh` → `external/` reference clones | No |
| **`training/`** | `precompute_latents.py`, `hf_export_to_sdx_manifest.py` | Prepares data for `train.py` |
| **`tools/`** | `python -m scripts.tools smoke_imports`, `verify_doc_links`, lint | CI / dev |
| **`enhanced/`** | `train_enhanced.py`, `sample_enhanced.py` | Parallel EnhancedDiT stack |
| **`cli.py`** | Analysis utilities | Optional |

**Dispatcher:** `scripts/tools/__main__.py` maps command names to scripts under `tools/dev`, `tools/data`, `tools/prompt`, etc.

---

## 13. `native/` — optional acceleration

| Part | Role |
|------|------|
| **`native/python/sdx_native/`** | Python package: `text_hygiene`, `caption_csv_fast`, `prompt_ops_native`, `diffusion_sigma_fast` |
| **`native/rust/*`** | JSONL tools, prompt ops CLI |
| **`native/cpp`, `native/cuda`** | Latent geometry C ABI |

**Relationship:** `utils/prompt/fast_paths.py` and `data/t2i_dataset.py` **try** native imports and fall back to pure Python. Training works without building native code.

---

## 14. `tests/` — what they guard

| Area | Example tests |
|------|----------------|
| Prompt stack | `test_prompt_stack.py`, `test_prompt_emphasis_import.py` |
| Diffusion math | `test_diffusion_math.py` |
| Data / captions | `test_data_pipeline.py`, `test_caption_truncate.py` |
| Book / pipelines | `test_book_*.py` |
| CLI smoke | `test_cli_entrypoints.py` |
| Native fast paths | `test_native_fast_paths.py` |

Run: `pytest tests/ -q` from repo root.

---

## 15. End-to-end: training step (file touch order)

1. **`training/train_cli_parser.py`** parses CLI → **`training/train_args.py`** builds **`config/TrainConfig`**.
2. **`train.py`** `main()`:
   - **`utils/training/device_perf.py`** — CUDA/TF32/SDPA.
   - Load T5/VAE via **`get_t5_and_vae()`**; optional **`text_encoder_bundle.py`**.
   - Build DiT from **`get_dit_build_kwargs()`** + **`models/`**.
   - **`data/Text2ImageDataset`** + **`utils/training/fast_dataloader.py`**.
   - **`diffusion/create_diffusion()`**.
3. Each batch:
   - Optional **`utils/prompt/originality_augment.py`** on captions.
   - VAE encode (unless **`latent_values`** in batch from cache).
   - **`encode_text()`** (+ optional **`prompt_emphasis.py`** token weights).
   - Forward + loss via **`GaussianDiffusion`**; backward; optimizer.
4. Checkpoint via **`CheckpointManager`** + config snapshot in **`utils/generation/run_artifacts.py`**.

---

## 16. End-to-end: sampling step (file touch order)

1. **`sample.py`** parses CLI; applies **`config/defaults/model_presets.py`** if `--preset`.
2. **`utils/prompt/stack`** builds positive/negative strings (see PROMPT_STACK.md).
3. **`utils/prompt/neg_filter.py`** (→ **`fast_paths.py`**) deduplicates neg vs pos tokens.
4. **`load_model_from_ckpt()`** → **`checkpoint_loading.py`**.
5. **`create_diffusion()`** + schedule from CLI.
6. Encode prompt → **`encoder_hidden_states`**; denoise loop in **`gaussian_diffusion.py`** / extras.
7. VAE decode; optional **`utils/quality/*`** post-process; write PNG.

---

## 17. Assets, examples, external

| Path | Role |
|------|------|
| **`assets/*.json`** | Character/style/scene presets for consistency tools |
| **`examples/*.py`** | Small demos (character consistency, eval packs) |
| **`external/`** | Cloned upstream repos (**reference only** — never import at runtime) |
| **`research/`** | Notes and experiment sketches |

---

## 18. Configuration and checkpoint contract

What must stay aligned across train → sample:

| Piece | Stored in checkpoint? | Must match in sample CLI? |
|-------|------------------------|---------------------------|
| DiT weights | Yes | Loads automatically |
| `model_name`, image size, text mode | In `config` blob | Overrides should be compatible |
| Triple fusion layers | If trained | `--text-encoder-mode triple` |
| RAE bridge | If used | Auto from checkpoint |
| VAE / T5 paths | Config fields | Same models under `pretrained/` |

**`inference.py`** is for loading this bundle programmatically without running the full `sample.py` argument surface.

---

## 19. Where to add new functionality

| Goal | Primary files | Also update |
|------|---------------|-------------|
| New training hyperparameter | `config/train_config.py`, `training/train_cli_parser.py`, `training/train_args.py` | Docs, `get_dit_build_kwargs` if model-related |
| New JSONL field | `data/t2i_dataset.py`, `collate_t2i` | `docs/` dataset format |
| New sampler / schedule | `diffusion/` | `sample.py` flags |
| New DiT block | `models/` | `models/__init__.py`, tests |
| New prompt stage | `utils/prompt/stack/stages/`, `runner.py` | PROMPT_STACK.md |
| New maintenance CLI | `scripts/tools/…`, `scripts/tools/__main__.py` | `scripts/tools/README.md` |
| Fast caption path | `native/` + `utils/prompt/fast_paths.py` | native README |

---

## 20. See also (by topic)

- **Architecture depth:** [MODEL_STACK.md](MODEL_STACK.md), [AR.md](AR.md), [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md)
- **Quality / issues:** [QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md), [MODEL_WEAKNESSES.md](MODEL_WEAKNESSES.md)
- **Book workflow:** [BOOK_COMIC_TECH.md](BOOK_COMIC_TECH.md), [../pipelines/book_comic/README.md](../pipelines/book_comic/README.md)
- **Contributing:** [../CONTRIBUTING.md](../CONTRIBUTING.md), [recipes/local_ci_mirror.md](recipes/local_ci_mirror.md)
- **Deprecations / shims:** [../DEPRECATIONS.md](../DEPRECATIONS.md)

---

*Last expanded: codebase guide covering train/sample relationships, utils prompt stack, and ops layout. Regenerate the file tree with `python -m scripts.tools update_project_structure`.*
