# SDX documentation

Quick links to all project docs, grouped by purpose.

---

## Project map & architecture

| Doc | Description |
|-----|--------------|
| [../pipelines/README.md](../pipelines/README.md) | **Two product lines:** `image_gen/` vs `book_comic/` (shared `train.py`; split docs + book script). |
| [SMOKE_TRAINING.md](SMOKE_TRAINING.md) | Minimal `train.py` run: synthetic data + small DiT + `--dry-run`. |
| [DANBOORU_HF.md](DANBOORU_HF.md) | Hugging Face Danbooru-style data → JSONL + `train.py`; one-shot `hf_download_and_train.py`. |
| [HF_DATASET_SHORTLIST.md](HF_DATASET_SHORTLIST.md) | Curated shortlist from provided HF dataset links: primary/secondary/optional picks + initial mix weights. |
| [CODEBASE.md](CODEBASE.md) | **Start here for code:** layers, conventions, repo tree, `scripts/` layout, contribution rules, ruff, where to edit. |
| [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) | Recent diffusion / flow ideas vs what SDX implements (timestep sampling, roadmap). |
| [DIFFUSION_LEVERAGE_ROADMAP.md](DIFFUSION_LEVERAGE_ROADMAP.md) | High-leverage diffusion upgrades: data, latents, conditioning, objectives, inference, alignment. |
| [LANDSCAPE_2026.md](LANDSCAPE_2026.md) | **Merged 2026 hub:** industry snapshot, post-diffusion themes, workflow/efficiency + disclaimers — mapped to SDX ([`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py)). |
| [BLUEPRINTS.md](BLUEPRINTS.md) | **Merged research notes:** few-step flow/solvers/distillation (Part 1) + prompt-accuracy / GLS / frequency (Part 2). |
| [FILES.md](FILES.md) | File map: every SDX file and key external references. |
| [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | **Auto-generated** full tree (`scripts/tools/repo/update_project_structure.py`). |
| [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) | **Merged:** Mermaid/ASCII pipeline diagram, step-by-step generation, config/checkpoint/data wiring (was CONNECTIONS + GENERATION_DIAGRAM + this doc). |
| [PROMPT_STACK.md](PROMPT_STACK.md) | **Inference text path:** `content_controls` → `neg_filter` → encoder; flag cheat sheet; links to preview CLI. |
| [NATIVE_AND_SYSTEM_LIBS.md](NATIVE_AND_SYSTEM_LIBS.md) | **Lower-level / native libs:** in-repo Rust/Zig/C++/Go tools + ecosystem picks (image I/O, tokenization, QA) mapped to **quality**, **training**, **prompt adherence**. |
| [MODEL_STACK.md](MODEL_STACK.md) | Local `model/` paths, triple encoders — plus **model enhancements** (RMSNorm, FiLM, cross-attn, cascade blend, RAE scales). |
| [../toolkit/README.md](../toolkit/README.md) | **QoL toolkit:** `python -m toolkit.training.env_health`, manifest digest, seeds, step timer, suggested optional pip packages. |
| [PROMPT_COOKBOOK.md](PROMPT_COOKBOOK.md) | Copy-paste `sample.py` recipes (presets, quality, book). |

---

## Operations & features

| Doc | Description |
|-----|--------------|
| [HARDWARE.md](HARDWARE.md) | PC specs, VRAM, storage, latent cache. |
| [AR.md](AR.md) | Block-wise autoregressive (AR): 0 vs 2 vs 4 blocks, when to use. |
| [TRAINING_TEXT_TO_PIXELS.md](TRAINING_TEXT_TO_PIXELS.md) | **Text tokens ↔ latent patches:** faithful “dissection” vs originality; captions, negatives, creativity, caption dropout. |
| [STYLE_ARTIST_TAGS.md](STYLE_ARTIST_TAGS.md) | Style/artist tags (PixAI, Danbooru): extraction, training, `--auto-style-from-prompt`. |
| [DOMAINS.md](DOMAINS.md) | 3D, realistic, interior/exterior domains. |
| [MODEL_WEAKNESSES.md](MODEL_WEAKNESSES.md) | Hands, faces, text, composition: causes and fixes. |
| [COMMON_SHORTCOMINGS_AI_IMAGES.md](COMMON_SHORTCOMINGS_AI_IMAGES.md) | Broad guide + `sample.py` / `train.py` / `normalize_captions` wiring for mitigation packs. |
| [QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md) | **Merged:** Civitai-style sampling fixes + community issue matrix (SDXL, Flux, Z-Image, …) and `sample.py` flags. |

---

## Releases (versioned source)

| Doc | Description |
|-----|-------------|
| [releases/v0.2.0.md](releases/v0.2.0.md) | **v0.2.0** — flow/bridge/OT sampling, DPO/KD, native + toolkit, docs; GitHub: [Releases](https://github.com/Llunarstack/sdx/releases). |
| [releases/v0.1.0.md](releases/v0.1.0.md) | **v0.1.0** — earlier baseline (2026-03-13). |
| [releases/v3.md](releases/v3.md) | **v3** — benchmark robustness, hard-case mining, hardcase-aware DPO remine, startup readiness checks. |
| [releases/v4.md](releases/v4.md) | **v4** — TCIS uncertainty scaling, elite-memory diversity bonus, constraint annealing, and expanded advanced-model wiring. |

---

## Roadmap & inspiration

| Doc | Description |
|-----|--------------|
| [LANDSCAPE_2026.md](LANDSCAPE_2026.md) | **2026 hub (merged):** industry context, post-diffusion architecture themes, workflow integration + disclaimers — mapped to SDX ([utils/generation/orchestration.py](../utils/generation/orchestration.py), [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py)). |
| [BOOK_COMIC_TECH.md](BOOK_COMIC_TECH.md) | Sequential art: techniques vs SDX, **prompt_lexicon** + `generate_book` flags, and **best-output checklist** (data, training, production tier, pick-best, OCR). |
| [../ViT/EXCELLENCE_VS_DIT.md](../ViT/EXCELLENCE_VS_DIT.md) | **ViT/ vs DiT:** scoring stack vs generator; Swin-DiT, FiT, reward/IQA papers; timm backbone presets. |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Roadmap: quality, fixes, novel ideas — includes **§11 Next-tier / insane quality** and **§12 Industry alignment (2026)**. |
| [INSPIRATION.md](INSPIRATION.md) | What we take from PixAI, ComfyUI, and cloned repos. |
| [PROMPT_COOKBOOK.md](PROMPT_COOKBOOK.md) | Copy‑paste prompt recipes using presets, op‑modes, hard styles, and all the quality flags. |

---

## Important recent additions

| Item | Where |
|-----|-------|
| Part-aware / grounding-aware training | [../utils/training/part_aware_training.py](../utils/training/part_aware_training.py), [../data/t2i_dataset.py](../data/t2i_dataset.py), [../train.py](../train.py) |
| LoRA / DoRA / LyCORIS routing improvements | [../models/lora.py](../models/lora.py), [../sample.py](../sample.py) |
| Reproducibility and strict training hygiene | [../train.py](../train.py), [../training/train_cli_parser.py](../training/train_cli_parser.py), [../training/train_args.py](../training/train_args.py) |
| Dataset shortlist and planning | [HF_DATASET_SHORTLIST.md](HF_DATASET_SHORTLIST.md) |
| Architecture figure generator | [../scripts/tools/dev/generate_sdx_architecture_diagram.py](../scripts/tools/dev/generate_sdx_architecture_diagram.py) |
| Robust benchmark + improvement loop | [../scripts/tools/benchmark_suite.py](../scripts/tools/benchmark_suite.py), [../scripts/tools/ops/auto_improve_loop.py](../scripts/tools/ops/auto_improve_loop.py), [../scripts/tools/training/mine_preference_pairs.py](../scripts/tools/training/mine_preference_pairs.py) |
| Startup readiness (no-train preflight report) | [../scripts/tools/ops/startup_readiness.py](../scripts/tools/ops/startup_readiness.py) |

---

## Maintenance / sanity checks

- `scripts/tools/dev/smoke_imports.py` — Import smoke-test for internal modules (catches broken imports early).
- `scripts/tools/repo/clean_repo_artifacts.py` — Remove generated cache artifacts (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `.pyc`) from repo tree.
- `scripts/tools/prompt/tag_coverage.py` — Scan a JSONL manifest for hard-style/person/anatomy/concept-bleed tag coverage.
- `scripts/tools/spatial_coverage.py` — Scan a JSONL manifest for spatial-wording coverage (`behind`, `next to`, `under`, `left of`, ...).
- `scripts/tools/training_timestep_preview.py` — Preview histograms for `--timestep-sample-mode` (uniform / logit_normal / high_noise) before long training runs.
- `scripts/tools/dit_variant_compare.py` — Parameter counts and FP32/BF16 GiB estimates for DiT / EnhancedDiT registry names.
- `scripts/tools/vit_inspect.py` — Inspect ViT quality checkpoints (config + optional module tree via `utils/modeling/nn_inspect.py`).
- `scripts/tools/ops/op_preflight.py` — One-shot “coverage + thresholds” gate (PASS/FAIL) before training.
- `scripts/tools/complex_prompt_coverage.py` — Check coverage for clothes/weapons/food/text/foreground/background/weird/NSFW categories.
- `scripts/tools/prompt_gap_scout.py` — Analyze a single prompt and suggest missing tricky category keywords.
- [`scripts/tools/preview_generation_prompt.py`](../scripts/tools/preview_generation_prompt.py) — Print effective positive/negative after `content_controls` + neg filter (no checkpoint).

Run from repo root so `config`, `data`, `diffusion`, `models`, `utils` are on `sys.path`.
