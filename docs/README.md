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
| [CANONICAL_STRUCTURE.md](CANONICAL_STRUCTURE.md) | Canonical-vs-compat folder map and migration table for professional naming/layout. |
| [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) | Recent diffusion / flow ideas vs what SDX implements (timestep sampling, roadmap). |
| [DIFFUSION_LEVERAGE_ROADMAP.md](DIFFUSION_LEVERAGE_ROADMAP.md) | High-leverage diffusion upgrades: data, latents, conditioning, objectives, inference, alignment. |
| [research/SAMPLING_EXPERIMENTS_BACKLOG.md](research/SAMPLING_EXPERIMENTS_BACKLOG.md) | CFG/steps/solver experiment grids. |
| [research/IMAGE_QUALITY_LEVERS_2026.md](research/IMAGE_QUALITY_LEVERS_2026.md) | **2026 research map**: CFG/flow sampling, data curation, DPO, perceptual losses — mapped to SDX. |`r`n| [LANDSCAPE_2026.md](LANDSCAPE_2026.md) | **Merged 2026 hub:** industry snapshot, post-diffusion themes, workflow/efficiency + disclaimers — mapped to SDX ([`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py)). |
| [BLUEPRINTS.md](BLUEPRINTS.md) | **Merged research notes:** few-step flow/solvers/distillation (Part 1) + prompt-accuracy / GLS / frequency (Part 2). |
| [recipes/quick_eval_holy_grail.md](recipes/quick_eval_holy_grail.md) | Quick evaluation recipe (demo, `sample.py` + Holy Grail, training manifests). |
| [recipes/eval_baseline_prompts.md](recipes/eval_baseline_prompts.md) | Baseline prompt pack + eval driver. |
| [recipes/local_ci_mirror.md](recipes/local_ci_mirror.md) | Mirror CI locally (Ruff, basedpyright, verify_doc_links, pytest). |
| [FILES.md](FILES.md) | File map: every SDX file and key external references. |
| [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | **Auto-generated** full tree (`python -m scripts.tools update_project_structure`). |
| [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) | **Merged:** Mermaid/ASCII pipeline diagram, step-by-step generation, config/checkpoint/data wiring (was CONNECTIONS + GENERATION_DIAGRAM + this doc). |
| [TCIS_OVERVIEW.md](TCIS_OVERVIEW.md) | TCIS hybrid loop: propose, critique, consensus, optional iterate (Mermaid). |
| [HOLY_GRAIL_OVERVIEW.md](HOLY_GRAIL_OVERVIEW.md) | Holy Grail adaptive sampling: how presets become per-step CFG/control/adapter plans (Mermaid). |
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
| [releases/v7.md](releases/v7.md) | **v7** - CI, Dependabot, pre-commit, eval prompt pack + driver, research playbooks, `run_artifacts`, `SECURITY.md`. |
| [releases/v6.md](releases/v6.md) | **v6** - native fast layer + C helpers, `sampling_extras`, book/visual memory, Pyright/clangd tooling, CI, research sketches. |
| [releases/v5.md](releases/v5.md) | **v5** - test-time scaling, manifest curation, DPO safeguards, ViT quality, docs. |
| [releases/v0.2.0.md](releases/v0.2.0.md) | **v0.2.0** — flow/bridge/OT sampling, DPO/KD, native + toolkit, docs; GitHub: [Releases](https://github.com/Llunarstack/sdx/releases). |
| [releases/v0.1.0.md](releases/v0.1.0.md) | **v0.1.0** — earlier baseline (2026-03-13). |
| [releases/v3.md](releases/v3.md) | **v3** — benchmark robustness, hard-case mining, hardcase-aware DPO remine, startup readiness checks. |
| [releases/v4.md](releases/v4.md) | **v4** — TCIS uncertainty scaling, elite-memory diversity bonus, constraint annealing, and expanded advanced-model wiring. |

---

## Roadmap & inspiration

| Doc | Description |
|-----|--------------|
| [research/SAMPLING_EXPERIMENTS_BACKLOG.md](research/SAMPLING_EXPERIMENTS_BACKLOG.md) | CFG/steps/solver experiment grids. |
| [research/IMAGE_QUALITY_LEVERS_2026.md](research/IMAGE_QUALITY_LEVERS_2026.md) | **2026 research map**: CFG/flow sampling, data curation, DPO, perceptual losses — mapped to SDX. |`r`n| [LANDSCAPE_2026.md](LANDSCAPE_2026.md) | **2026 hub (merged):** industry context, post-diffusion architecture themes, workflow integration + disclaimers — mapped to SDX ([utils/generation/orchestration.py](../utils/generation/orchestration.py), [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py)). |
| [BOOK_COMIC_TECH.md](BOOK_COMIC_TECH.md) | Sequential art: techniques vs SDX, **prompt_lexicon** + `generate_book` flags, and **best-output checklist** (data, training, production tier, pick-best, OCR). |
| [MODEL_STACK.md](MODEL_STACK.md) | **Model stack** and how ViT-style quality tooling (it_quality/) relates to DiT generation. |
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

- `python -m scripts.tools smoke_imports` — Import smoke-test for internal modules (catches broken imports early).
- `python -m scripts.tools clean_repo_artifacts` — Remove generated cache artifacts (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `.pyc`) from repo tree.
- `python -m scripts.tools tag_coverage` — Scan a JSONL manifest for hard-style/person/anatomy/concept-bleed tag coverage.
- `python -m scripts.tools spatial_coverage` — Scan a JSONL manifest for spatial-wording coverage (`behind`, `next to`, `under`, `left of`, ...).
- `python -m scripts.tools training_timestep_preview` — Preview histograms for `--timestep-sample-mode` (uniform / logit_normal / high_noise) before long training runs.
- `python -m scripts.tools dit_variant_compare` — Parameter counts and FP32/BF16 GiB estimates for DiT / EnhancedDiT registry names.
- `python -m scripts.tools vit_inspect` — Inspect ViT quality checkpoints (config + optional module tree via `utils/modeling/nn_inspect.py`).
- `python -m scripts.tools op_preflight` — One-shot “coverage + thresholds” gate (PASS/FAIL) before training.
- `python -m scripts.tools complex_prompt_coverage` — Check coverage for clothes/weapons/food/text/foreground/background/weird/NSFW categories.
- `python -m scripts.tools prompt_gap_scout` — Analyze a single prompt and suggest missing tricky category keywords.
- [`scripts/tools/preview_generation_prompt.py`](../scripts/tools/preview_generation_prompt.py) — Print effective positive/negative after `content_controls` + neg filter (no checkpoint).

Run from repo root so `config`, `data`, `diffusion`, `models`, `utils` are on `sys.path`.
