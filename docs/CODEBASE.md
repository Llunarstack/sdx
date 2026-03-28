# Codebase guide

How the SDX repo is organized: **layers**, **repo tree**, **`scripts/` layout**, **contribution rules**, **conventions** (ruff/pytest), and **where to change what**.

---

## Layers (mental model)

| Layer | Directory | Responsibility |
|:------|:----------|:---------------|
| **Entry points** | `train.py`, `sample.py`, `inference.py`, `scripts/cli.py` | CLI and orchestration |
| **Config** | `config/` | `TrainConfig`, model presets, domain/style tag tables |
| **Data** | `data/` | `Text2ImageDataset`, caption parsing, JSONL → tensors |
| **Diffusion** | `diffusion/` | Noise schedules, `GaussianDiffusion`, sampling utilities |
| **Models** | `models/` | DiT, ControlNet, MoE, RAE bridge, multimodal fusion; shared blocks in [`model_enhancements.py`](../models/model_enhancements.py) — see [MODEL_STACK.md](MODEL_STACK.md) |
| **Utils** | `utils/` | Checkpoints, text encoders, REPA, pick-best, **`utils/prompt/`** (content controls, neg filter, blueprint, RAG), lint, LLM client |
| **ViT tools** | `ViT/` | **Separate** from the generator: quality scoring, ranking, prompt tools |
| **Pipelines** | `pipelines/` | **image_gen** vs **book_comic** docs; book workflow script (`pipelines/book_comic/scripts/generate_book.py`); not a second copy of DiT |
| **Scripts** | `scripts/` | Downloads, thin `scripts/book/` launcher, one-off tools (not imported as a package) |
| **Native** | `native/` | Optional Rust/Zig/C++/Go CLIs + `libsdx_latent`; see [native/README.md](../native/README.md) and [NATIVE_AND_SYSTEM_LIBS.md](NATIVE_AND_SYSTEM_LIBS.md) |
| **Toolkit** | `toolkit/` | QoL: env report, seeds, manifest digest, timing, optional-lib hints — [toolkit/README.md](../toolkit/README.md) |
| **Tests** | `tests/` | PyTest; run from repo root |

End-to-end flow and diagrams: [README § Architecture and pipeline](../README.md#architecture-and-pipeline) and [FILES.md](FILES.md).

---

## Conventions

1. **Imports** — Prefer absolute imports from package roots (`from config.train_config import …`) when running from repo root (default for `train.py` / `sample.py`).
2. **Public APIs** — Package `__init__.py` files export a small `__all__`; large modules stay explicit.
3. **Formatting** — `ruff format .` (line length 120, double quotes). Run before large PRs.
4. **Lint** — `ruff check .` must pass (see `pyproject.toml` `[tool.ruff]`).
5. **Types** — `from __future__ import annotations` where helpful; full typing is incremental.

---

## Tooling

```bash
# Install dev-style checks (ruff is standalone; no extra requirements file)
pip install ruff

ruff format .
ruff check .
pytest tests/ -q
```

Weights and HF cache live under `model/` (gitignored); paths resolve via `utils/modeling/model_paths.py`.

---

## Where to change what

| Goal | Start here |
|:-----|:-----------|
| Training hyperparameters / DiT flags | `config/train_config.py`, `train.py` argparse |
| Caption & JSONL behavior | `data/t2i_dataset.py`, `data/caption_utils.py`; Unicode hygiene [`native/python/sdx_native/text_hygiene.py`](../native/python/sdx_native/text_hygiene.py), CLI [`scripts/tools/data/caption_hygiene.py`](../scripts/tools/data/caption_hygiene.py), `train.py --caption-unicode-normalize` |
| Diffusion / schedulers | `diffusion/gaussian_diffusion.py`, `diffusion/respace.py` |
| DiT architecture | `models/dit_text.py`, `models/dit_predecessor.py` |
| Sampling CLI | `sample.py` |
| Prompt scaffolding (SFW/NSFW, quality, de-AI, LoRA hints) | `utils/prompt/content_controls.py`, `utils/prompt/neg_filter.py` — overview [PROMPT_STACK.md](PROMPT_STACK.md) |
| Checkpoint load / fusion | `utils/checkpoint/checkpoint_loading.py`, `utils/modeling/text_encoder_bundle.py` |

---

## Repository tree and entry points

Use this when you need the **ASCII tree**, **`scripts/` layout**, or canonical **CLI entry points**. Same assumptions as above: working directory = **repo root** (`sdx/`).

### Top-level map

```
sdx/
├── train.py, sample.py, inference.py   # Main T2I entry points (stay at root for imports & docs)
├── config/                             # TrainConfig, presets, domains
│   └── reference/                      # Canonical prompt catalogs & presets (shim *.py at config/ root)
├── data/                               # Datasets, caption pipeline
├── diffusion/                          # Gaussian diffusion, timestep sampling, cascaded scaffold
│   └── losses/                         # Timestep loss weights (shim loss_weighting at diffusion/ root)
├── models/                             # DiT, ControlNet, MoE, RAE bridge, multimodal scaffolds
├── utils/                              # Checkpoints, text encoders, quality, pick-best, …
├── training/                           # Enhanced trainer module (used by scripts below)
├── ViT/                                # Quality / adherence scoring (not the DiT generator)
├── pipelines/                          # image_gen vs book_comic docs + book workflow
├── scripts/                            # cli.py, download/, tools/, enhanced/, … (see scripts/README.md)
├── tests/                              # pytest (see tests/diffusion/ for diffusion unit tests)
├── examples/                           # Small usage examples
├── native/                             # Optional fast JSONL helpers (Rust, Go, …)
├── docs/                               # All markdown documentation
├── user_data/                          # Your images + captions for training (see user_data/README.md)
├── model/                              # Downloaded weights (gitignored)
└── consistency_data/                   # Sample JSON for character/style consistency tools
```

### Entry points (canonical)

| Goal | Command / file |
|------|----------------|
| Train DiT (default stack) | `python train.py …` |
| Sample / generate | `python sample.py …` |
| Programmatic API | `python inference.py` or import from repo root |
| Book / comic pages | `python pipelines/book_comic/scripts/generate_book.py …` |
| ViT dataset QA / scores | `python ViT/train.py` · `ViT/infer.py` |

Run from **repo root** so `config`, `data`, `models`, `utils` resolve without extra `PYTHONPATH`.

### `scripts/` layout

| Path | Role |
|------|------|
| **`scripts/download/`** | Pull T5, VAE, CLIP, LLM, optional stacks into `model/` |
| **`scripts/setup/`** | Clone upstream repos into `external/` (reference only) |
| **`scripts/training/`** | HF → JSONL, precompute latents, `hf_download_and_train`, … |
| **`scripts/tools/`** | Utilities — grouped entrypoints (`dev/`, `data/`, `prompt/`, `ops/`, `export/`, `repo/`) + **`python -m scripts.tools <cmd>`** dispatcher — **[scripts/tools/README.md](../scripts/tools/README.md)** |
| **`scripts/book/`** | Thin launcher → `pipelines/book_comic/scripts/generate_book.py` |
| **`scripts/enhanced/`** | **EnhancedDiT** training, sampling, setup, checkpoint seed — optional path parallel to main `train.py` |
| **`scripts/cascade_generate.py`** | Stable Cascade stub (optional) |

See **[scripts/README.md](../scripts/README.md)** and **[scripts/enhanced/README.md](../scripts/enhanced/README.md)**.

### Product lines (same engine, different docs)

| Folder | Audience |
|--------|----------|
| **[pipelines/image_gen/](../pipelines/image_gen/README.md)** | General text-to-image |
| **[pipelines/book_comic/](../pipelines/book_comic/README.md)** | Multi-page, OCR, speech bubbles |

### Full file index

Per-path descriptions: **[FILES.md](FILES.md)**. Machine-generated tree: **[../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)** (`python -m scripts.tools update_project_structure`).

---

## Contribution layout (rules of thumb)

Keep new work in predictable places so imports and docs stay stable.

### Principles

1. **Core library stays importable from repo root** — `train.py`, `sample.py`, and packages `config`, `data`, `diffusion`, `models`, `utils` are the stable API.
2. **One optional script layer** — `scripts/` holds downloads, training helpers, tools, enhanced DiT, and **`scripts/cli.py`**. Nothing in `scripts/` is imported by `train.py` at import time for the default path.
3. **Product lines are documented, not duplicated** — `pipelines/image_gen` vs `pipelines/book_comic` share the same `train.py` / checkpoints; only docs and orchestration differ.
4. **ViT vs DiT** — `ViT/` is **scoring / QA**, not the diffusion generator. See [ViT/EXCELLENCE_VS_DIT.md](../ViT/EXCELLENCE_VS_DIT.md).

### Layer diagram

```
                    ┌─────────────┐
                    │  docs/      │  Human docs (you are here)
                    └─────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
┌────────┐          ┌───────────┐         ┌──────────┐
│ train  │          │ config/   │         │ scripts/ │
│ sample │◄────────►│ data/     │         │ download │
│infer…  │          │diffusion/ │         │ tools/   │
└────────┘          │ models/   │         │ enhanced/│
                    │ utils/    │         └──────────┘
                    └───────────┘
```

### Where to add new code

| You are adding… | Location | Notes |
|-----------------|----------|-------|
| New DiT block / attention | `models/` | Register in `models/__init__.py` if new public API |
| Loss / schedule / diffusion math | `diffusion/` | Keep `GaussianDiffusion` API stable when possible |
| New prompt lists / presets (not train hyperparams) | `config/reference/` | |
| Dataset field or collate | `data/` | Update `t2i_dataset.py` + docs for JSONL fields |
| Training flag / config field | `config/train_config.py` + `get_dit_build_kwargs` | Mirror in `sample.py` / checkpoint if needed |
| Sampling or checkpoint behavior | `sample.py`, `utils/checkpoint/checkpoint_loading.py` | |
| Standalone maintenance CLI | `scripts/tools/` (prefer `python -m scripts.tools <cmd>`) | Add row to [scripts/tools/README.md](../scripts/tools/README.md) |
| Multi-page / book workflow | `pipelines/book_comic/` | Canonical script: `pipelines/book_comic/scripts/generate_book.py` |
| Optional EnhancedDiT workflow | `scripts/enhanced/` | Parallel to main `train.py` |
| Documentation | `docs/` and link from [docs/README.md](README.md) | |
| Tests | `tests/test_*.py` | Mirror package structure in name |

### What we avoid

- **Moving `config/`, `models/`, … under `src/`** without a dedicated migration — it breaks every import and doc link.
- **Duplicating `generate_book.py`** — use `scripts/book/generate_book.py` as a thin launcher only.
- **Importing `external/`** at runtime — clones are reference-only.

---

## See also

- [FILES.md](FILES.md) — per-file map  
- [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) — diagram + config ↔ checkpoint ↔ sample (§13)  
- [PROMPT_STACK.md](PROMPT_STACK.md) — inference prompt pipeline before T5  
- [NATIVE_AND_SYSTEM_LIBS.md](NATIVE_AND_SYSTEM_LIBS.md) — native tools + C/Rust ecosystem for data quality & training adjacency  
- [MODEL_STACK.md](MODEL_STACK.md) — local weights + RMSNorm, FiLM, cross-attn fusion, cascade blend, RAE scales  
- [../toolkit/README.md](../toolkit/README.md) — training QoL modules (`env_health`, `manifest_digest`, seeds, timing)  
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — PR expectations  
