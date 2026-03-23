# Repository structure — where everything lives

Use this page when you’re lost in the tree or deciding **where new code should go**. Commands assume the **repo root** (`sdx/`) as the working directory.

---

## Top-level map

```
sdx/
├── train.py, sample.py, inference.py   # Main T2I entry points (stay at root for imports & docs)
├── config/                             # TrainConfig, presets, domains
│   └── reference/                      # canonical prompt catalogs & presets (shim *.py at config/ root)
├── data/                               # Datasets, caption pipeline
├── diffusion/                          # Gaussian diffusion, timestep sampling, cascaded scaffold
│   └── losses/                         # timestep loss weights (shim loss_weighting at diffusion/ root)
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
├── user_data/                          # **Your** images + captions for training (`train/` — see user_data/README.md)
├── model/                              # Downloaded weights (gitignored)
└── consistency_data/                   # Sample JSON for character/style consistency tools
```

---

## Entry points (canonical)

| Goal | Command / file |
|------|----------------|
| Train DiT (default stack) | `python train.py …` |
| Sample / generate | `python sample.py …` |
| Programmatic API | `python inference.py` or import from repo root |
| Book / comic pages | `python pipelines/book_comic/scripts/generate_book.py …` |
| ViT dataset QA / scores | `python ViT/train.py` · `ViT/infer.py` |

Run from **repo root** so `config`, `data`, `models`, `utils` resolve without extra `PYTHONPATH`.

---

## `scripts/` layout

| Path | Role |
|------|------|
| **`scripts/download/`** | Pull T5, VAE, CLIP, LLM, optional stacks into `model/` |
| **`scripts/setup/`** | Clone upstream repos into `external/` (reference only) |
| **`scripts/training/`** | HF → JSONL, precompute latents, `hf_download_and_train`, … |
| **`scripts/tools/`** | Utilities — **[scripts/tools/README.md](../scripts/tools/README.md)** (full categorized index) |
| **`scripts/book/`** | Thin launcher → `pipelines/book_comic/scripts/generate_book.py` |
| **`scripts/enhanced/`** | **EnhancedDiT** training, sampling, setup, checkpoint seed — optional path parallel to main `train.py` |
| **`scripts/cascade_generate.py`** | Stable Cascade stub (optional) |

See **[scripts/README.md](../scripts/README.md)** and **[scripts/enhanced/README.md](../scripts/enhanced/README.md)**.

---

## Product lines (same engine, different docs)

| Folder | Audience |
|--------|----------|
| **[pipelines/image_gen/](../pipelines/image_gen/README.md)** | General text-to-image |
| **[pipelines/book_comic/](../pipelines/book_comic/README.md)** | Multi-page, OCR, speech bubbles |

---

## Where to add new code

| You’re adding… | Put it in… |
|----------------|------------|
| New DiT block / attention | `models/` |
| Loss or noise schedule tweak | `diffusion/` (schedules, or `diffusion/losses/` for weighting) |
| New prompt lists / presets (not train hyperparams) | `config/reference/` |
| Dataset field / collation | `data/` |
| Training flag / config field | `config/train_config.py` + `get_dit_build_kwargs` |
| Sampling or checkpoint behavior | `sample.py` / `utils/checkpoint_loading.py` |
| One-off maintenance script | `scripts/tools/` |
| Documentation | `docs/` and link from [docs/README.md](README.md) |
| Tests | `tests/test_*.py` |

---

## Full file index

For every path and description, see **[FILES.md](FILES.md)**.

For a **machine-generated** directory tree (refresh after moves), see **[../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)** (`python scripts/tools/update_project_structure.py`).
