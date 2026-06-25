# SDX Documentation

**Start here:** [GETTING_STARTED.md](GETTING_STARTED.md) · [../README.md](../README.md) (install + quick demo)

---

## Essentials (read these first)

| Doc | What it covers |
|-----|----------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Install, first image, train/sample basics, FAQ |
| [CODEBASE.md](CODEBASE.md) | **Repo map:** layers, modules, train/sample flows, where to edit |
| [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) | End-to-end pipeline diagrams + checkpoint wiring |
| [PROMPT_COOKBOOK.md](PROMPT_COOKBOOK.md) | Copy-paste `sample.py` recipes |
| [QUALITY.md](QUALITY.md) | Sampling fixes, weaknesses, failure modes, mitigations |

---

## Core reference

| Doc | What it covers |
|-----|----------------|
| [MODEL_STACK.md](MODEL_STACK.md) | Encoders, DiT, VAE/RAE, local `pretrained/` paths |
| [PROMPT_STACK.md](PROMPT_STACK.md) | Text path before T5 (inference + training parity) |
| [HOLY_GRAIL_OVERVIEW.md](HOLY_GRAIL_OVERVIEW.md) | Per-step CFG/control/adapter scheduling |
| [TCIS.md](TCIS.md) | Tri-consensus hybrid loop (DiT + ViT quality) |
| [AR.md](AR.md) | Block-wise autoregressive training + extensions |
| [SUPERIOR_STACK.md](SUPERIOR_STACK.md) | RAG, pick-best, CLIP self-correction |
| [NATIVE_KERNELS.md](NATIVE_KERNELS.md) | Optional native acceleration |
| [NATIVE_AND_SYSTEM_LIBS.md](NATIVE_AND_SYSTEM_LIBS.md) | Native + ecosystem lib map |

---

## Guides (`guides/`)

| Doc | What it covers |
|-----|----------------|
| [guides/TRAINING_TEXT_TO_PIXELS.md](guides/TRAINING_TEXT_TO_PIXELS.md) | Captions ↔ latent patches, originality levers |
| [guides/SMOKE_TRAINING.md](guides/SMOKE_TRAINING.md) | Minimal `train.py` dry-run |
| [guides/DANBOORU_HF.md](guides/DANBOORU_HF.md) | HF Danbooru-style data → JSONL |
| [guides/HF_DATASET_SHORTLIST.md](guides/HF_DATASET_SHORTLIST.md) | Curated dataset mix weights |
| [guides/HARDWARE.md](guides/HARDWARE.md) | VRAM, storage, latent cache |
| [guides/BOOK_COMIC_TECH.md](guides/BOOK_COMIC_TECH.md) | Sequential art + `generate_book` |
| [guides/REPRODUCIBILITY.md](guides/REPRODUCIBILITY.md) | `--seed`, `--deterministic` |
| [guides/ADVANCED_OPTIMIZATION.md](guides/ADVANCED_OPTIMIZATION.md) | Speed / quantization |
| [guides/INTEGRATION.md](guides/INTEGRATION.md) | Data cleaning, spatial DSL |

---

## Reference (`reference/`)

| Doc | What it covers |
|-----|----------------|
| [reference/FILES.md](reference/FILES.md) | Per-file map (large; use CODEBASE.md first) |
| [reference/REGION_CAPTIONS.md](reference/REGION_CAPTIONS.md) | JSONL `parts` / `region_captions` |
| [reference/STYLE_ARTIST_TAGS.md](reference/STYLE_ARTIST_TAGS.md) | Style/artist tag extraction |
| [reference/DOMAINS.md](reference/DOMAINS.md) | 3D, interior, realistic domains |

---

## Research (`research/`)

Roadmaps and industry notes — not required for day-to-day use.

| Doc | What it covers |
|-----|----------------|
| [research/IMPROVEMENTS.md](research/IMPROVEMENTS.md) | Feature roadmap |
| [research/LANDSCAPE_2026.md](research/LANDSCAPE_2026.md) | 2026 industry snapshot |
| [research/MODERN_DIFFUSION.md](research/MODERN_DIFFUSION.md) | Flow matching, modern sampling |
| [research/BLUEPRINTS.md](research/BLUEPRINTS.md) | Distillation + prompt-accuracy notes |
| [research/DIFFUSION_LEVERAGE_ROADMAP.md](research/DIFFUSION_LEVERAGE_ROADMAP.md) | High-leverage upgrades |
| [research/IMAGE_QUALITY_LEVERS_2026.md](research/IMAGE_QUALITY_LEVERS_2026.md) | 2026 quality research map |
| [research/IMPROVEMENT_IDEAS.md](research/IMPROVEMENT_IDEAS.md) | Creative backlog |
| [research/SAMPLING_EXPERIMENTS_BACKLOG.md](research/SAMPLING_EXPERIMENTS_BACKLOG.md) | CFG/steps experiment grids |
| [research/INSPIRATION.md](research/INSPIRATION.md) | PixAI, ComfyUI influences |
| [research/NEXTGEN_SUPERMODEL_ARCHITECTURE.md](research/NEXTGEN_SUPERMODEL_ARCHITECTURE.md) | Architecture themes |

---

## Agentic, frontier, pipelines

| Doc | What it covers |
|-----|----------------|
| [agentic/AGENTIC_STACK.md](agentic/AGENTIC_STACK.md) | Agent orchestration |
| [agentic/QUALITY_AGENTS.md](agentic/QUALITY_AGENTS.md) | Quality agents |
| [../innovations/README.md](../innovations/README.md) | Innovations package |
| [../frontier/README.md](../frontier/README.md) | Frontier research (layout, guidance) |
| [../pipelines/README.md](../pipelines/README.md) | image_gen vs book_comic |

---

## Recipes & releases

| Doc | What it covers |
|-----|----------------|
| [recipes/quick_eval_holy_grail.md](recipes/quick_eval_holy_grail.md) | Quick Holy Grail eval |
| [recipes/fast_training.md](recipes/fast_training.md) | Fast training defaults |
| [recipes/local_ci_mirror.md](recipes/local_ci_mirror.md) | Mirror CI locally |
| [releases/v12.md](releases/v12.md) | Latest release notes |
| [releases/v11.md](releases/v11.md) | v11 release notes |

---

## Other

| Doc | What it covers |
|-----|----------------|
| [design/WEBSITE_DESIGN_BRIEF.md](design/WEBSITE_DESIGN_BRIEF.md) | Product UI mockup brief (not implemented) |
| [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | Auto-generated full tree |

---

## Maintenance

```bash
python -m scripts.tools smoke_imports
python -m scripts.tools verify_doc_links
pytest tests/ -q
```

Old paths at `docs/*.md` root may be **redirect stubs** pointing to the new location.
