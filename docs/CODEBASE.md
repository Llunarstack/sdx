# Codebase guide

How the SDX repo is organized, how to navigate it, and how we keep style consistent.

---

## Layers (mental model)

| Layer | Directory | Responsibility |
|:------|:----------|:---------------|
| **Entry points** | `train.py`, `sample.py`, `inference.py`, `scripts/cli.py` | CLI and orchestration |
| **Config** | `config/` | `TrainConfig`, model presets, domain/style tag tables |
| **Data** | `data/` | `Text2ImageDataset`, caption parsing, JSONL → tensors |
| **Diffusion** | `diffusion/` | Noise schedules, `GaussianDiffusion`, sampling utilities |
| **Models** | `models/` | DiT, ControlNet, MoE, RAE bridge, multimodal fusion; shared blocks in [`model_enhancements.py`](../models/model_enhancements.py) — see [MODEL_ENHANCEMENTS.md](MODEL_ENHANCEMENTS.md) |
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

## See also

- [FILES.md](FILES.md) — per-file map  
- [CONNECTIONS.md](CONNECTIONS.md) — config ↔ checkpoint ↔ sample  
- [PROMPT_STACK.md](PROMPT_STACK.md) — inference prompt pipeline before T5  
- [NATIVE_AND_SYSTEM_LIBS.md](NATIVE_AND_SYSTEM_LIBS.md) — native tools + C/Rust ecosystem for data quality & training adjacency  
- [MODEL_ENHANCEMENTS.md](MODEL_ENHANCEMENTS.md) — RMSNorm, FiLM, cross-attn fusion, cascade blend, RAE scales  
- [../toolkit/README.md](../toolkit/README.md) — training QoL modules (`env_health`, `manifest_digest`, seeds, timing)  
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — PR expectations  
