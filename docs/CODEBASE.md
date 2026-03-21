# Codebase guide

How the SDX repo is organized, how to navigate it, and how we keep style consistent.

---

## Layers (mental model)

| Layer | Directory | Responsibility |
|:------|:----------|:---------------|
| **Entry points** | `train.py`, `sample.py`, `inference.py`, `cli.py` | CLI and orchestration |
| **Config** | `config/` | `TrainConfig`, model presets, domain/style tag tables |
| **Data** | `data/` | `Text2ImageDataset`, caption parsing, JSONL → tensors |
| **Diffusion** | `diffusion/` | Noise schedules, `GaussianDiffusion`, sampling utilities |
| **Models** | `models/` | DiT, ControlNet, MoE, RAE bridge, experimental multimodal scaffolds |
| **Utils** | `utils/` | Checkpoints, text encoders, REPA, pick-best, lint, LLM client |
| **ViT tools** | `ViT/` | **Separate** from the generator: quality scoring, ranking, prompt tools |
| **Scripts** | `scripts/` | Downloads, book generation, one-off tools (not imported as a package) |
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

Weights and HF cache live under `model/` (gitignored); paths resolve via `utils/model_paths.py`.

---

## Where to change what

| Goal | Start here |
|:-----|:-----------|
| Training hyperparameters / DiT flags | `config/train_config.py`, `train.py` argparse |
| Caption & JSONL behavior | `data/t2i_dataset.py`, `data/caption_utils.py` |
| Diffusion / schedulers | `diffusion/gaussian_diffusion.py`, `diffusion/respace.py` |
| DiT architecture | `models/dit_text.py`, `models/dit_predecessor.py` |
| Sampling CLI | `sample.py` |
| Checkpoint load / fusion | `utils/checkpoint_loading.py`, `utils/text_encoder_bundle.py` |

---

## See also

- [FILES.md](FILES.md) — per-file map  
- [CONNECTIONS.md](CONNECTIONS.md) — config ↔ checkpoint ↔ sample  
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — PR expectations  
