# Codebase organization

This document explains **how SDX is laid out** and **where to put new work** so the tree stays navigable. For a **file-by-file** list, see [FILES.md](FILES.md). For a **navigation tree**, see [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md).

---

## Principles

1. **Core library stays importable from repo root** — `train.py`, `sample.py`, and packages `config`, `data`, `diffusion`, `models`, `utils` are the stable API. Run commands from **`sdx/`** (repo root).
2. **One optional script layer** — `scripts/` holds downloads, training helpers, tools, enhanced DiT, and **`scripts/cli.py`**. Nothing in `scripts/` is imported by `train.py` at import time for the default path.
3. **Product lines are documented, not duplicated** — `pipelines/image_gen` vs `pipelines/book_comic` share the same `train.py` / checkpoints; only docs and orchestration differ.
4. **ViT vs DiT** — `ViT/` is **scoring / QA**, not the diffusion generator. See [ViT/EXCELLENCE_VS_DIT.md](../ViT/EXCELLENCE_VS_DIT.md).

---

## Layer diagram

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

---

## Where to add new code

| You are adding… | Location | Notes |
|-----------------|----------|--------|
| New DiT block / attention | `models/` | Register in `models/__init__.py` if new public API |
| Loss / schedule / diffusion math | `diffusion/` | Keep `GaussianDiffusion` API stable when possible |
| Dataset field or collate | `data/` | Update `t2i_dataset.py` + docs for JSONL fields |
| Training flag | `config/train_config.py` + `get_dit_build_kwargs` | Mirror in `sample.py` / checkpoint if needed |
| Sampling behavior | `sample.py`, `utils/checkpoint/checkpoint_loading.py` | |
| Standalone maintenance CLI | `scripts/tools/` (prefer `python -m scripts.tools <cmd>`) | Add row to [scripts/tools/README.md](../scripts/tools/README.md) |
| Multi-page / book workflow | `pipelines/book_comic/` | Canonical script: `pipelines/book_comic/scripts/generate_book.py` |
| Optional EnhancedDiT workflow | `scripts/enhanced/` | Parallel to main `train.py` |
| Tests | `tests/test_*.py` | Mirror package structure in name |

---

## What we avoid

- **Moving `config/`, `models/`, … under `src/`** without a dedicated migration — it breaks every import and doc link.
- **Duplicating `generate_book.py`** — use `scripts/book/generate_book.py` as a thin launcher only.
- **Importing `external/`** at runtime — clones are reference-only.

---

## Related

- [CODEBASE.md](CODEBASE.md) — layers and conventions for contributors  
- [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) — directory tree and entry points  
- [scripts/tools/README.md](../scripts/tools/README.md) — categorized tool listing
