# `scripts/` — operations and utilities

All paths are relative to the **repository root**. Prefer running scripts with `python scripts/...` from root so imports match `train.py` / `sample.py`.

## Subdirectories

| Directory | Contents |
|-----------|----------|
| **`download/`** | `download_models.py`, `download_llm.py`, `download_revolutionary_stack.py`, prune/remove helpers |
| **`setup/`** | `clone_repos.ps1` / `.sh` — clones DiT, Flux, ControlNet, etc. into `external/` (read-only reference) |
| **`training/`** | `hf_export_to_sdx_manifest.py`, `hf_download_and_train.py`, `precompute_latents.py`, … |
| **`tools/`** | Day-to-day utilities — **[tools/README.md](tools/README.md)** (categorized index) |
| **`book/`** | Legacy launcher that forwards to **`pipelines/book_comic/scripts/generate_book.py`** |
| **`enhanced/`** | Optional **EnhancedDiT** train / sample / setup — see [enhanced/README.md](enhanced/README.md) |

## Main CLI

| Script | Purpose |
|--------|---------|
| **`cli.py`** | Dataset analysis, config validation, checkpoints, prompt helpers — `python scripts/cli.py --help` |

## Root-level scripts

| Script | Purpose |
|--------|---------|
| **`cascade_generate.py`** | Optional Stable Cascade generation (separate from DiT) |

## See also

- [docs/REPOSITORY_STRUCTURE.md](../docs/REPOSITORY_STRUCTURE.md) — full tree orientation  
- [docs/FILES.md](../docs/FILES.md) — exhaustive file map  
- [README.md](../README.md) — main project entry
