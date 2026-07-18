# `scripts/` — operations and utilities

All paths are relative to the **repository root**. Prefer running scripts with `python scripts/...` from root so imports match `train.py` / `sample.py`.

## Subdirectories

| Directory | Contents |
|-----------|----------|
| **`download/`** | `download_models.py`, `download_llm.py`, prune/remove helpers |
| **`setup/`** | `clone_repos.ps1` / `.sh` — clones reference repos into `external/` |
| **`training/`** | HF export, precompute latents, download-and-train helpers |
| **`tools/`** | Canonical ops CLI — `python -m scripts.tools <cmd>` — see [tools/README.md](tools/README.md) |
| **`enhanced/`** | Optional **EnhancedDiT** train / sample / setup |

## Main CLI

| Command | Purpose |
|---------|---------|
| **`python -m scripts.tools`** | Dataset, config, checkpoints, prompt preview, quality ops |
| **`python scripts/run_pipeline.py`** | Full RunPod-style pipeline orchestrator |

## Compatibility stubs (prefer `scripts.tools`)

Thin redirects remain at `scripts/cascade_generate.py`, `prompt_compose.py`, `profile_image_cli.py`, `research_image_prompt.py`, `integration_smoke.py`.

## See also

- [docs/CODEBASE.md](../docs/CODEBASE.md) — repo tree, entry points
- [docs/reference/FILES.md](../docs/reference/FILES.md) — file map
- [README.md](../README.md) — project entry
