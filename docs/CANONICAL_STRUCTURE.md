# Canonical Structure Map

This map defines which folders are canonical implementation roots, which are compatibility surfaces, and which are generated or data-only areas.

## Classification by top-level folder

| Folder | Classification | Notes |
|---|---|---|
| `config/` | Canonical implementation | Training config, defaults, and stable shims for moved config catalogs. |
| `data/` | Canonical implementation | Dataset loaders, caption processing, manifest parsing. |
| `diffusion/` | Canonical implementation | Diffusion core and canonical loss modules under `diffusion/losses/`. |
| `models/` | Canonical implementation | Generator model stack and canonical aliases (`dit_text_variants`, `superior_vit`). |
| `training/` | Canonical implementation | Training args/parser and trainer support modules. |
| `utils/` | Canonical implementation | Shared runtime/tooling helpers and architecture bridges. |
| `pipelines/` | Canonical implementation | Product-line orchestration and docs-facing workflows. |
| `scripts/` | Canonical implementation | Operational and maintenance CLIs; `scripts.tools` dispatcher is preferred. |
| `toolkit/` | Canonical implementation | QoL helpers (`env_health`, digest, timing, seeds). |
| `vit_quality/` | Canonical implementation | Canonical ViT quality/adherence package and entrypoints. |
| `ViT/` | Compatibility surface | Legacy namespace kept as thin re-export shims + migration docs. |
| `native/` | Canonical implementation | Optional native tooling and Python bridge package. |
| `docs/` | Canonical documentation | Source-of-truth technical docs and release notes. |
| `tests/` | Canonical verification | Unit/smoke/compat tests. |
| `pretrained/` | Runtime asset storage | Downloaded weights (non-source, gitignored). |
| `datasets/` | Runtime dataset storage | User datasets (non-source, typically gitignored). |
| `results/` | Runtime output storage | Training/sampling outputs (non-source, typically gitignored). |
| `external/` | Reference-only | Cloned upstream repos, not imported at runtime. |
| `enhanced_dit/` | Optional parallel tree | Experimental/parallel path, intentionally excluded from default CI linting. |

## Canonical naming policy

- Use snake_case package/module paths for imports.
- Keep one canonical implementation path per subsystem.
- Keep legacy paths only as thin compatibility shims and document them.

## Current migration map

| Legacy path | Canonical path | Status |
|---|---|---|
| `ViT.*` | `vit_quality.*` | In migration window; prefer canonical for all new code/docs. |
| `python -m ViT.train` | `python -m vit_quality.train` | Legacy launcher retained for compatibility. |
| `python -m ViT.infer` | `python -m vit_quality.infer` | Legacy launcher retained for compatibility. |
| `python -m ViT.export_embeddings` | `python -m vit_quality.export_embeddings` | Legacy launcher retained for compatibility. |
| `ViT.model` | `vit_quality.model` | Legacy import shim retained for compatibility. |
| `ViT.dataset` | `vit_quality.dataset` | Legacy import shim retained for compatibility. |
| `ViT.checkpoint_utils` | `vit_quality.checkpoint_utils` | Legacy import shim retained for compatibility. |
| `diffusion.loss_weighting` | `diffusion.losses.loss_weighting` | Shim retained for compatibility. |
| `diffusion.timestep_loss_weight` | `diffusion.losses.timestep_loss_weight` | Shim retained for compatibility. |
| `models.dit_predecessor` | `models.dit_text_variants` | Alias retained for compatibility. |
| `models.vit_superior` | `models.superior_vit` | Alias retained for compatibility. |
| `utils.architecture.ar_dit_vit` | `utils.architecture.ar_block_conditioning` | Legacy removed; canonical only. |
