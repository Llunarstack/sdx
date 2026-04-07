# Deprecations and Migrations

This file tracks active compatibility shims and canonical paths.

## Canonical import paths

- `vit_quality.*` is canonical for ViT quality/adherence modules and CLI entrypoints.
- `utils.architecture.ar_block_conditioning` is canonical for AR block conditioning utilities.
- `diffusion.losses.*` is canonical for diffusion loss helpers.

## Compatibility shims currently retained

| Legacy path | Canonical path | State |
|---|---|---|
| `ViT.*` | `vit_quality.*` | Retained as thin compatibility shims/re-exports. Prefer canonical for all new imports/docs. |
| `python -m ViT.train` | `python -m vit_quality.train` | Legacy CLI launcher retained for compatibility. |
| `python -m ViT.infer` | `python -m vit_quality.infer` | Legacy CLI launcher retained for compatibility. |
| `python -m ViT.export_embeddings` | `python -m vit_quality.export_embeddings` | Legacy CLI launcher retained for compatibility. |
| `ViT.model` | `vit_quality.model` | Legacy module shim retained for compatibility. |
| `ViT.dataset` | `vit_quality.dataset` | Legacy module shim retained for compatibility. |
| `ViT.checkpoint_utils` | `vit_quality.checkpoint_utils` | Legacy module shim retained for compatibility. |
| `diffusion.loss_weighting` | `diffusion.losses.loss_weighting` | Shim retained for backward compatibility. |
| `diffusion.timestep_loss_weight` | `diffusion.losses.timestep_loss_weight` | Shim retained for backward compatibility. |
| `models.dit_predecessor` | `models.dit_text_variants` | Legacy alias retained for compatibility. |
| `models.vit_superior` | `models.superior_vit` | Legacy alias retained for compatibility. |
| `config.prompt_domains` | `config.defaults.prompt_domains` | Shim retained for backward compatibility. |
| `config.style_artists` | `config.defaults.style_artists` | Shim retained for backward compatibility. |
| `config.style_guidance` | `config.defaults.style_guidance` | Shim retained for backward compatibility. |
| `config.art_mediums` | `config.defaults.art_mediums` | Shim retained for backward compatibility. |
| `config.ai_image_shortcomings` | `config.defaults.ai_image_shortcomings` | Shim retained for backward compatibility. |
| `scripts/book/*.py` launchers | `pipelines/book_comic/scripts/*.py` | Legacy launcher entrypoints retained for compatibility. |

## Removed legacy path

- `utils.architecture.ar_dit_vit` has been removed. Use `utils.architecture.ar_block_conditioning`.
