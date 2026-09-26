# Deprecations and Migrations

This file tracks active compatibility shims and canonical paths.

## Canonical import paths

- `vit_quality.*` is canonical for ViT quality/adherence modules and CLI entrypoints.
- `utils.architecture.ar_block_conditioning` is canonical for AR block conditioning utilities.
- `diffusion.losses.*` is canonical for diffusion loss helpers.
- `config.defaults.*` is canonical for prompt/style/config defaults.
- `models.dit_text_variants` and `models.superior_vit` are canonical for their model families.
- `diffusion.sampling.*` is canonical for Holy Grail presets and advanced sampling helpers.
- `python -m scripts.tools <cmd>` is the canonical ops CLI (not `scripts/cli.py`).
- `python -m scripts.tools preview_prompt_stack` is canonical for prompt preview (not `preview_generation_prompt`).

## Removed legacy paths

| Legacy path | Use instead |
|---|---|
| `ViT.*` (incl. `python -m ViT.train` / `ViT.infer` / `ViT.export_embeddings`) | `vit_quality.*` |
| `diffusion.loss_weighting`, `diffusion.timestep_loss_weight` | `diffusion.losses.*` |
| `models.dit_predecessor` | `models.dit_text_variants` |
| `models.vit_superior` | `models.superior_vit` |
| `config.prompt_domains` / `config.style_artists` / `config.style_guidance` / `config.art_mediums` / `config.ai_image_shortcomings` / `config.reference.*` | `config.defaults.*` |
| `utils._archive.*` | `utils.*` |
| `diffusion.holy_grail` / `diffusion.sampling_extras` | `diffusion.sampling` |
| `diffusion.adaptive_cfg_scheduler` / `adversarial_distillation` / `attention_steering` / `self_conditioning` / `consistency_utils` | `diffusion.cfg_schedulers` / `diffusion.sampling` / training distill scripts |
| `scripts/cli.py` | `python -m scripts.tools` / `sample.py` / `train.py` |
| `utils.generation.master_integration` | `sample.py` / `scripts.tools` |
| `utils.generation.image_editing` / `iterative_refinement` | `utils.generation.editing_phase` / `scripts.tools edit_inpaint` |
| `models.generation_pipeline` | `sample.py` + `models.dit_text` |
| `utils.optimization.*` / `utils.speed.*` / `utils.inference.*` / `utils.compression.*` / `utils.distributed.*` / `utils.monitoring.*` | `utils.superior.*` caches / VAE tiling / `utils.quantization` / real train flags |
| `utils.quality.vit_critic_loop` | `vit_quality` / `utils.superior.vit_mining` |
| `innovations.*` | `utils.agentic` / `utils.quality` / `utils.superior` / `frontier` |
| `docs/NATIVE_KERNELS.md` | `docs/NATIVE_AND_SYSTEM_LIBS.md` + `native/README.md` |
| `scripts/book/*.py` launchers | `pipelines/book_comic/scripts/*.py` |
| `utils.architecture.ar_dit_vit` | `utils.architecture.ar_block_conditioning` |
