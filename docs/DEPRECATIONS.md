# Deprecations and Migrations

This file tracks active compatibility shims and canonical paths.

## Canonical import paths

- `vit_quality.*` is canonical for ViT quality/adherence modules and CLI entrypoints.
- `utils.architecture.ar_block_conditioning` is canonical for AR block conditioning utilities.
- `diffusion.timestep_sampling` is canonical for training timestep sampling and loss weights.
- `config.defaults.*` is canonical for prompt/style/config defaults.
- `models.dit_text` (DiT-Text, DiT-P, DiT-Supreme) and `models.superior_vit` are canonical for their model families.
- `diffusion.sampling` is canonical for Holy Grail presets and advanced sampling helpers.
- `python -m scripts.tools <cmd>` is the canonical ops CLI (not `scripts/cli.py`).
- `python -m scripts.tools preview_prompt_stack` is canonical for prompt preview (not `preview_generation_prompt`).

## Removed legacy paths

| Legacy path | Use instead |
|---|---|
| `ViT.*` (incl. `python -m ViT.train` / `ViT.infer` / `ViT.export_embeddings`) | `vit_quality.*` |
| `diffusion.loss_weighting`, `diffusion.timestep_loss_weight`, `diffusion.losses.*` | `diffusion.timestep_sampling` |
| `models.dit_predecessor`, `models.dit_text_variants` | `models.dit_text` |
| `diffusion.sampling.<submodule>` (`blueprint`, `presets`, `runtime_guard`, …) | `diffusion.sampling` |
| `diffusion.flow_rectified` | `diffusion.flow_matching` |
| `models.vit_superior` | `models.superior_vit` |
| `config.prompt_domains` / `config.style_artists` / `config.style_guidance` / `config.art_mediums` / `config.ai_image_shortcomings` / `config.reference.*` | `config.defaults.*` |
| `utils._archive.*` | `utils.*` |
| `diffusion.holy_grail` / `diffusion.sampling_extras` | `diffusion.sampling` |
| `diffusion.adaptive_cfg_scheduler` / `adversarial_distillation` / `attention_steering` / `self_conditioning` / `consistency_utils` | `diffusion.cfg_schedulers` / `diffusion.sampling` / training distill scripts |
| `scripts/cli.py` | `python -m scripts.tools` / `sample.py` / `train.py` |
| `scripts/{cascade_generate,integration_smoke,profile_image_cli,prompt_compose,research_image_prompt}.py` | `python -m scripts.tools <name>` |
| `pipelines.video.<submodule>` (93 modules) | 17 modules: `frontier_compiler`, `motion`, `quality`, `studio_compiler`, `segment_processor`, `video_io`, `controls`, `pipeline`, `neural_sample`, `interpolate`, `scene_graph`, … |
| `utils.quality.palette_lock`, `utils.analysis.data_analysis` | removed (unused) |
| `frontier.<analyzer>` (archetype, cinema, vibe, …) | `frontier.imagination` |
| `frontier.{anatomy,creatures,mature,medium,realism}` | `frontier.subject` |
| `frontier.{atmosphere,era,materials,motion,safety,typography}` | `frontier.perfect` |
| `frontier.{adherence,causality,economy,semantics,uncertainty}` | `frontier.synthesis` |
| `frontier.{chaos,logic,memory,narrative}` | `frontier.engine` |
| `utils.superior.{block_cache,feature_cache,taylor_cache,dynamic_dit,cfg_rejection}` | `utils.superior.caches` |
| `utils.superior.{dpo_pipeline,flywheel,hard_negative,vit_mining,model_soup,eval_report}` | `utils.superior.auto_loop` |
| `utils.superior.{composite_ranker,ensemble,auto_stack,retrieval,prompt_expand,self_correct,quality_gates}` | `utils.superior.inference_pipeline` |
| `utils.training.{branch,dense,flash,turning_point}_grpo`, `grpo_guard` | `utils.training.grpo` |
| `utils.training.{dpo_advanced,preference_jsonl,preference_image_dataset}` | `utils.training.preference` |
| `diffusion.respace`, `diffusion.snr_utils` | `diffusion.schedules` |
| `vit_quality.config`, `vit_quality.losses` | `vit_quality.model` |
| `scripts.tools superior_* / run_flywheel / agentic_*` | `superior <sub>` / `agentic <sub>` (old names still work as aliases) |
| `utils.generation.master_integration` | `sample.py` / `scripts.tools` |
| `utils.generation.image_editing` / `iterative_refinement` | `utils.generation.editing_phase` / `scripts.tools edit_inpaint` |
| `models.generation_pipeline` | `sample.py` + `models.dit_text` |
| `utils.optimization.*` / `utils.speed.*` / `utils.inference.*` / `utils.compression.*` / `utils.distributed.*` / `utils.monitoring.*` | `utils.superior.*` caches / VAE tiling / real train flags |
| `utils.quality.vit_critic_loop` | `vit_quality` / `utils.superior.auto_loop` |
| `innovations.*` | `utils.agentic` / `utils.quality` / `utils.superior` / `frontier` |
| `docs/NATIVE_KERNELS.md` | `docs/MODEL_STACK.md` + `native/README.md` |
| `scripts/book/*.py` launchers | `pipelines/book_comic/scripts/*.py` |
| `utils.architecture.ar_dit_vit` | `utils.architecture.ar_block_conditioning` |
