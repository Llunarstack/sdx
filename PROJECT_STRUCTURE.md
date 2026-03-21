# SDX project structure

> **Auto-generated** — do not edit by hand. Regenerate after moving files:
>
> ```bash
> python scripts/tools/update_project_structure.py
> ```
>
> Generated: **2026-03-21 17:50:42 UTC** · max depth: **5** · repo root: `sdx/`
>
> Skipped directories: **enhanced_dit, external, model** (see `--help` to include).

## Tree

```
sdx/
├── checkpoints/
├── config/
│   ├── __init__.py
│   ├── model_presets.py
│   ├── pixai_reference.py
│   ├── prompt_domains.py
│   ├── style_artists.py
│   └── train_config.py
├── configs/
│   ├── inference/
│   ├── models/
│   └── training/
├── consistency_data/
│   ├── references/
│   ├── characters.json
│   ├── scenes.json
│   └── styles.json
├── data/
│   ├── __init__.py
│   ├── caption_utils.py
│   ├── enhanced_dataset.py
│   └── t2i_dataset.py
├── diffusion/
│   ├── __init__.py
│   ├── cascaded_multimodal_pipeline.py
│   ├── gaussian_diffusion.py
│   ├── loss_weighting.py
│   ├── respace.py
│   ├── sampling_utils.py
│   └── timestep_sampling.py
├── docs/
│   ├── api/
│   ├── guides/
│   ├── tutorials/
│   ├── AR.md
│   ├── BOOK_COMIC_TECH.md
│   ├── BOOK_MODEL_EXCELLENCE.md
│   ├── CIVITAI_QUALITY_TIPS.md
│   ├── CODEBASE.md
│   ├── CODEBASE_ORGANIZATION.md
│   ├── COMMON_ISSUES.md
│   ├── CONNECTIONS.md
│   ├── DANBOORU_HF.md
│   ├── DOMAINS.md
│   ├── ENHANCED_FEATURES.md
│   ├── FILES.md
│   ├── GENERATION_DIAGRAM.md
│   ├── HARDWARE.md
│   ├── HOW_GENERATION_WORKS.md
│   ├── IMPROVEMENTS.md
│   ├── INSPIRATION.md
│   ├── LANDSCAPE_2026.md
│   ├── MODEL_STACK.md
│   ├── MODEL_WEAKNESSES.md
│   ├── MODERN_DIFFUSION.md
│   ├── PROMPT_COOKBOOK.md
│   ├── README.md
│   ├── REGION_CAPTIONS.md
│   ├── REPOSITORY_STRUCTURE.md
│   ├── REPRODUCIBILITY.md
│   ├── SMOKE_TRAINING.md
│   └── STYLE_ARTIST_TAGS.md
├── enhanced_results/
│   ├── 000-EnhancedDiT-XL-2/
│   │   └── checkpoints/
│   ├── 001-EnhancedDiT-XL-2/
│   │   └── checkpoints/
│   └── 002-EnhancedDiT-XL-2/
│       └── checkpoints/
├── examples/
│   ├── notebooks/
│   ├── __init__.py
│   ├── example_character_consistency.py
│   └── example_style_harmonization.py
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── cascaded_multimodal_diffusion.py
│   ├── controlnet.py
│   ├── dit.py
│   ├── dit_predecessor.py
│   ├── dit_text.py
│   ├── enhanced_dit.py
│   ├── lora.py
│   ├── moe.py
│   ├── native_multimodal_transformer.py
│   ├── pixart_blocks.py
│   └── rae_latent_bridge.py
├── native/
│   ├── cpp/
│   │   ├── build/
│   │   │   ├── ALL_BUILD.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── CMakeFiles/
│   │   │   │   ├── 4.2.3/
│   │   │   │   ├── c95a51217f6554e8915ff6cacf54c047/
│   │   │   │   ├── CMakeScratch/
│   │   │   │   ├── pkgRedirects/
│   │   │   │   ├── cmake.check_cache
│   │   │   │   ├── CMakeConfigureLog.yaml
│   │   │   │   ├── generate.stamp
│   │   │   │   ├── generate.stamp.depend
│   │   │   │   ├── generate.stamp.list
│   │   │   │   ├── InstallScripts.json
│   │   │   │   └── TargetDirectories.txt
│   │   │   ├── Debug/
│   │   │   │   ├── sdx_latent.exp
│   │   │   │   ├── sdx_latent.lib
│   │   │   │   └── sdx_latent.pdb
│   │   │   ├── INSTALL.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_latent.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── x64/
│   │   │   │   └── Debug/
│   │   │   ├── ZERO_CHECK.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── ALL_BUILD.vcxproj
│   │   │   ├── ALL_BUILD.vcxproj.filters
│   │   │   ├── cmake_install.cmake
│   │   │   ├── CMakeCache.txt
│   │   │   ├── INSTALL.vcxproj
│   │   │   ├── INSTALL.vcxproj.filters
│   │   │   ├── sdx_latent.sln
│   │   │   ├── sdx_latent.vcxproj
│   │   │   ├── sdx_latent.vcxproj.filters
│   │   │   ├── ZERO_CHECK.vcxproj
│   │   │   └── ZERO_CHECK.vcxproj.filters
│   │   ├── include/
│   │   │   └── sdx/
│   │   │       └── latent.h
│   │   ├── src/
│   │   │   └── sdx_latent.cpp
│   │   └── CMakeLists.txt
│   ├── go/
│   │   └── sdx-manifest/
│   │       ├── go.mod
│   │       └── main.go
│   ├── js/
│   │   ├── sdx-jsonl-stat.mjs
│   │   └── sdx-promptlint.mjs
│   ├── rust/
│   │   └── sdx-jsonl-tools/
│   │       ├── consistency_data/
│   │       │   └── references/
│   │       ├── src/
│   │       │   └── main.rs
│   │       ├── target/
│   │       │   ├── debug/
│   │       │   ├── release/
│   │       │   └── CACHEDIR.TAG
│   │       ├── Cargo.lock
│   │       └── Cargo.toml
│   ├── zig/
│   │   └── sdx-linecrc/
│   │       ├── src/
│   │       │   └── main.zig
│   │       └── build.zig
│   └── README.md
├── pipelines/
│   ├── book_comic/
│   │   ├── scripts/
│   │   │   └── generate_book.py
│   │   ├── __init__.py
│   │   ├── book_helpers.py
│   │   ├── prompt_lexicon.py
│   │   └── README.md
│   ├── image_gen/
│   │   └── README.md
│   ├── __init__.py
│   └── README.md
├── results/
├── scripts/
│   ├── book/
│   │   └── generate_book.py
│   ├── download/
│   │   ├── download_llm.py
│   │   ├── download_models.py
│   │   ├── download_revolutionary_stack.py
│   │   ├── prune_model_files.py
│   │   └── remove_unused_models.py
│   ├── enhanced/
│   │   ├── README.md
│   │   ├── sample_enhanced.py
│   │   ├── save_model_checkpoint.py
│   │   ├── setup_enhanced.py
│   │   └── train_enhanced.py
│   ├── setup/
│   │   ├── clone_repos.ps1
│   │   └── clone_repos.sh
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── book_scene_split.py
│   │   ├── ckpt_info.py
│   │   ├── complex_prompt_coverage.py
│   │   ├── data_quality.py
│   │   ├── dit_variant_compare.py
│   │   ├── eval_prompts.py
│   │   ├── export_onnx.py
│   │   ├── export_safetensors.py
│   │   ├── image_quality_qc.py
│   │   ├── make_smoke_dataset.py
│   │   ├── normalize_captions.py
│   │   ├── op_pipeline.ps1
│   │   ├── op_preflight.py
│   │   ├── prompt_gap_scout.py
│   │   ├── prompt_i18n.py
│   │   ├── prompt_lint.py
│   │   ├── quick_test.py
│   │   ├── README.md
│   │   ├── seed_explorer.py
│   │   ├── smoke_imports.py
│   │   ├── spatial_coverage.py
│   │   ├── tag_coverage.py
│   │   ├── training_timestep_preview.py
│   │   ├── update_project_structure.py
│   │   ├── verify_doc_links.py
│   │   └── vit_inspect.py
│   ├── training/
│   │   ├── hf_download_and_train.py
│   │   ├── hf_export_to_sdx_manifest.py
│   │   └── precompute_latents.py
│   ├── __init__.py
│   ├── cascade_generate.py
│   ├── cli.py
│   └── README.md
├── tests/
│   ├── fixtures/
│   ├── integration/
│   ├── unit/
│   ├── __init__.py
│   ├── test_book_helpers.py
│   ├── test_book_scene_split.py
│   ├── test_character_consistency.py
│   ├── test_dit_architecture.py
│   ├── test_dit_text_extras.py
│   ├── test_enhanced_integration.py
│   ├── test_image_quality_qc.py
│   ├── test_integration.py
│   ├── test_model_creation.py
│   ├── test_native_helpers.py
│   ├── test_native_multimodal_cascade.py
│   ├── test_naturalize_human_art.py
│   ├── test_orchestration.py
│   ├── test_prompt_lexicon.py
│   ├── test_prompt_lint_tool.py
│   ├── test_rae_bridge.py
│   ├── test_region_captions.py
│   ├── test_style_harmonization.py
│   ├── test_text_encoder_fusion.py
│   ├── test_timestep_sampling.py
│   ├── test_update_project_structure.py
│   ├── test_vit_advanced_utils.py
│   ├── test_vit_backbone_presets.py
│   ├── test_vit_module_smoke.py
│   └── test_vit_prompt_system.py
├── training/
│   ├── __init__.py
│   └── enhanced_trainer.py
├── utils/
│   ├── __init__.py
│   ├── advanced_inference.py
│   ├── advanced_prompting.py
│   ├── anatomy_correction.py
│   ├── character_consistency.py
│   ├── checkpoint_loading.py
│   ├── checkpoint_manager.py
│   ├── config_validator.py
│   ├── consistency_losses.py
│   ├── consistency_system.py
│   ├── data_analysis.py
│   ├── dit_architecture.py
│   ├── enhanced_utils.py
│   ├── error_handling.py
│   ├── image_editing.py
│   ├── image_quality_metrics.py
│   ├── llm_client.py
│   ├── master_integration.py
│   ├── metrics.py
│   ├── model_paths.py
│   ├── model_viz.py
│   ├── multimodal_generation.py
│   ├── nn_inspect.py
│   ├── orchestration.py
│   ├── precision_control.py
│   ├── prompt_lint.py
│   ├── quality.py
│   ├── style_harmonization.py
│   ├── test_time_pick.py
│   ├── text_encoder_bundle.py
│   └── text_rendering.py
├── ViT/
│   ├── __init__.py
│   ├── backbone_presets.py
│   ├── checkpoint_utils.py
│   ├── config.py
│   ├── dataset.py
│   ├── ema.py
│   ├── EXCELLENCE_VS_DIT.md
│   ├── export_embeddings.py
│   ├── infer.py
│   ├── losses.py
│   ├── model.py
│   ├── prompt_system.py
│   ├── prompt_tool.py
│   ├── rank.py
│   ├── README.md
│   ├── train.py
│   └── tta.py
├── .editorconfig
├── .env.example
├── .gitignore
├── character_consistency_demo_report.md
├── character_consistency_demo_results.json
├── CHARACTER_CONSISTENCY_IMPLEMENTATION.md
├── CONTRIBUTING.md
├── inference.py
├── LICENSE
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── README.md
├── requirements.txt
├── sample.py
└── train.py
```

## See also

- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) — how to navigate and where to add code
- [docs/CODEBASE_ORGANIZATION.md](docs/CODEBASE_ORGANIZATION.md) — layout principles
- [docs/FILES.md](docs/FILES.md) — full file map

