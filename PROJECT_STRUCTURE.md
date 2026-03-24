# SDX project structure

> **Auto-generated** — do not edit by hand. Regenerate after moving files:
>
> ```bash
> python -m scripts.tools update_project_structure
> ```
>
> Generated: **2026-03-24 04:33:00 UTC** · max depth: **5** · repo root: `sdx/`
>
> Skipped directories: **enhanced_dit, external, model** (see `--help` to include).

## Tree

```
sdx/
├── checkpoints/
├── config/
│   ├── reference/
│   │   ├── __init__.py
│   │   ├── model_presets.py
│   │   ├── pixai_reference.py
│   │   ├── prompt_domains.py
│   │   └── style_artists.py
│   ├── __init__.py
│   ├── model_presets.py
│   ├── pixai_reference.py
│   ├── prompt_domains.py
│   ├── README.md
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
│   ├── civitai/
│   │   ├── model_names.txt
│   │   ├── nsfw_illustrious_noobai_models.csv
│   │   ├── README.md
│   │   ├── SEARCHES.md
│   │   ├── top_triggers_by_frequency.txt
│   │   └── triggers_unique.txt
│   ├── __init__.py
│   ├── bucket_batch_sampler.py
│   ├── caption_utils.py
│   ├── enhanced_dataset.py
│   └── t2i_dataset.py
├── diffusion/
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── loss_weighting.py
│   │   └── timestep_loss_weight.py
│   ├── __init__.py
│   ├── cascaded_multimodal_pipeline.py
│   ├── gaussian_diffusion.py
│   ├── loss_weighting.py
│   ├── README.md
│   ├── respace.py
│   ├── sampling_utils.py
│   ├── schedules.py
│   ├── snr_utils.py
│   ├── timestep_loss_weight.py
│   └── timestep_sampling.py
├── docs/
│   ├── api/
│   ├── guides/
│   │   └── CHARACTER_CONSISTENCY_IMPLEMENTATION.md
│   ├── releases/
│   │   └── v0.1.0.md
│   ├── reports/
│   │   ├── character_consistency_demo_report.md
│   │   └── character_consistency_demo_results.json
│   ├── tutorials/
│   ├── AR.md
│   ├── ARCHITECTURE_SHIFT_2026.md
│   ├── BOOK_COMIC_TECH.md
│   ├── BOOK_MODEL_EXCELLENCE.md
│   ├── CIVITAI_QUALITY_TIPS.md
│   ├── CODEBASE.md
│   ├── CODEBASE_ORGANIZATION.md
│   ├── COMMON_ISSUES.md
│   ├── CONNECTIONS.md
│   ├── DANBOORU_HF.md
│   ├── DIFFUSION_LEVERAGE_ROADMAP.md
│   ├── DOMAINS.md
│   ├── ENHANCED_FEATURES.md
│   ├── FILES.md
│   ├── GENERATION_DIAGRAM.md
│   ├── HARDWARE.md
│   ├── HOW_GENERATION_WORKS.md
│   ├── IMPROVEMENTS.md
│   ├── INSPIRATION.md
│   ├── LANDSCAPE_2026.md
│   ├── MODEL_ENHANCEMENTS.md
│   ├── MODEL_STACK.md
│   ├── MODEL_WEAKNESSES.md
│   ├── MODERN_DIFFUSION.md
│   ├── NATIVE_AND_SYSTEM_LIBS.md
│   ├── PROMPT_COOKBOOK.md
│   ├── PROMPT_STACK.md
│   ├── README.md
│   ├── REGION_CAPTIONS.md
│   ├── REPOSITORY_STRUCTURE.md
│   ├── REPRODUCIBILITY.md
│   ├── SMOKE_TRAINING.md
│   ├── STYLE_ARTIST_TAGS.md
│   ├── TRAINING_TEXT_TO_PIXELS.md
│   └── WORKFLOW_INTEGRATION_2026.md
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
│   ├── model_enhancements.py
│   ├── moe.py
│   ├── native_multimodal_transformer.py
│   ├── pixart_blocks.py
│   ├── rae_latent_bridge.py
│   └── reference_token_projection.py
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
│   │   ├── cuda/
│   │   │   └── hwc_to_chw.cu
│   │   ├── include/
│   │   │   └── sdx/
│   │   │       ├── beta_schedules.h
│   │   │       ├── hwc_to_chw.h
│   │   │       ├── inference_timesteps.h
│   │   │       ├── latent.h
│   │   │       └── line_stats.h
│   │   ├── src/
│   │   │   ├── sdx_beta_schedules.cpp
│   │   │   ├── sdx_inference_timesteps.cpp
│   │   │   ├── sdx_latent.cpp
│   │   │   └── sdx_line_stats.cpp
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   ├── cuda/
│   │   └── README.md
│   ├── go/
│   │   ├── sdx-manifest/
│   │   │   ├── go.mod
│   │   │   └── main.go
│   │   └── README.md
│   ├── js/
│   │   └── README.md
│   ├── mojo/
│   │   ├── mojopy/
│   │   │   ├── __init__.py
│   │   │   └── launcher.py
│   │   ├── src/
│   │   │   └── sdx_stub.mojo
│   │   ├── install_mojo_wsl.ps1
│   │   ├── pixi.lock
│   │   ├── pixi.toml
│   │   └── README.md
│   ├── python/
│   │   ├── sdx_native/
│   │   │   ├── __init__.py
│   │   │   ├── cuda_hwc_to_chw.py
│   │   │   ├── jsonl_manifest_pure.py
│   │   │   ├── latent_geometry.py
│   │   │   ├── line_stats_native.py
│   │   │   ├── native_tools.py
│   │   │   └── text_hygiene.py
│   │   └── README.md
│   ├── rust/
│   │   └── sdx-jsonl-tools/
│   │       ├── src/
│   │       │   └── main.rs
│   │       ├── target/
│   │       │   ├── debug/
│   │       │   ├── release/
│   │       │   └── CACHEDIR.TAG
│   │       ├── Cargo.lock
│   │       ├── Cargo.toml
│   │       └── README.md
│   ├── zig/
│   │   ├── sdx-linecrc/
│   │   │   ├── src/
│   │   │   │   └── main.zig
│   │   │   ├── build.zig
│   │   │   └── README.md
│   │   └── sdx-pathstat/
│   │       ├── src/
│   │       │   └── main.zig
│   │       ├── build.zig
│   │       └── README.md
│   └── README.md
├── pipelines/
│   ├── book_comic/
│   │   ├── scripts/
│   │   │   └── generate_book.py
│   │   ├── __init__.py
│   │   ├── book_helpers.py
│   │   ├── consistency_helpers.py
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
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── ar_tag_manifest.py
│   │   │   ├── caption_hygiene.py
│   │   │   ├── data_quality.py
│   │   │   ├── jsonl_merge.py
│   │   │   └── manifest_paths.py
│   │   ├── dev/
│   │   │   ├── __init__.py
│   │   │   ├── ckpt_info.py
│   │   │   ├── quick_test.py
│   │   │   └── smoke_imports.py
│   │   ├── export/
│   │   │   ├── __init__.py
│   │   │   ├── export_onnx.py
│   │   │   └── export_safetensors.py
│   │   ├── ops/
│   │   │   ├── __init__.py
│   │   │   ├── op_preflight.py
│   │   │   └── orchestrate_pipeline.py
│   │   ├── prompt/
│   │   │   ├── __init__.py
│   │   │   ├── prompt_lint.py
│   │   │   └── tag_coverage.py
│   │   ├── repo/
│   │   │   ├── __init__.py
│   │   │   ├── update_project_structure.py
│   │   │   └── verify_doc_links.py
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── _run_legacy.py
│   │   ├── book_scene_split.py
│   │   ├── complex_prompt_coverage.py
│   │   ├── curate_civitai_triggers.py
│   │   ├── dit_variant_compare.py
│   │   ├── download_all_danbooru_categorized_tags.py
│   │   ├── eval_prompts.py
│   │   ├── extract_civitai_snippets_for_content_controls.py
│   │   ├── fetch_civitai_nsfw_concepts.py
│   │   ├── fetch_danbooru_tags.py
│   │   ├── image_quality_qc.py
│   │   ├── make_smoke_dataset.py
│   │   ├── merge_danbooru_categorized_tags.py
│   │   ├── normalize_captions.py
│   │   ├── op_pipeline.ps1
│   │   ├── preview_generation_prompt.py
│   │   ├── prompt_gap_scout.py
│   │   ├── prompt_i18n.py
│   │   ├── README.md
│   │   ├── seed_explorer.py
│   │   ├── spatial_coverage.py
│   │   ├── split_danbooru_general_tags.py
│   │   ├── training_timestep_preview.py
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
│   ├── diffusion/
│   │   ├── test_schedules.py
│   │   ├── test_timestep_loss_weight.py
│   │   └── test_timestep_sampling.py
│   ├── fixtures/
│   ├── integration/
│   │   ├── README.md
│   │   └── test_integration.py
│   ├── unit/
│   │   ├── README.md
│   │   ├── test_architecture_map.py
│   │   ├── test_book_helpers.py
│   │   ├── test_character_customization.py
│   │   ├── test_consistency_helpers.py
│   │   ├── test_content_controls.py
│   │   ├── test_danbooru_tag_split.py
│   │   ├── test_face_region_enhance.py
│   │   ├── test_latent_geometry.py
│   │   ├── test_native_tools.py
│   │   ├── test_neg_filter.py
│   │   ├── test_news_features.py
│   │   ├── test_originality_augment.py
│   │   ├── test_prompt_emphasis.py
│   │   ├── test_reference_tokens_and_sag.py
│   │   ├── test_scene_blueprint.py
│   │   ├── test_test_time_pick.py
│   │   ├── test_text_hygiene.py
│   │   └── test_toolkit_basics.py
│   ├── __init__.py
│   ├── test_ar_dit_vit.py
│   ├── test_book_helpers.py
│   ├── test_book_scene_split.py
│   ├── test_character_consistency.py
│   ├── test_dit_architecture.py
│   ├── test_dit_text_extras.py
│   ├── test_enhanced_integration.py
│   ├── test_image_quality_qc.py
│   ├── test_model_creation.py
│   ├── test_model_enhancements.py
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
│   ├── test_update_project_structure.py
│   ├── test_vit_advanced_utils.py
│   ├── test_vit_backbone_presets.py
│   ├── test_vit_module_smoke.py
│   └── test_vit_prompt_system.py
├── toolkit/
│   ├── extras/
│   │   └── requirements-suggested.txt
│   ├── libs/
│   │   ├── __init__.py
│   │   └── optional_imports.py
│   ├── qol/
│   │   ├── __init__.py
│   │   └── timing.py
│   ├── quality/
│   │   ├── __init__.py
│   │   └── manifest_digest.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── env_health.py
│   │   └── seed_utils.py
│   ├── __init__.py
│   └── README.md
├── training/
│   ├── __init__.py
│   └── enhanced_trainer.py
├── user_data/
│   ├── train/
│   └── README.md
├── utils/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── data_analysis.py
│   │   └── llm_client.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── ar_dit_vit.py
│   │   ├── architecture_map.py
│   │   ├── dit_architecture.py
│   │   └── enhanced_utils.py
│   ├── checkpoint/
│   │   ├── __init__.py
│   │   ├── checkpoint_loading.py
│   │   └── checkpoint_manager.py
│   ├── consistency/
│   │   ├── __init__.py
│   │   ├── character_consistency.py
│   │   ├── character_customization.py
│   │   ├── character_lock.py
│   │   ├── consistency_losses.py
│   │   ├── consistency_system.py
│   │   └── style_harmonization.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── advanced_inference.py
│   │   ├── anatomy_correction.py
│   │   ├── clip_reference_embed.py
│   │   ├── image_editing.py
│   │   ├── master_integration.py
│   │   ├── multimodal_generation.py
│   │   ├── orchestration.py
│   │   ├── precision_control.py
│   │   └── text_rendering.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── model_paths.py
│   │   ├── model_viz.py
│   │   ├── nn_inspect.py
│   │   └── text_encoder_bundle.py
│   ├── native/
│   │   ├── __init__.py
│   │   ├── latent_geometry.py
│   │   ├── native_tools.py
│   │   └── text_hygiene.py
│   ├── prompt/
│   │   ├── __init__.py
│   │   ├── advanced_prompting.py
│   │   ├── civitai_vocab.py
│   │   ├── content_controls.py
│   │   ├── neg_filter.py
│   │   ├── originality_augment.py
│   │   ├── prompt_emphasis.py
│   │   ├── prompt_lint.py
│   │   ├── rag_prompt.py
│   │   └── scene_blueprint.py
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── face_region_enhance.py
│   │   ├── quality.py
│   │   └── test_time_pick.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config_validator.py
│   │   ├── error_handling.py
│   │   └── metrics.py
│   ├── __init__.py
│   └── image_quality_metrics.py
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
├── website/
├── .editorconfig
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── inference.py
├── LICENSE
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── README.md
├── requirements-cuda128.txt
├── requirements.txt
├── sample.py
├── snippets_err.txt
└── train.py
```

## See also

- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) — how to navigate and where to add code
- [docs/CODEBASE_ORGANIZATION.md](docs/CODEBASE_ORGANIZATION.md) — layout principles
- [docs/FILES.md](docs/FILES.md) — full file map

