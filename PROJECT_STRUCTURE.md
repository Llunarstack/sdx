# SDX project structure

> **Auto-generated** — do not edit by hand. Regenerate after moving files:
>
> ```bash
> python -m scripts.tools update_project_structure
> ```
>
> Generated: **2026-04-06 22:23:38 UTC** · max depth: **5** · repo root: `sdx/`
>
> Skipped directories: **enhanced_dit, external, model** (see `--help` to include).

## Tree

```
sdx/
├── assets/
│   ├── references/
│   ├── characters.json
│   ├── scenes.json
│   └── styles.json
├── checkpoints/
├── config/
│   ├── defaults/
│   │   ├── __init__.py
│   │   ├── ai_image_shortcomings.py
│   │   ├── art_mediums.py
│   │   ├── model_presets.py
│   │   ├── pixai_reference.py
│   │   ├── prompt_domains.py
│   │   ├── style_artists.py
│   │   └── style_guidance.py
│   ├── __init__.py
│   ├── ai_image_shortcomings.py
│   ├── art_mediums.py
│   ├── model_presets.py
│   ├── pixai_reference.py
│   ├── prompt_domains.py
│   ├── README.md
│   ├── style_artists.py
│   ├── style_guidance.py
│   └── train_config.py
├── data/
│   ├── civitai/
│   │   ├── model_names.txt
│   │   ├── nsfw_illustrious_noobai_models.csv
│   │   ├── README.md
│   │   ├── SEARCHES.md
│   │   ├── top_triggers_by_frequency.txt
│   │   └── triggers_unique.txt
│   ├── danbooru/
│   │   └── README.md
│   ├── prompt_tags/
│   │   ├── 01_scores_quality_adherence.csv
│   │   ├── 02_sfw.csv
│   │   ├── 03_nsfw_core.csv
│   │   ├── 04_scene_people_objects.csv
│   │   ├── 05_pose_camera_hands.csv
│   │   ├── 06_clothing_lighting_skin.csv
│   │   ├── 07_nsfw_detail_poses_env.csv
│   │   ├── 08_style_media_lora.csv
│   │   └── 09_misc.csv
│   ├── __init__.py
│   ├── bucket_batch_sampler.py
│   ├── caption_utils.py
│   ├── enhanced_dataset.py
│   ├── t2i_dataset.py
│   └── vector_index_sampler.py
├── datasets/
│   ├── train/
│   └── README.md
├── diffusion/
│   ├── holy_grail/
│   │   ├── __init__.py
│   │   ├── blueprint.py
│   │   ├── condition_annealing.py
│   │   ├── guidance_fusion.py
│   │   ├── latent_refiner.py
│   │   ├── presets.py
│   │   ├── prompt_coverage.py
│   │   ├── README.md
│   │   ├── recommender.py
│   │   ├── runtime_guard.py
│   │   └── style_router.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── loss_weighting.py
│   │   └── timestep_loss_weight.py
│   ├── __init__.py
│   ├── attention_steering.py
│   ├── bridge_training.py
│   ├── cascaded_multimodal_pipeline.py
│   ├── cfg_schedulers.py
│   ├── consistency_utils.py
│   ├── flow_matching.py
│   ├── gaussian_diffusion.py
│   ├── inference_timesteps.py
│   ├── latent_bridge.py
│   ├── loss_weighting.py
│   ├── README.md
│   ├── respace.py
│   ├── sampling_utils.py
│   ├── schedules.py
│   ├── self_conditioning.py
│   ├── snr_utils.py
│   ├── spectral_sfp.py
│   ├── timestep_loss_weight.py
│   └── timestep_sampling.py
├── docs/
│   ├── api/
│   ├── assets/
│   │   └── gallery/
│   ├── guides/
│   │   └── CHARACTER_CONSISTENCY_IMPLEMENTATION.md
│   ├── releases/
│   │   ├── v0.1.0.md
│   │   ├── v0.2.0.md
│   │   ├── v3.md
│   │   └── v4.md
│   ├── reports/
│   │   ├── character_consistency_demo_report.md
│   │   └── character_consistency_demo_results.json
│   ├── tutorials/
│   ├── AR.md
│   ├── AR_EXTENSIONS.md
│   ├── BLUEPRINTS.md
│   ├── BOOK_COMIC_TECH.md
│   ├── CANONICAL_STRUCTURE.md
│   ├── CODEBASE.md
│   ├── COMMON_SHORTCOMINGS_AI_IMAGES.md
│   ├── DANBOORU_HF.md
│   ├── DIFFUSION_LEVERAGE_ROADMAP.md
│   ├── DOMAINS.md
│   ├── ENHANCED_FEATURES.md
│   ├── FILES.md
│   ├── HARDWARE.md
│   ├── HF_DATASET_SHORTLIST.md
│   ├── HOW_GENERATION_WORKS.md
│   ├── IMPROVEMENTS.md
│   ├── INSPIRATION.md
│   ├── LANDSCAPE_2026.md
│   ├── MODEL_STACK.md
│   ├── MODEL_WEAKNESSES.md
│   ├── MODERN_DIFFUSION.md
│   ├── NATIVE_AND_SYSTEM_LIBS.md
│   ├── NEXTGEN_SUPERMODEL_ARCHITECTURE.md
│   ├── PROMPT_COOKBOOK.md
│   ├── PROMPT_STACK.md
│   ├── QUALITY_AND_ISSUES.md
│   ├── README.md
│   ├── REGION_CAPTIONS.md
│   ├── REPRODUCIBILITY.md
│   ├── SMOKE_TRAINING.md
│   ├── STYLE_ARTIST_TAGS.md
│   ├── TCIS_MODEL.md
│   └── TRAINING_TEXT_TO_PIXELS.md
├── examples/
│   ├── notebooks/
│   ├── __init__.py
│   ├── example_character_consistency.py
│   ├── example_style_harmonization.py
│   ├── multi_character_scene.example.json
│   ├── prompt_layout.example.json
│   └── prompt_layout_group_mansion_nsfw.json
├── models/
│   ├── __init__.py
│   ├── anti_ai_naturalness.py
│   ├── ar_masks_extended.py
│   ├── attention.py
│   ├── camera_perspective.py
│   ├── cascaded_multimodal_diffusion.py
│   ├── complex_prompt_handler.py
│   ├── controlnet.py
│   ├── dit.py
│   ├── dit_predecessor.py
│   ├── dit_text.py
│   ├── dit_text_variants.py
│   ├── dynamic_patch.py
│   ├── enhanced_dit.py
│   ├── generation_pipeline.py
│   ├── linear_attention.py
│   ├── long_prompt_encoder.py
│   ├── lora.py
│   ├── model_enhancements.py
│   ├── moe.py
│   ├── multi_character.py
│   ├── native_multimodal_transformer.py
│   ├── pixart_blocks.py
│   ├── prompt_adherence.py
│   ├── rae_latent_bridge.py
│   ├── reference_token_projection.py
│   ├── register_tokens.py
│   ├── rope2d.py
│   ├── scene_composer.py
│   ├── superior_vit.py
│   ├── taca.py
│   ├── vit_next_blocks.py
│   └── vit_superior.py
├── native/
│   ├── c/
│   │   ├── include/
│   │   │   └── sdx_c_image_metrics.h
│   │   └── src/
│   │       └── sdx_c_image_metrics.c
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
│   │   │   │   ├── pkgRedirects/
│   │   │   │   ├── cmake.check_cache
│   │   │   │   ├── CMakeConfigureLog.yaml
│   │   │   │   ├── generate.stamp
│   │   │   │   ├── generate.stamp.depend
│   │   │   │   ├── generate.stamp.list
│   │   │   │   ├── InstallScripts.json
│   │   │   │   └── TargetDirectories.txt
│   │   │   ├── INSTALL.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── Release/
│   │   │   │   ├── sdx_beta_schedules.exp
│   │   │   │   ├── sdx_beta_schedules.lib
│   │   │   │   ├── sdx_cuda_flow_matching.exp
│   │   │   │   ├── sdx_cuda_flow_matching.lib
│   │   │   │   ├── sdx_cuda_gaussian_blur.exp
│   │   │   │   ├── sdx_cuda_gaussian_blur.lib
│   │   │   │   ├── sdx_cuda_hwc_to_chw.exp
│   │   │   │   ├── sdx_cuda_hwc_to_chw.lib
│   │   │   │   ├── sdx_cuda_image_metrics.exp
│   │   │   │   ├── sdx_cuda_image_metrics.lib
│   │   │   │   ├── sdx_cuda_ml.exp
│   │   │   │   ├── sdx_cuda_ml.lib
│   │   │   │   ├── sdx_cuda_nf4.exp
│   │   │   │   ├── sdx_cuda_nf4.lib
│   │   │   │   ├── sdx_cuda_percentile_clamp.exp
│   │   │   │   ├── sdx_cuda_percentile_clamp.lib
│   │   │   │   ├── sdx_cuda_rmsnorm.exp
│   │   │   │   ├── sdx_cuda_rmsnorm.lib
│   │   │   │   ├── sdx_cuda_rope.exp
│   │   │   │   ├── sdx_cuda_rope.lib
│   │   │   │   ├── sdx_cuda_sdpa_online.exp
│   │   │   │   ├── sdx_cuda_sdpa_online.lib
│   │   │   │   ├── sdx_cuda_silu_gate.exp
│   │   │   │   ├── sdx_cuda_silu_gate.lib
│   │   │   │   ├── sdx_fnv64_file.exp
│   │   │   │   ├── sdx_fnv64_file.lib
│   │   │   │   ├── sdx_image_metrics.exp
│   │   │   │   ├── sdx_image_metrics.lib
│   │   │   │   ├── sdx_inference_timesteps.exp
│   │   │   │   ├── sdx_inference_timesteps.lib
│   │   │   │   ├── sdx_latent.exp
│   │   │   │   ├── sdx_latent.lib
│   │   │   │   ├── sdx_line_stats.exp
│   │   │   │   ├── sdx_line_stats.lib
│   │   │   │   ├── sdx_mask_ops.exp
│   │   │   │   ├── sdx_mask_ops.lib
│   │   │   │   ├── sdx_rmsnorm_rows_cpu.exp
│   │   │   │   └── sdx_rmsnorm_rows_cpu.lib
│   │   │   ├── sdx_beta_schedules.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_flow_matching.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_gaussian_blur.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_hwc_to_chw.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_image_metrics.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_ml.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_nf4.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_percentile_clamp.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_rmsnorm.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_rope.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_sdpa_online.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_cuda_silu_gate.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_fnv64_file.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_image_metrics.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_inference_timesteps.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_latent.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_line_stats.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_mask_ops.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_rmsnorm_rows_cpu.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── x64/
│   │   │   │   └── Release/
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
│   │   │   ├── sdx_beta_schedules.vcxproj
│   │   │   ├── sdx_beta_schedules.vcxproj.filters
│   │   │   ├── sdx_cuda_flow_matching.vcxproj
│   │   │   ├── sdx_cuda_flow_matching.vcxproj.filters
│   │   │   ├── sdx_cuda_gaussian_blur.vcxproj
│   │   │   ├── sdx_cuda_gaussian_blur.vcxproj.filters
│   │   │   ├── sdx_cuda_hwc_to_chw.vcxproj
│   │   │   ├── sdx_cuda_hwc_to_chw.vcxproj.filters
│   │   │   ├── sdx_cuda_image_metrics.vcxproj
│   │   │   ├── sdx_cuda_image_metrics.vcxproj.filters
│   │   │   ├── sdx_cuda_ml.vcxproj
│   │   │   ├── sdx_cuda_ml.vcxproj.filters
│   │   │   ├── sdx_cuda_nf4.vcxproj
│   │   │   ├── sdx_cuda_nf4.vcxproj.filters
│   │   │   ├── sdx_cuda_percentile_clamp.vcxproj
│   │   │   ├── sdx_cuda_percentile_clamp.vcxproj.filters
│   │   │   ├── sdx_cuda_rmsnorm.vcxproj
│   │   │   ├── sdx_cuda_rmsnorm.vcxproj.filters
│   │   │   ├── sdx_cuda_rope.vcxproj
│   │   │   ├── sdx_cuda_rope.vcxproj.filters
│   │   │   ├── sdx_cuda_sdpa_online.vcxproj
│   │   │   ├── sdx_cuda_sdpa_online.vcxproj.filters
│   │   │   ├── sdx_cuda_silu_gate.vcxproj
│   │   │   ├── sdx_cuda_silu_gate.vcxproj.filters
│   │   │   ├── sdx_fnv64_file.vcxproj
│   │   │   ├── sdx_fnv64_file.vcxproj.filters
│   │   │   ├── sdx_image_metrics.vcxproj
│   │   │   ├── sdx_image_metrics.vcxproj.filters
│   │   │   ├── sdx_inference_timesteps.vcxproj
│   │   │   ├── sdx_inference_timesteps.vcxproj.filters
│   │   │   ├── sdx_latent.sln
│   │   │   ├── sdx_latent.vcxproj
│   │   │   ├── sdx_latent.vcxproj.filters
│   │   │   ├── sdx_line_stats.vcxproj
│   │   │   ├── sdx_line_stats.vcxproj.filters
│   │   │   ├── sdx_mask_ops.vcxproj
│   │   │   ├── sdx_mask_ops.vcxproj.filters
│   │   │   ├── sdx_rmsnorm_rows_cpu.vcxproj
│   │   │   ├── sdx_rmsnorm_rows_cpu.vcxproj.filters
│   │   │   ├── ZERO_CHECK.vcxproj
│   │   │   └── ZERO_CHECK.vcxproj.filters
│   │   ├── cuda/
│   │   │   ├── flow_matching_velocity.cu
│   │   │   ├── gaussian_blur_latent.cu
│   │   │   ├── hwc_to_chw.cu
│   │   │   ├── image_metrics.cu
│   │   │   ├── l2_normalize_rows.cu
│   │   │   ├── nf4_dequant.cu
│   │   │   ├── percentile_clamp.cu
│   │   │   ├── rmsnorm_rows.cu
│   │   │   ├── rope_apply.cu
│   │   │   ├── sdpa_online_softmax.cu
│   │   │   └── silu_gate.cu
│   │   ├── include/
│   │   │   └── sdx/
│   │   │       ├── experimental/
│   │   │       ├── beta_schedules.h
│   │   │       ├── flow_matching_velocity.h
│   │   │       ├── fnv64_file.h
│   │   │       ├── gaussian_blur_latent.h
│   │   │       ├── hwc_to_chw.h
│   │   │       ├── image_metrics.h
│   │   │       ├── image_metrics_cuda.h
│   │   │       ├── inference_timesteps.h
│   │   │       ├── l2_normalize_rows.h
│   │   │       ├── latent.h
│   │   │       ├── line_stats.h
│   │   │       ├── mask_ops.h
│   │   │       ├── nf4_dequant.h
│   │   │       ├── percentile_clamp.h
│   │   │       ├── rmsnorm_rows.h
│   │   │       ├── rmsnorm_rows_cpu.h
│   │   │       ├── rope_apply.h
│   │   │       ├── score_ops.h
│   │   │       ├── sdpa_online_softmax.h
│   │   │       └── silu_gate.h
│   │   ├── src/
│   │   │   ├── sdx_beta_schedules.cpp
│   │   │   ├── sdx_fnv64_file.cpp
│   │   │   ├── sdx_image_metrics.cpp
│   │   │   ├── sdx_inference_timesteps.cpp
│   │   │   ├── sdx_latent.cpp
│   │   │   ├── sdx_line_stats.cpp
│   │   │   ├── sdx_mask_ops.cpp
│   │   │   ├── sdx_rmsnorm_rows_cpu.cpp
│   │   │   └── sdx_score_ops.cpp
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   ├── cuda/
│   │   └── README.md
│   ├── go/
│   │   ├── sdx-manifest/
│   │   │   ├── go.mod
│   │   │   └── main.go
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
│   │   │   ├── beta_schedules_native.py
│   │   │   ├── cuda_hwc_to_chw.py
│   │   │   ├── cuda_image_metrics_native.py
│   │   │   ├── cuda_l2_normalize.py
│   │   │   ├── diffusion_math_native.py
│   │   │   ├── flow_matching_velocity_native.py
│   │   │   ├── fnv64_file_native.py
│   │   │   ├── gaussian_blur_native.py
│   │   │   ├── image_metrics_native.py
│   │   │   ├── inference_timesteps_native.py
│   │   │   ├── jsonl_manifest_pure.py
│   │   │   ├── latent_geometry.py
│   │   │   ├── line_stats_native.py
│   │   │   ├── mask_ops_native.py
│   │   │   ├── native_tools.py
│   │   │   ├── nf4_dequant_native.py
│   │   │   ├── percentile_clamp_native.py
│   │   │   ├── rmsnorm_native.py
│   │   │   ├── rope_apply_native.py
│   │   │   ├── score_ops_native.py
│   │   │   ├── sdpa_online_native.py
│   │   │   ├── silu_gate_native.py
│   │   │   └── text_hygiene.py
│   │   └── README.md
│   ├── rust/
│   │   ├── sdx-diffusion-math/
│   │   │   ├── src/
│   │   │   │   └── lib.rs
│   │   │   ├── target/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── sdx-image-metrics/
│   │   │   ├── src/
│   │   │   │   └── main.rs
│   │   │   ├── target/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── sdx-jsonl-tools/
│   │   │   ├── src/
│   │   │   │   └── main.rs
│   │   │   ├── target/
│   │   │   │   ├── debug/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   ├── Cargo.toml
│   │   │   └── README.md
│   │   └── sdx-noise-schedule/
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
│   │   │   ├── generate_book.py
│   │   │   ├── prepare_and_train_book.py
│   │   │   └── train_book_model.py
│   │   ├── __init__.py
│   │   ├── book_helpers.py
│   │   ├── book_training_helpers.py
│   │   ├── consistency_helpers.py
│   │   ├── prompt_lexicon.py
│   │   └── README.md
│   ├── image_gen/
│   │   └── README.md
│   ├── __init__.py
│   └── README.md
├── pretrained/
│   ├── AnyDoor-Ref/
│   │   ├── dinov2_vitb14_pretrain.pth
│   │   ├── dinov2_vitg14_pretrain.pth
│   │   ├── dinov2_vitl14_pretrain.pth
│   │   ├── dinov2_vits14_pretrain.pth
│   │   └── epoch=1-step=8687-pruned.ckpt
│   ├── CLIP-ViT-bigG-14/
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── open_clip_model.safetensors
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model-00001-of-00002.safetensors
│   │   ├── pytorch_model-00002-of-00002.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── CLIP-ViT-L-14/
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── CodeFormer/
│   │   └── weights/
│   │       ├── CodeFormer/
│   │       │   └── codeformer.pth
│   │       └── facelib/
│   │           ├── detection_Resnet50_Final.pth
│   │           └── parsing_parsenet.pth
│   ├── Consistency-Decoder/
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   └── diffusion_pytorch_model.safetensors
│   ├── ConvNeXtV2-Large/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   └── pytorch_model.bin
│   ├── CountGD/
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── Depth-Anything-V2-Large/
│   │   └── depth_anything_v2_vitl.pth
│   ├── DINOv2-Giant/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   └── README.md
│   ├── DINOv2-Large/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   └── README.md
│   ├── GenSearcher-8B/
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── merges.txt
│   │   ├── model-00001-of-00004.safetensors
│   │   ├── model-00002-of-00004.safetensors
│   │   ├── model-00003-of-00004.safetensors
│   │   ├── model-00004-of-00004.safetensors
│   │   ├── model.safetensors.index.json
│   │   ├── preprocessor_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── video_preprocessor_config.json
│   │   └── vocab.json
│   ├── GroundingDINO-Base/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── ImageReward/
│   │   ├── ImageReward.pt
│   │   └── med_config.json
│   ├── LAION-Aesthetic-v2/
│   │   ├── ava+logos-l14-linearMSE.pth
│   │   ├── ava+logos-l14-reluMSE.pth
│   │   └── sac+logos+ava1-l14-linearMSE.pth
│   ├── LongCLIP-L/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── Marigold-Depth-v1-1/
│   │   ├── text_encoder/
│   │   │   ├── model.fp16.safetensors
│   │   │   ├── model.safetensors
│   │   │   ├── pytorch_model.bin
│   │   │   └── pytorch_model.fp16.bin
│   │   ├── unet/
│   │   │   ├── diffusion_pytorch_model.bin
│   │   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── vae/
│   │   │   ├── diffusion_pytorch_model.bin
│   │   │   ├── diffusion_pytorch_model.fp16.bin
│   │   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   └── model_index.json
│   ├── Marigold-Normals-v1-1/
│   │   ├── text_encoder/
│   │   │   ├── model.fp16.safetensors
│   │   │   ├── model.safetensors
│   │   │   ├── pytorch_model.bin
│   │   │   └── pytorch_model.fp16.bin
│   │   ├── unet/
│   │   │   ├── diffusion_pytorch_model.bin
│   │   │   ├── diffusion_pytorch_model.fp16.bin
│   │   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── vae/
│   │   │   ├── diffusion_pytorch_model.bin
│   │   │   ├── diffusion_pytorch_model.fp16.bin
│   │   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   └── model_index.json
│   ├── moondream2/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── PerceptCLIP_IQA/
│   │   ├── environment.yml
│   │   ├── modeling.py
│   │   └── perceptCLIP_IQA.pth
│   ├── PickScore_v1/
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── Qwen2.5-14B-Instruct/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── merges.txt
│   │   ├── model-00001-of-00008.safetensors
│   │   ├── model-00002-of-00008.safetensors
│   │   ├── model-00003-of-00008.safetensors
│   │   ├── model-00004-of-00008.safetensors
│   │   ├── model-00005-of-00008.safetensors
│   │   ├── model-00006-of-00008.safetensors
│   │   ├── model-00007-of-00008.safetensors
│   │   ├── model-00008-of-00008.safetensors
│   │   ├── model.safetensors.index.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── Real-ESRGAN/
│   │   ├── RealESRGAN_x2.pth
│   │   ├── RealESRGAN_x4.pth
│   │   └── RealESRGAN_x8.pth
│   ├── SAM2-Hiera-Large/
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── SigLIP-SO400M/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── spiece.model
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── StableCascade-Decoder/
│   │   ├── decoder/
│   │   │   ├── config.json
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── scheduler/
│   │   │   └── scheduler_config.json
│   │   ├── text_encoder/
│   │   │   ├── config.json
│   │   │   └── model.safetensors
│   │   ├── tokenizer/
│   │   │   ├── merges.txt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── vqgan/
│   │   │   ├── config.json
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── effnet_encoder.safetensors
│   │   ├── LICENSE
│   │   ├── model_index.json
│   │   ├── previewer.safetensors
│   │   └── README.md
│   ├── StableCascade-Prior/
│   │   ├── feature_extractor/
│   │   │   └── preprocessor_config.json
│   │   ├── image_encoder/
│   │   │   ├── config.json
│   │   │   └── model.safetensors
│   │   ├── prior/
│   │   │   ├── config.json
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── scheduler/
│   │   │   └── scheduler_config.json
│   │   ├── text_encoder/
│   │   │   ├── config.json
│   │   │   └── model.safetensors
│   │   ├── tokenizer/
│   │   │   ├── merges.txt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── LICENSE
│   │   ├── model_index.json
│   │   └── README.md
│   ├── T5-XXL/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── spiece.model
│   │   └── tokenizer_config.json
│   ├── TAESD/
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.bin
│   │   ├── diffusion_pytorch_model.safetensors
│   │   ├── taesd_decoder.safetensors
│   │   └── taesd_encoder.safetensors
│   ├── TAESDXL/
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.bin
│   │   ├── diffusion_pytorch_model.safetensors
│   │   ├── taesdxl_decoder.safetensors
│   │   └── taesdxl_encoder.safetensors
│   └── TrOCR-Large-Printed/
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model.safetensors
│       ├── preprocessor_config.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.json
├── results/
├── runs/
│   ├── 000-EnhancedDiT-XL-2/
│   │   └── checkpoints/
│   ├── 001-EnhancedDiT-XL-2/
│   │   └── checkpoints/
│   └── 002-EnhancedDiT-XL-2/
│       └── checkpoints/
├── scripts/
│   ├── book/
│   │   ├── generate_book.py
│   │   ├── prepare_and_train_book.py
│   │   └── train_book_model.py
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
│   │   │   ├── ar_mask_inspect.py
│   │   │   ├── architecture_themes.py
│   │   │   ├── ckpt_info.py
│   │   │   ├── generate_sdx_architecture_diagram.py
│   │   │   ├── make_gallery.py
│   │   │   ├── quick_test.py
│   │   │   ├── smoke_imports.py
│   │   │   └── validate_config_json.py
│   │   ├── export/
│   │   │   ├── __init__.py
│   │   │   ├── export_onnx.py
│   │   │   └── export_safetensors.py
│   │   ├── native/
│   │   │   ├── build_native.ps1
│   │   │   └── build_native.sh
│   │   ├── ops/
│   │   │   ├── __init__.py
│   │   │   ├── auto_improve_loop.py
│   │   │   ├── gen_searcher_bridge.py
│   │   │   ├── hybrid_dit_vit_generate.py
│   │   │   ├── op_preflight.py
│   │   │   ├── orchestrate_pipeline.py
│   │   │   ├── pretrained_status.py
│   │   │   └── startup_readiness.py
│   │   ├── prompt/
│   │   │   ├── __init__.py
│   │   │   ├── prompt_lint.py
│   │   │   ├── suggest_style_packs.py
│   │   │   └── tag_coverage.py
│   │   ├── repo/
│   │   │   ├── __init__.py
│   │   │   ├── clean_repo_artifacts.py
│   │   │   ├── update_project_structure.py
│   │   │   └── verify_doc_links.py
│   │   ├── tr/
│   │   │   ├── mine_preference_pairs.py
│   │   │   ├── noise_schedule_export.py
│   │   │   ├── train_diffusion_dpo.py
│   │   │   └── train_kd_distill.py
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── _run_legacy.py
│   │   ├── benchmark_suite.py
│   │   ├── book_scene_split.py
│   │   ├── complex_prompt_coverage.py
│   │   ├── curate_civitai_triggers.py
│   │   ├── dit_variant_compare.py
│   │   ├── download_all_danbooru_categorized_tags.py
│   │   ├── dump_prompt_tag_csvs.py
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
│   ├── tr/
│   │   ├── hf_download_and_train.py
│   │   ├── hf_export_to_sdx_manifest.py
│   │   └── precompute_latents.py
│   ├── __init__.py
│   ├── cascade_generate.py
│   ├── cli.py
│   └── README.md
├── tests/
│   ├── __init__.py
│   ├── test_ai_image_shortcomings.py
│   ├── test_ar_curriculum.py
│   ├── test_ar_masks_extended.py
│   ├── test_art_mediums.py
│   ├── test_auto_improve_loop_tool.py
│   ├── test_auto_oc.py
│   ├── test_benchmark_suite_tool.py
│   ├── test_book_helpers.py
│   ├── test_book_training_helpers.py
│   ├── test_data_pipeline.py
│   ├── test_diffusion_math.py
│   ├── test_holy_grail.py
│   ├── test_hybrid_dit_vit_generate.py
│   ├── test_image_resize.py
│   ├── test_mine_preference_pairs_tool.py
│   ├── test_model_forward.py
│   ├── test_model_paths_gen_searcher.py
│   ├── test_naming_compat.py
│   ├── test_new_modules.py
│   ├── test_photo_realism.py
│   ├── test_prompt_lexicon_artist_helpers.py
│   ├── test_rag_prompt_gen_searcher.py
│   ├── test_startup_readiness_tool.py
│   ├── test_style_artists.py
│   ├── test_style_guidance.py
│   └── test_test_time_pick.py
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
│   ├── tr/
│   │   ├── __init__.py
│   │   ├── env_health.py
│   │   └── seed_utils.py
│   ├── __init__.py
│   └── README.md
├── tr/
│   ├── __init__.py
│   ├── enhanced_trainer.py
│   ├── train_args.py
│   └── train_cli_parser.py
├── utils/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── data_analysis.py
│   │   └── llm_client.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── ar_block_conditioning.py
│   │   ├── ar_block_layout.py
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
│   │   ├── ar_latent_ops.py
│   │   ├── clip_alignment.py
│   │   ├── clip_reference_embed.py
│   │   ├── image_editing.py
│   │   ├── inference_research_hooks.py
│   │   ├── master_integration.py
│   │   ├── multimodal_generation.py
│   │   ├── orchestration.py
│   │   ├── precision_control.py
│   │   ├── speculative_denoise.py
│   │   └── text_rendering.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── model_paths.py
│   │   ├── model_viz.py
│   │   ├── nn_inspect.py
│   │   ├── t5_segmented_encode.py
│   │   └── text_encoder_bundle.py
│   ├── native/
│   │   └── __init__.py
│   ├── prompt/
│   │   ├── __init__.py
│   │   ├── advanced_prompting.py
│   │   ├── auto_oc.py
│   │   ├── civitai_vocab.py
│   │   ├── content_control_tag_data.py
│   │   ├── content_control_tags.py
│   │   ├── content_controls.py
│   │   ├── multi_subject.py
│   │   ├── neg_filter.py
│   │   ├── originality_augment.py
│   │   ├── photo_realism.py
│   │   ├── prompt_emphasis.py
│   │   ├── prompt_layout.py
│   │   ├── prompt_lint.py
│   │   ├── rag_prompt.py
│   │   ├── scene_blueprint.py
│   │   └── shape_scaffold.py
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── face_region_enhance.py
│   │   ├── quality.py
│   │   └── test_time_pick.py
│   ├── quantization/
│   │   ├── __init__.py
│   │   └── nf4_codec.py
│   ├── tr/
│   │   ├── __init__.py
│   │   ├── ar_curriculum.py
│   │   ├── config_validator.py
│   │   ├── diffusion_dpo_loss.py
│   │   ├── error_handling.py
│   │   ├── ladd_distillation.py
│   │   ├── metrics.py
│   │   ├── ot_noise_pairing.py
│   │   ├── part_aware_training.py
│   │   ├── preference_image_dataset.py
│   │   └── preference_jsonl.py
│   ├── __init__.py
│   ├── image_quality_metrics.py
│   └── image_resize.py
├── ViT/
│   ├── __init__.py
│   ├── backbone_presets.py
│   ├── checkpoint_utils.py
│   ├── config.py
│   ├── dataset.py
│   ├── DIT_NEXTGEN_NOTES.md
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
│   ├── tta.py
│   └── VIT_G_ARCHITECTURE_VISION.md
├── vq/
│   ├── __init__.py
│   ├── backbone_presets.py
│   ├── checkpoint_utils.py
│   ├── config.py
│   ├── dataset.py
│   ├── ema.py
│   ├── export_embeddings.py
│   ├── infer.py
│   ├── losses.py
│   ├── model.py
│   ├── prompt_system.py
│   ├── prompt_tool.py
│   ├── rank.py
│   ├── train.py
│   └── tta.py
├── .editorconfig
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── demo.py
├── DEPRECATIONS.md
├── inference.py
├── LICENSE
├── pretrained_status.json
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── README.md
├── requirements-cuda128.txt
├── requirements.txt
├── sample.py
└── train.py
```

## See also

- [docs/CODEBASE.md](docs/CODEBASE.md) — navigate the tree, `scripts/` layout, contribution rules
- [docs/FILES.md](docs/FILES.md) — full file map

