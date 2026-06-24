# SDX project structure

> **Auto-generated** — do not edit by hand. Regenerate after moving files:
>
> ```bash
> python -m scripts.tools update_project_structure
> ```
>
> Generated: **2026-06-24 03:00:20 UTC** · max depth: **5** · repo root: `sdx/`
>
> Skipped directories: **enhanced_dit, external, model** (see `--help` to include).

## Tree

```
sdx/
├── assets/
│   ├── characters.json
│   ├── scenes.json
│   └── styles.json
├── checkpoints/
├── config/
│   ├── defaults/
│   │   ├── __init__.py
│   │   ├── agentic_stack.py
│   │   ├── ai_image_shortcomings.py
│   │   ├── art_mediums.py
│   │   ├── creature_character_prompts.py
│   │   ├── model_presets.py
│   │   ├── physics_material_prompts.py
│   │   ├── prompt_domains.py
│   │   ├── style_artists.py
│   │   ├── style_guidance.py
│   │   └── superior_stack.py
│   ├── reference/
│   │   ├── __init__.py
│   │   └── prompt_domains.py
│   ├── __init__.py
│   ├── ai_image_shortcomings.py
│   ├── art_mediums.py
│   ├── prompt_domains.py
│   ├── README.md
│   ├── style_artists.py
│   ├── style_guidance.py
│   └── train_config.py
├── data/
│   ├── danbooru/
│   │   └── README.md
│   ├── style_genomes/
│   │   └── explore_manifest.jsonl
│   ├── __init__.py
│   ├── bucket_batch_sampler.py
│   ├── caption_truncate.py
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
│   │   └── README.md
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── loss_weighting.py
│   │   └── timestep_loss_weight.py
│   ├── sampling/
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
│   ├── sampling_extras/
│   │   ├── __init__.py
│   │   └── README.md
│   ├── __init__.py
│   ├── adaptive_cfg_scheduler.py
│   ├── adversarial_distillation.py
│   ├── attention_steering.py
│   ├── bridge_training.py
│   ├── cascaded_multimodal_pipeline.py
│   ├── cfg_schedulers.py
│   ├── consistency_utils.py
│   ├── flow_matching.py
│   ├── flow_rectified.py
│   ├── gaussian_diffusion.py
│   ├── inference_timesteps.py
│   ├── latent_bridge.py
│   ├── pixel_perfect.py
│   ├── README.md
│   ├── respace.py
│   ├── sampling_utils.py
│   ├── schedules.py
│   ├── self_conditioning.py
│   ├── snr_utils.py
│   ├── spectral_sfp.py
│   └── timestep_sampling.py
├── docs/
│   ├── agentic/
│   │   ├── AGENTIC_STACK.md
│   │   ├── INNOVATIONS_AGENTIC.md
│   │   └── QUALITY_AGENTS.md
│   ├── assets/
│   │   └── gallery/
│   ├── brain/
│   │   └── VISUAL_BRAIN.md
│   ├── guides/
│   │   ├── ADVANCED_OPTIMIZATION.md
│   │   ├── CHARACTER_CONSISTENCY_IMPLEMENTATION.md
│   │   └── INTEGRATION.md
│   ├── recipes/
│   │   ├── eval_baseline_prompts.md
│   │   ├── fast_training.md
│   │   ├── local_ci_mirror.md
│   │   └── quick_eval_holy_grail.md
│   ├── releases/
│   │   ├── v0.1.0.md
│   │   ├── v0.2.0.md
│   │   ├── v10-github-release.md
│   │   ├── v10.md
│   │   ├── v11-github-release.md
│   │   ├── v11.md
│   │   ├── v3.md
│   │   ├── v4.md
│   │   ├── v5.md
│   │   ├── v6.md
│   │   ├── v7.md
│   │   ├── v8-github-release.md
│   │   ├── v8.md
│   │   ├── v9.md
│   │   └── VERSION_COMPARISON.md
│   ├── reports/
│   │   ├── character_consistency_demo_report.md
│   │   └── character_consistency_demo_results.json
│   ├── research/
│   │   ├── AGENTIC_RESEARCH_2026.md
│   │   ├── IMAGE_QUALITY_LEVERS_2026.md
│   │   ├── IMPROVEMENT_IDEAS.md
│   │   ├── SAMPLING_EXPERIMENTS_BACKLOG.md
│   │   └── SUPERIOR_RESEARCH_2026.md
│   ├── AR.md
│   ├── AR_EXTENSIONS.md
│   ├── BLUEPRINTS.md
│   ├── BOOK_COMIC_TECH.md
│   ├── CANONICAL_STRUCTURE.md
│   ├── CODEBASE.md
│   ├── CODEBASE_GUIDE.md
│   ├── COMMON_SHORTCOMINGS_AI_IMAGES.md
│   ├── DANBOORU_HF.md
│   ├── DIFFUSION_LEVERAGE_ROADMAP.md
│   ├── DOMAINS.md
│   ├── ENHANCED_FEATURES.md
│   ├── FILES.md
│   ├── GETTING_STARTED.md
│   ├── HARDWARE.md
│   ├── HF_DATASET_SHORTLIST.md
│   ├── HOLY_GRAIL_OVERVIEW.md
│   ├── HOW_GENERATION_WORKS.md
│   ├── IMPROVEMENTS.md
│   ├── INSPIRATION.md
│   ├── LANDSCAPE_2026.md
│   ├── MODEL_STACK.md
│   ├── MODEL_WEAKNESSES.md
│   ├── MODERN_DIFFUSION.md
│   ├── NATIVE_AND_SYSTEM_LIBS.md
│   ├── NATIVE_KERNELS.md
│   ├── NEXTGEN_SUPERMODEL_ARCHITECTURE.md
│   ├── PROMPT_COOKBOOK.md
│   ├── PROMPT_STACK.md
│   ├── QUALITY_AND_ISSUES.md
│   ├── README.md
│   ├── REGION_CAPTIONS.md
│   ├── REPRODUCIBILITY.md
│   ├── SMOKE_TRAINING.md
│   ├── STYLE_ARTIST_TAGS.md
│   ├── SUPERIOR_STACK.md
│   ├── TCIS_MODEL.md
│   ├── TCIS_OVERVIEW.md
│   └── TRAINING_TEXT_TO_PIXELS.md
├── examples/
│   ├── __init__.py
│   ├── book_visual_memory.example.json
│   ├── box_layout.example.json
│   ├── box_layout_sketch.example.json
│   ├── eval_prompts_baseline.json
│   ├── example_character_consistency.py
│   ├── example_style_harmonization.py
│   ├── multi_character_scene.example.json
│   ├── prompt_layout.example.json
│   ├── prompt_layout_group_mansion_nsfw.json
│   └── run_baseline_eval.py
├── frontier/
│   ├── attention/
│   │   ├── __init__.py
│   │   └── layout_plan.py
│   ├── chaos/
│   │   ├── __init__.py
│   │   ├── entropy_budget.py
│   │   └── serendipity.py
│   ├── compose/
│   │   ├── __init__.py
│   │   └── multi_reference.py
│   ├── guidance/
│   │   ├── __init__.py
│   │   ├── dynamic_cfg.py
│   │   └── guidance_interval.py
│   ├── layout/
│   │   ├── __init__.py
│   │   ├── coordinate_bind.py
│   │   ├── lamic_schedule.py
│   │   ├── layout_metrics.py
│   │   └── omost_canvas.py
│   ├── logic/
│   │   ├── __init__.py
│   │   ├── absence.py
│   │   └── contradiction.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── generation_echo.py
│   ├── narrative/
│   │   ├── __init__.py
│   │   ├── moment.py
│   │   └── witness.py
│   ├── __init__.py
│   ├── engine.py
│   ├── hooks.py
│   ├── README.md
│   └── registry.py
├── innovations/
│   ├── agentic/
│   │   ├── __init__.py
│   │   ├── adaptive_learning.py
│   │   ├── adversarial.py
│   │   ├── artifact_detector.py
│   │   ├── composition_reasoner.py
│   │   ├── drift_detector.py
│   │   ├── ensemble.py
│   │   ├── explainable_scoring.py
│   │   ├── flow_consistency.py
│   │   ├── memory_prefs.py
│   │   ├── perceptual_metrics.py
│   │   ├── prompt_adherence.py
│   │   ├── prompt_optimizer.py
│   │   ├── quality_control.py
│   │   ├── quality_framework.py
│   │   ├── quality_monitor.py
│   │   ├── refinement_loop.py
│   │   ├── rlhf.py
│   │   ├── vision_reward.py
│   │   └── visual_reasoning.py
│   ├── capabilities/
│   │   ├── __init__.py
│   │   ├── animation.py
│   │   ├── dynamic.py
│   │   ├── engine.py
│   │   ├── eraser.py
│   │   ├── hooks.py
│   │   ├── inpainting.py
│   │   ├── loop_video.py
│   │   ├── outpainting.py
│   │   ├── remix.py
│   │   └── weights.py
│   ├── consistency/
│   │   ├── __init__.py
│   │   ├── character.py
│   │   ├── color.py
│   │   ├── engine.py
│   │   ├── hooks.py
│   │   ├── seeding.py
│   │   ├── semantic.py
│   │   ├── style.py
│   │   ├── temporal.py
│   │   └── variation.py
│   ├── control/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── color.py
│   │   ├── detail.py
│   │   ├── effects.py
│   │   ├── engine.py
│   │   ├── hooks.py
│   │   ├── lighting.py
│   │   └── spatial.py
│   ├── multimodal/
│   │   ├── __init__.py
│   │   ├── audio2img.py
│   │   ├── depth.py
│   │   ├── engine.py
│   │   ├── hooks.py
│   │   ├── img2img.py
│   │   ├── scene_graph.py
│   │   ├── sketch2img.py
│   │   ├── text_3d.py
│   │   └── video_style.py
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── cloth.py
│   │   ├── engine.py
│   │   ├── global_light.py
│   │   ├── hooks.py
│   │   ├── liquid.py
│   │   ├── metallic.py
│   │   ├── skin.py
│   │   └── subpixel.py
│   ├── semantics/
│   │   ├── __init__.py
│   │   ├── ambiguity.py
│   │   ├── decomposer.py
│   │   ├── engine.py
│   │   ├── hooks.py
│   │   ├── nuance.py
│   │   └── style.py
│   ├── speed/
│   │   ├── __init__.py
│   │   ├── adaptive.py
│   │   ├── batching.py
│   │   ├── cache.py
│   │   ├── engine.py
│   │   ├── hooks.py
│   │   ├── layer_skip.py
│   │   ├── lora_accel.py
│   │   ├── tiling.py
│   │   └── token_prune.py
│   ├── __init__.py
│   ├── INNOVATION_GUIDE.md
│   ├── pipeline.py
│   ├── README.md
│   └── registry.py
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
│   ├── _experimental/
│   │   ├── c/
│   │   │   ├── include/
│   │   │   │   ├── sdx_c_buffer_stats.h
│   │   │   │   └── sdx_c_image_metrics.h
│   │   │   ├── src/
│   │   │   │   ├── sdx_c_buffer_stats.c
│   │   │   │   └── sdx_c_image_metrics.c
│   │   │   └── compile_flags.txt
│   │   ├── python/
│   │   │   ├── sdx_native/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── attention_mask_pack.py
│   │   │   │   ├── batching_pad_fast.py
│   │   │   │   ├── beta_schedules_native.py
│   │   │   │   ├── buffer_scan_fast.py
│   │   │   │   ├── c_buffer_stats_native.py
│   │   │   │   ├── caption_csv_fast.py
│   │   │   │   ├── coord_grid_fast.py
│   │   │   │   ├── cuda_hwc_to_chw.py
│   │   │   │   ├── cuda_image_metrics_native.py
│   │   │   │   ├── cuda_l2_normalize.py
│   │   │   │   ├── cuda_style_pick_native.py
│   │   │   │   ├── diffusion_math_native.py
│   │   │   │   ├── diffusion_sigma_fast.py
│   │   │   │   ├── flow_matching_velocity_native.py
│   │   │   │   ├── fnv64_file_native.py
│   │   │   │   ├── gaussian_blur_native.py
│   │   │   │   ├── image_metrics_native.py
│   │   │   │   ├── inference_timesteps_native.py
│   │   │   │   ├── jsonl_caption_hygiene.py
│   │   │   │   ├── jsonl_manifest_pure.py
│   │   │   │   ├── latent_geometry.py
│   │   │   │   ├── line_stats_native.py
│   │   │   │   ├── manifest_line_index.py
│   │   │   │   ├── mask_ops_native.py
│   │   │   │   ├── native_fast_stack_status.py
│   │   │   │   ├── native_tools.py
│   │   │   │   ├── nf4_dequant_native.py
│   │   │   │   ├── numpy_chw_pack.py
│   │   │   │   ├── numpy_latent_ops.py
│   │   │   │   ├── percentile_clamp_native.py
│   │   │   │   ├── prompt_hash_fast.py
│   │   │   │   ├── prompt_ops_native.py
│   │   │   │   ├── relpath_norm_fast.py
│   │   │   │   ├── resize_nearest_np.py
│   │   │   │   ├── rmsnorm_native.py
│   │   │   │   ├── rope_apply_native.py
│   │   │   │   ├── score_ops_native.py
│   │   │   │   ├── sdpa_online_native.py
│   │   │   │   ├── silu_gate_native.py
│   │   │   │   ├── style_ops_native.py
│   │   │   │   ├── style_tokens_mojo.py
│   │   │   │   ├── text_hygiene.py
│   │   │   │   ├── timestep_grid_fast.py
│   │   │   │   ├── torch_contiguous_fast.py
│   │   │   │   └── uint8_histogram_fast.py
│   │   │   └── README.md
│   │   └── zig/
│   │       ├── sdx-linecrc/
│   │       │   ├── src/
│   │       │   ├── build.zig
│   │       │   └── README.md
│   │       └── sdx-pathstat/
│   │           ├── src/
│   │           ├── build.zig
│   │           └── README.md
│   ├── cpp/
│   │   ├── cuda/
│   │   │   ├── flow_matching_velocity.cu
│   │   │   ├── gaussian_blur_latent.cu
│   │   │   ├── hwc_to_chw.cu
│   │   │   ├── image_metrics.cu
│   │   │   ├── l2_normalize_rows.cu
│   │   │   ├── nf4_dequant.cu
│   │   │   ├── percentile_clamp.cu
│   │   │   ├── README.md
│   │   │   ├── rmsnorm_rows.cu
│   │   │   ├── rope_apply.cu
│   │   │   ├── sdpa_online_softmax.cu
│   │   │   ├── silu_gate.cu
│   │   │   └── style_pick_best.cu
│   │   ├── include/
│   │   │   ├── sdx/
│   │   │   │   ├── experimental/
│   │   │   │   ├── beta_schedules.h
│   │   │   │   ├── flow_matching_velocity.h
│   │   │   │   ├── fnv64_file.h
│   │   │   │   ├── gaussian_blur_latent.h
│   │   │   │   ├── hwc_to_chw.h
│   │   │   │   ├── image_metrics.h
│   │   │   │   ├── image_metrics_cuda.h
│   │   │   │   ├── inference_timesteps.h
│   │   │   │   ├── l2_normalize_rows.h
│   │   │   │   ├── latent.h
│   │   │   │   ├── line_stats.h
│   │   │   │   ├── mask_ops.h
│   │   │   │   ├── nf4_dequant.h
│   │   │   │   ├── percentile_clamp.h
│   │   │   │   ├── rmsnorm_rows.h
│   │   │   │   ├── rmsnorm_rows_cpu.h
│   │   │   │   ├── rope_apply.h
│   │   │   │   ├── score_ops.h
│   │   │   │   ├── sdpa_online_softmax.h
│   │   │   │   └── silu_gate.h
│   │   │   └── sdx_kernels.h
│   │   ├── src/
│   │   │   ├── kernels.cu
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
│   │   ├── compile_flags.txt
│   │   └── README.md
│   ├── go/
│   │   ├── sdx-manifest/
│   │   │   ├── explore.go
│   │   │   ├── go.mod
│   │   │   └── main.go
│   │   ├── attention.go
│   │   ├── linear.go
│   │   ├── parallel.go
│   │   └── README.md
│   ├── julia/
│   │   ├── Project.toml
│   │   └── sdx_kernels.jl
│   ├── mojo/
│   │   ├── mojopy/
│   │   │   ├── __init__.py
│   │   │   └── launcher.py
│   │   ├── src/
│   │   │   ├── sdx_stub.mojo
│   │   │   └── sdx_style_tokens.mojo
│   │   ├── install_mojo_wsl.ps1
│   │   ├── kernels.mojo
│   │   ├── pixi.lock
│   │   ├── pixi.toml
│   │   └── README.md
│   ├── rust/
│   │   ├── sdx-diffusion-math/
│   │   │   ├── src/
│   │   │   │   └── lib.rs
│   │   │   ├── target/
│   │   │   │   ├── debug/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── sdx-image-metrics/
│   │   │   ├── src/
│   │   │   │   └── main.rs
│   │   │   ├── target/
│   │   │   │   ├── debug/
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
│   │   ├── sdx-noise-schedule/
│   │   │   ├── src/
│   │   │   │   └── main.rs
│   │   │   ├── target/
│   │   │   │   ├── debug/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   ├── Cargo.toml
│   │   │   └── README.md
│   │   ├── sdx-prompt-ops/
│   │   │   ├── src/
│   │   │   │   └── lib.rs
│   │   │   ├── target/
│   │   │   │   ├── debug/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── src/
│   │   │   ├── advanced.rs
│   │   │   ├── lib.rs
│   │   │   ├── main.rs
│   │   │   └── py_module.rs
│   │   └── Cargo.toml
│   ├── wasm/
│   │   └── wasm_kernels.rs
│   ├── .gitignore
│   ├── benchmark_suite.py
│   ├── INTEGRATION_EXAMPLES.md
│   ├── NATIVE_LANGUAGES_INDEX.md
│   └── README.md
├── pipelines/
│   ├── book_comic/
│   │   ├── scripts/
│   │   │   ├── generate_book.py
│   │   │   ├── prepare_and_train_book.py
│   │   │   └── train_book_model.py
│   │   ├── __init__.py
│   │   ├── book_challenging_content.py
│   │   ├── book_helpers.py
│   │   ├── book_manifest_utils.py
│   │   ├── book_model_readiness.py
│   │   ├── book_project.py
│   │   ├── book_prompt_intel.py
│   │   ├── book_style_authenticity.py
│   │   ├── book_style_fusion.py
│   │   ├── book_text_continuity.py
│   │   ├── book_training_helpers.py
│   │   ├── consistency_helpers.py
│   │   ├── prompt_lexicon.py
│   │   ├── README.md
│   │   ├── visual_memory.py
│   │   └── visual_memory_bridge.py
│   ├── image_gen/
│   │   └── README.md
│   ├── __init__.py
│   └── README.md
├── pretrained/
│   ├── AnyDoor-Ref/
│   ├── BLIP-image-captioning-base/
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── tf_model.h5
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── CLIP-ViT-bigG-14/
│   ├── CLIP-ViT-L-14/
│   ├── CodeFormer/
│   ├── Consistency-Decoder/
│   ├── ConvNeXtV2-Large/
│   ├── CountGD/
│   ├── CRAFT-text-detector/
│   │   ├── craft_mlt_25k.pth
│   │   └── craft_refiner_CTW1500.pth
│   ├── Depth-Anything-V2-Large/
│   ├── DINOv2-Giant/
│   ├── DINOv2-Large/
│   ├── Donut-base/
│   │   ├── .gitignore
│   │   ├── added_tokens.json
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── README.md
│   │   ├── sentencepiece.bpe.model
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── GenSearcher-8B/
│   ├── GroundingDINO-Base/
│   ├── HPSv2-hf/
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── ImageReward/
│   ├── Kosmos-2-patch14-224/
│   │   ├── added_tokens.json
│   │   ├── annotated_snowman.jpg
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── preprocessor_config.json
│   │   ├── README.md
│   │   ├── sentencepiece.bpe.model
│   │   ├── snowman.jpg
│   │   ├── snowman.png
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── two_dogs.jpg
│   ├── LAION-Aesthetic-v2/
│   ├── LongCLIP-L/
│   ├── Marigold-Depth-v1-1/
│   ├── Marigold-Normals-v1-1/
│   ├── moondream2/
│   ├── OwlViT-base-patch32/
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── preprocessor_config.json
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── PerceptCLIP_IQA/
│   ├── PickScore_v1/
│   ├── Qwen2.5-14B-Instruct/
│   ├── Real-ESRGAN/
│   ├── SAM2-Hiera-Large/
│   ├── SigLIP-SO400M/
│   ├── StableCascade-Decoder/
│   ├── StableCascade-Prior/
│   ├── T5-XXL/
│   ├── TAESD/
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.bin
│   │   ├── diffusion_pytorch_model.safetensors
│   │   ├── taesd_decoder.safetensors
│   │   └── taesd_encoder.safetensors
│   ├── TAESDXL/
│   └── TrOCR-Large-Printed/
├── research/
│   ├── agi_image/
│   │   ├── alignment/
│   │   │   ├── __init__.py
│   │   │   └── generation_policy.py
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   └── task_taxonomy.py
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── capability_rubric.py
│   │   │   └── holism_score.py
│   │   ├── integrations/
│   │   │   ├── __init__.py
│   │   │   ├── knowledge_bridge.py
│   │   │   └── sample_hints.py
│   │   ├── intents/
│   │   │   ├── __init__.py
│   │   │   ├── decomposition.py
│   │   │   └── goal_spec.py
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   └── episodic.py
│   │   ├── planning/
│   │   │   ├── __init__.py
│   │   │   ├── generation_plan.py
│   │   │   └── iterate_until.py
│   │   ├── reasoning/
│   │   │   ├── __init__.py
│   │   │   ├── counterfactuals.py
│   │   │   └── latent_hypotheses.py
│   │   ├── reflection/
│   │   │   ├── __init__.py
│   │   │   └── meta_controller.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── agent_messages.py
│   │   ├── tooling/
│   │   │   ├── __init__.py
│   │   │   └── tool_hooks.py
│   │   ├── world/
│   │   │   ├── __init__.py
│   │   │   ├── scene_graph.py
│   │   │   └── temporal.py
│   │   └── __init__.py
│   ├── visual_quality/
│   │   ├── __init__.py
│   │   ├── perceptual_proxies.py
│   │   └── rank_and_gate.py
│   ├── __init__.py
│   ├── autoregressive_plans.py
│   ├── creature_character_guidance.py
│   ├── diffusion_noise_structures.py
│   ├── hybrid_sampling_schedules.py
│   ├── latent_agreement.py
│   ├── physics_visual_guidance.py
│   └── quality_timestep_weights.py
├── results/
├── runs/
│   ├── 000-EnhancedDiT-XL-2/
│   │   └── checkpoints/
│   ├── 001-EnhancedDiT-XL-2/
│   │   └── checkpoints/
│   └── 002-EnhancedDiT-XL-2/
│       └── checkpoints/
├── scripts/
│   ├── download/
│   │   ├── download_hf_scaffold.py
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
│   │   │   ├── manifest_enrich.py
│   │   │   ├── manifest_gate.py
│   │   │   └── manifest_paths.py
│   │   ├── dev/
│   │   │   ├── __init__.py
│   │   │   ├── ar_mask_inspect.py
│   │   │   ├── architecture_themes.py
│   │   │   ├── ckpt_info.py
│   │   │   ├── cursorfix.sh
│   │   │   ├── gen_archive_shims.py
│   │   │   ├── generate_sdx_architecture_diagram.py
│   │   │   ├── make_gallery.py
│   │   │   ├── prepare-commit-msg
│   │   │   ├── quick_test.py
│   │   │   ├── refresh_native_exports.py
│   │   │   ├── smoke_imports.py
│   │   │   ├── strip_ai_contributors.py
│   │   │   ├── strip_ai_git_trailers.py
│   │   │   ├── test_style_native_stack.py
│   │   │   └── validate_config_json.py
│   │   ├── export/
│   │   │   ├── __init__.py
│   │   │   ├── export_onnx.py
│   │   │   └── export_safetensors.py
│   │   ├── native/
│   │   │   ├── build_native.ps1
│   │   │   ├── build_native.sh
│   │   │   └── clean_native_builds.ps1
│   │   ├── ops/
│   │   │   ├── __init__.py
│   │   │   ├── agentic_evolve.py
│   │   │   ├── agentic_flywheel.py
│   │   │   ├── agentic_generate.py
│   │   │   ├── agentic_roles.py
│   │   │   ├── auto_improve_loop.py
│   │   │   ├── gen_searcher_bridge.py
│   │   │   ├── hybrid_dit_vit_generate.py
│   │   │   ├── model_soup.py
│   │   │   ├── op_preflight.py
│   │   │   ├── orchestrate_pipeline.py
│   │   │   ├── pretrained_status.py
│   │   │   ├── run_agentic.ps1
│   │   │   ├── run_flywheel.py
│   │   │   ├── startup_readiness.py
│   │   │   ├── superior_auto_loop.py
│   │   │   ├── superior_curate.py
│   │   │   ├── superior_dpo_loop.py
│   │   │   ├── superior_ensemble.py
│   │   │   ├── superior_eval_report.py
│   │   │   ├── superior_generate.py
│   │   │   └── visual_brain_generate.py
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
│   │   ├── training/
│   │   │   ├── mine_preference_pairs.py
│   │   │   ├── noise_schedule_export.py
│   │   │   ├── run_superior_flywheel.ps1
│   │   │   ├── torchrun_ddp_train.ps1
│   │   │   ├── train_consistency_distill.py
│   │   │   ├── train_diffusion_dpo.py
│   │   │   ├── train_flow_grpo.py
│   │   │   ├── train_kd_distill.py
│   │   │   ├── train_ladd_distill.py
│   │   │   ├── train_with_expandable_segments.ps1
│   │   │   └── train_with_expandable_segments.sh
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── _repo_bootstrap.py
│   │   ├── benchmark_suite.py
│   │   ├── book_manifest_check.py
│   │   ├── book_prompt_audit.py
│   │   ├── book_scene_split.py
│   │   ├── complex_prompt_coverage.py
│   │   ├── dit_variant_compare.py
│   │   ├── download_all_danbooru_categorized_tags.py
│   │   ├── edit_inpaint.py
│   │   ├── eval_prompts.py
│   │   ├── explore_styles.py
│   │   ├── fetch_danbooru_tags.py
│   │   ├── image_quality_qc.py
│   │   ├── make_smoke_dataset.py
│   │   ├── merge_danbooru_categorized_tags.py
│   │   ├── normalize_captions.py
│   │   ├── op_pipeline.ps1
│   │   ├── preview_generation_prompt.py
│   │   ├── preview_prompt_stack.py
│   │   ├── prompt_gap_scout.py
│   │   ├── README.md
│   │   ├── seed_explorer.py
│   │   ├── spatial_coverage.py
│   │   ├── split_danbooru_general_tags.py
│   │   ├── training_timestep_preview.py
│   │   ├── visual_memory_patch.py
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
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_advanced_agentic_systems.py
│   ├── test_advanced_quality_systems.py
│   ├── test_agentic_stack.py
│   ├── test_agentic_systems.py
│   ├── test_agi_image_scaffold.py
│   ├── test_ai_image_shortcomings.py
│   ├── test_ar_curriculum.py
│   ├── test_ar_masks_extended.py
│   ├── test_art_mediums.py
│   ├── test_artist_composition_controls.py
│   ├── test_auto_improve_loop_tool.py
│   ├── test_auto_oc.py
│   ├── test_benchmark_suite_tool.py
│   ├── test_book_challenging_content.py
│   ├── test_book_helpers.py
│   ├── test_book_manifest_utils.py
│   ├── test_book_model_readiness.py
│   ├── test_book_project.py
│   ├── test_book_prompt_intel.py
│   ├── test_book_style_authenticity.py
│   ├── test_book_style_fusion.py
│   ├── test_book_text_continuity.py
│   ├── test_book_train_preset.py
│   ├── test_book_training_helpers.py
│   ├── test_book_visual_memory.py
│   ├── test_caption_truncate.py
│   ├── test_cfg_batched.py
│   ├── test_checkpoint_analysis.py
│   ├── test_ckpt_text_stack.py
│   ├── test_cli_entrypoints.py
│   ├── test_composition_brief.py
│   ├── test_consistency_text_continuity.py
│   ├── test_creature_character_prompts.py
│   ├── test_data_pipeline.py
│   ├── test_data_quality_pipeline.py
│   ├── test_detailed_scene_entities.py
│   ├── test_device_perf.py
│   ├── test_diffusion_dpo_loss.py
│   ├── test_diffusion_math.py
│   ├── test_dit_ar_latent_compat.py
│   ├── test_edit_masks.py
│   ├── test_error_handling_utils.py
│   ├── test_eval_prompt_pack.py
│   ├── test_eval_report.py
│   ├── test_frontier.py
│   ├── test_frontier_ideas.py
│   ├── test_generation_pkg_exports.py
│   ├── test_hf_control.py
│   ├── test_hf_index.py
│   ├── test_hf_loaders.py
│   ├── test_hf_reward.py
│   ├── test_hf_scaffold.py
│   ├── test_hf_upscale.py
│   ├── test_human_made.py
│   ├── test_hybrid_dit_vit_generate.py
│   ├── test_image_dissection.py
│   ├── test_image_resize.py
│   ├── test_inference_research_hooks.py
│   ├── test_inference_stages.py
│   ├── test_innovations.py
│   ├── test_jsonl_caption_hygiene_native.py
│   ├── test_jsonutil.py
│   ├── test_latent_edit_helpers.py
│   ├── test_manifest_gate_tool.py
│   ├── test_mine_preference_pairs_tool.py
│   ├── test_model_forward.py
│   ├── test_model_paths_gen_searcher.py
│   ├── test_multi_encoder_encode.py
│   ├── test_multi_instance_scene.py
│   ├── test_naming_compat.py
│   ├── test_native_fast_paths.py
│   ├── test_part_compositing.py
│   ├── test_photo_realism.py
│   ├── test_physics_material_prompts.py
│   ├── test_pixel_perfect.py
│   ├── test_plain_dict_snapshot.py
│   ├── test_prompt_breakdown.py
│   ├── test_prompt_emphasis_import.py
│   ├── test_prompt_lexicon_artist_helpers.py
│   ├── test_prompt_ops_native.py
│   ├── test_prompt_stack.py
│   ├── test_prompt_stack_exports.py
│   ├── test_prompt_training_pkg_lazy.py
│   ├── test_rag_prompt_gen_searcher.py
│   ├── test_refinement_loop.py
│   ├── test_regional_box_prompting.py
│   ├── test_research_sketches.py
│   ├── test_research_systems.py
│   ├── test_run_artifacts.py
│   ├── test_run_baseline_eval.py
│   ├── test_runtime_profiling.py
│   ├── test_sample_cli_passthrough.py
│   ├── test_sample_edit_runner.py
│   ├── test_sampling.py
│   ├── test_sampling_flex.py
│   ├── test_scripts_tools_dispatcher.py
│   ├── test_segmentation_to_mask.py
│   ├── test_simple_latent_generate.py
│   ├── test_startup_readiness_tool.py
│   ├── test_style_artists.py
│   ├── test_style_genome.py
│   ├── test_style_guidance.py
│   ├── test_style_native.py
│   ├── test_superior_extended.py
│   ├── test_superior_stack.py
│   ├── test_superior_wave10.py
│   ├── test_superior_wave11.py
│   ├── test_superior_wave12.py
│   ├── test_superior_wave3.py
│   ├── test_superior_wave4.py
│   ├── test_superior_wave5.py
│   ├── test_superior_wave6.py
│   ├── test_superior_wave7.py
│   ├── test_superior_wave8.py
│   ├── test_superior_wave9.py
│   ├── test_test_time_pick.py
│   ├── test_text_encoder_penta.py
│   ├── test_text_encoder_stack.py
│   ├── test_timestep_curriculum.py
│   ├── test_validate_checkpoint.py
│   ├── test_visual_brain.py
│   ├── test_visual_design.py
│   ├── test_visual_design_full.py
│   ├── test_visual_memory_bridge.py
│   └── test_visual_quality_research.py
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
│   ├── book_train_preset.py
│   ├── enhanced_trainer.py
│   ├── train_args.py
│   └── train_cli_parser.py
├── utils/
│   ├── _archive/
│   │   ├── agentic/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── experience.py
│   │   │   ├── planner.py
│   │   │   ├── reflector.py
│   │   │   ├── roles.py
│   │   │   ├── state.py
│   │   │   └── tools.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── data_analysis.py
│   │   │   └── llm_client.py
│   │   ├── architecture/
│   │   │   ├── __init__.py
│   │   │   ├── ar_block_conditioning.py
│   │   │   ├── ar_block_layout.py
│   │   │   ├── architecture_map.py
│   │   │   ├── dit_architecture.py
│   │   │   └── enhanced_utils.py
│   │   ├── brain/
│   │   │   ├── __init__.py
│   │   │   ├── image_search.py
│   │   │   ├── scene_brief.py
│   │   │   ├── understand.py
│   │   │   └── visual_brain.py
│   │   ├── checkpoint/
│   │   │   ├── __init__.py
│   │   │   ├── checkpoint_loading.py
│   │   │   └── checkpoint_manager.py
│   │   ├── consistency/
│   │   │   ├── __init__.py
│   │   │   ├── character_consistency.py
│   │   │   ├── character_customization.py
│   │   │   ├── character_lock.py
│   │   │   ├── consistency_losses.py
│   │   │   ├── consistency_system.py
│   │   │   └── style_harmonization.py
│   │   ├── modeling/
│   │   │   ├── __init__.py
│   │   │   ├── ckpt_text_stack.py
│   │   │   ├── hf_control.py
│   │   │   ├── hf_index.py
│   │   │   ├── hf_loaders.py
│   │   │   ├── hf_reward.py
│   │   │   ├── hf_scaffold.py
│   │   │   ├── hf_upscale.py
│   │   │   ├── model_paths.py
│   │   │   ├── model_viz.py
│   │   │   ├── multi_encoder_encode.py
│   │   │   ├── nn_inspect.py
│   │   │   ├── t5_segmented_encode.py
│   │   │   ├── text_encoder_bundle.py
│   │   │   └── text_encoder_stack.py
│   │   ├── quantization/
│   │   │   ├── __init__.py
│   │   │   └── nf4_codec.py
│   │   ├── runtime/
│   │   │   ├── __init__.py
│   │   │   ├── jsonutil.py
│   │   │   ├── plain_dict.py
│   │   │   └── profiling.py
│   │   ├── superior/
│   │   │   ├── __init__.py
│   │   │   ├── auto_loop.py
│   │   │   ├── auto_stack.py
│   │   │   ├── block_cache.py
│   │   │   ├── cfg_rejection.py
│   │   │   ├── composite_ranker.py
│   │   │   ├── dbc_cache.py
│   │   │   ├── distill.py
│   │   │   ├── dpo_pipeline.py
│   │   │   ├── dynamic_dit.py
│   │   │   ├── ensemble.py
│   │   │   ├── eval_report.py
│   │   │   ├── feature_cache.py
│   │   │   ├── flywheel.py
│   │   │   ├── frequency_cfg.py
│   │   │   ├── glyph_encoder.py
│   │   │   ├── hard_negative.py
│   │   │   ├── inference_pipeline.py
│   │   │   ├── linear_attention.py
│   │   │   ├── model_soup.py
│   │   │   ├── online_reward.py
│   │   │   ├── prompt_expand.py
│   │   │   ├── quality_gates.py
│   │   │   ├── retrieval.py
│   │   │   ├── reward_scorer.py
│   │   │   ├── self_correct.py
│   │   │   ├── taylor_cache.py
│   │   │   └── vit_mining.py
│   │   └── visual_design/
│   │       ├── __init__.py
│   │       ├── argv.py
│   │       ├── compose.py
│   │       ├── negatives.py
│   │       ├── presets.py
│   │       ├── registry.py
│   │       ├── registry_core.py
│   │       ├── registry_extra.py
│   │       ├── sampling.py
│   │       └── validate.py
│   ├── agentic/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── experience.py
│   │   ├── planner.py
│   │   ├── reflector.py
│   │   ├── roles.py
│   │   ├── state.py
│   │   └── tools.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── ar_block_conditioning.py
│   │   ├── ar_block_layout.py
│   │   ├── architecture_map.py
│   │   ├── dit_architecture.py
│   │   └── enhanced_utils.py
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── image_search.py
│   │   ├── scene_brief.py
│   │   ├── understand.py
│   │   └── visual_brain.py
│   ├── compression/
│   │   └── model_compression.py
│   ├── data_quality/
│   │   ├── cleanup/
│   │   │   ├── __init__.py
│   │   │   └── dataset_cleaner.py
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── distributed/
│   │   └── distributed_inference.py
│   ├── generation/
│   │   ├── spatial_layout_dsl/
│   │   │   ├── __init__.py
│   │   │   └── layout_compiler.py
│   │   ├── __init__.py
│   │   ├── advanced_inference.py
│   │   ├── anatomy_correction.py
│   │   ├── apg_guidance.py
│   │   ├── ar_latent_ops.py
│   │   ├── cfg_batched.py
│   │   ├── cfg_interval.py
│   │   ├── cfg_pp.py
│   │   ├── cfg_zero_star.py
│   │   ├── clip_alignment.py
│   │   ├── clip_reference_embed.py
│   │   ├── dit_ar_latent_compat.py
│   │   ├── edit_masks.py
│   │   ├── eval_prompt_pack.py
│   │   ├── guidance_probe.py
│   │   ├── guidance_session.py
│   │   ├── guidance_stack.py
│   │   ├── image_dissection.py
│   │   ├── image_editing.py
│   │   ├── inference_research_hooks.py
│   │   ├── inference_stages.py
│   │   ├── iterative_refinement.py
│   │   ├── latent_edit_helpers.py
│   │   ├── master_integration.py
│   │   ├── micrograin_stabilizer.py
│   │   ├── multimodal_generation.py
│   │   ├── orchestration.py
│   │   ├── part_compositing.py
│   │   ├── precision_control.py
│   │   ├── rectified_cfgpp.py
│   │   ├── regional_box_prompting.py
│   │   ├── regional_box_sketch.py
│   │   ├── run_artifacts.py
│   │   ├── sample_cli_passthrough.py
│   │   ├── sample_edit_runner.py
│   │   ├── segmentation_to_mask.py
│   │   ├── simple_latent_generate.py
│   │   ├── slg_guidance.py
│   │   ├── speculative_denoise.py
│   │   ├── tcfg.py
│   │   ├── text_rendering.py
│   │   └── zeresfdg.py
│   ├── inference/
│   │   ├── batch_optimization/
│   │   │   ├── __init__.py
│   │   │   └── batch_optimizer.py
│   │   └── inference_optimizer.py
│   ├── monitoring/
│   │   └── performance_profiler.py
│   ├── native/
│   │   ├── __init__.py
│   │   └── kernel_selector.py
│   ├── optimization/
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   └── flash_attention.py
│   │   ├── quantization/
│   │   │   ├── __init__.py
│   │   │   └── quantizer.py
│   │   └── advanced_model_optimization.py
│   ├── prompt/
│   │   ├── prompt_difficulty/
│   │   │   ├── __init__.py
│   │   │   └── difficulty_scorer.py
│   │   ├── stack/
│   │   │   ├── stages/
│   │   │   │   ├── content.py
│   │   │   │   ├── finalize.py
│   │   │   │   ├── guidance.py
│   │   │   │   ├── negative.py
│   │   │   │   ├── special.py
│   │   │   │   └── style_genome.py
│   │   │   ├── __init__.py
│   │   │   ├── clauses.py
│   │   │   ├── context.py
│   │   │   ├── controls.py
│   │   │   ├── intelligence.py
│   │   │   ├── runner.py
│   │   │   ├── sample_bridge.py
│   │   │   └── tokens.py
│   │   ├── __init__.py
│   │   ├── advanced_prompting.py
│   │   ├── auto_oc.py
│   │   ├── composition_brief.py
│   │   ├── content_control_tag_data.py
│   │   ├── content_control_tags.py
│   │   ├── content_control_tags_builtin.py
│   │   ├── content_controls.py
│   │   ├── creative_rag.py
│   │   ├── detailed_scene_entities.py
│   │   ├── fast_paths.py
│   │   ├── multi_instance_scene.py
│   │   ├── multi_subject.py
│   │   ├── neg_filter.py
│   │   ├── originality_augment.py
│   │   ├── photo_realism.py
│   │   ├── prompt_breakdown.py
│   │   ├── prompt_emphasis.py
│   │   ├── prompt_i18n.py
│   │   ├── prompt_layout.py
│   │   ├── prompt_lint.py
│   │   ├── prompt_mutation.py
│   │   ├── rag_prompt.py
│   │   ├── scene_blueprint.py
│   │   ├── shape_scaffold.py
│   │   ├── special_prompt_helpers.py
│   │   ├── style_explore.py
│   │   ├── style_genome.py
│   │   ├── style_genome_chaos.py
│   │   ├── style_inventor.py
│   │   ├── style_memory.py
│   │   └── style_native.py
│   ├── quality/
│   │   ├── adaptive_training/
│   │   │   ├── __init__.py
│   │   │   └── adaptive_trainer.py
│   │   ├── latent_enhancement/
│   │   │   ├── __init__.py
│   │   │   └── latent_improver.py
│   │   ├── quality_prediction/
│   │   │   ├── __init__.py
│   │   │   └── quality_predictor.py
│   │   ├── __init__.py
│   │   ├── artistic_post_process.py
│   │   ├── face_region_enhance.py
│   │   ├── human_made.py
│   │   ├── quality.py
│   │   ├── test_time_pick.py
│   │   └── vit_critic_loop.py
│   ├── speed/
│   │   ├── extreme_quantization.py
│   │   ├── numba_acceleration.py
│   │   └── operator_fusion.py
│   ├── superior/
│   │   ├── __init__.py
│   │   ├── auto_loop.py
│   │   ├── auto_stack.py
│   │   ├── block_cache.py
│   │   ├── cfg_rejection.py
│   │   ├── composite_ranker.py
│   │   ├── dbc_cache.py
│   │   ├── distill.py
│   │   ├── dpo_pipeline.py
│   │   ├── dynamic_dit.py
│   │   ├── ensemble.py
│   │   ├── eval_report.py
│   │   ├── feature_cache.py
│   │   ├── flywheel.py
│   │   ├── frequency_cfg.py
│   │   ├── glyph_encoder.py
│   │   ├── hard_negative.py
│   │   ├── inference_pipeline.py
│   │   ├── linear_attention.py
│   │   ├── model_soup.py
│   │   ├── online_reward.py
│   │   ├── prompt_expand.py
│   │   ├── quality_gates.py
│   │   ├── retrieval.py
│   │   ├── reward_scorer.py
│   │   ├── self_correct.py
│   │   ├── taylor_cache.py
│   │   └── vit_mining.py
│   ├── training/
│   │   ├── contrastive_objectives/
│   │   │   ├── __init__.py
│   │   │   └── contrastive_losses.py
│   │   ├── ensemble_training/
│   │   │   ├── __init__.py
│   │   │   └── ensemble_trainer.py
│   │   ├── hard_negative_mining/
│   │   │   ├── __init__.py
│   │   │   └── hard_negative_miner.py
│   │   ├── __init__.py
│   │   ├── ar_curriculum.py
│   │   ├── auxiliary_structure_supervision.py
│   │   ├── branch_grpo.py
│   │   ├── config_validator.py
│   │   ├── dense_grpo.py
│   │   ├── device_perf.py
│   │   ├── diffusion_dpo_loss.py
│   │   ├── dpo_advanced.py
│   │   ├── dpo_reward_pipeline.py
│   │   ├── error_handling.py
│   │   ├── fast_dataloader.py
│   │   ├── flash_grpo.py
│   │   ├── flow_grpo.py
│   │   ├── grpo_guard.py
│   │   ├── ladd_distillation.py
│   │   ├── metrics.py
│   │   ├── ot_noise_pairing.py
│   │   ├── part_aware_training.py
│   │   ├── preference_image_dataset.py
│   │   ├── preference_jsonl.py
│   │   ├── self_improvement_loop.py
│   │   ├── throughput.py
│   │   ├── timestep_curriculum.py
│   │   └── turning_point_grpo.py
│   ├── __init__.py
│   ├── image_quality_metrics.py
│   ├── image_resize.py
│   └── terminal.py
├── vit_quality/
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
├── compile_flags.txt
├── CONTRIBUTING.md
├── demo.py
├── DEPRECATIONS.md
├── inference.py
├── LICENSE
├── pretrained_status.json
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── pyrightconfig.json
├── README.md
├── requirements-cuda128.txt
├── requirements-perf.txt
├── requirements.txt
├── sample.py
├── SECURITY.md
└── train.py
```

## See also

- [docs/CODEBASE.md](docs/CODEBASE.md) — navigate the tree, `scripts/` layout, contribution rules
- [docs/FILES.md](docs/FILES.md) — full file map

