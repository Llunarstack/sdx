# SDX project structure

> **Auto-generated** — do not edit by hand. Regenerate after moving files:
>
> ```bash
> python -m scripts.tools update_project_structure
> ```
>
> Generated: **2026-07-18 19:55:41 UTC** · max depth: **5** · repo root: `sdx/`
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
│   ├── __init__.py
│   ├── README.md
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
│   ├── manifest_utils.py
│   ├── t2i_dataset.py
│   ├── vector_index_sampler.py
│   └── video_catalog.json
├── datasets/
│   ├── train/
│   └── README.md
├── diffusion/
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
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dpm_solver_pp.py
│   │   ├── flow_ode.py
│   │   └── unipc.py
│   ├── __init__.py
│   ├── bridge_training.py
│   ├── cascaded_multimodal_pipeline.py
│   ├── cfg_schedulers.py
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
│   ├── snr_utils.py
│   ├── spectral_sfp.py
│   └── timestep_sampling.py
├── docs/
│   ├── agentic/
│   │   └── AGENTIC_STACK.md
│   ├── assets/
│   │   └── gallery/
│   ├── brain/
│   │   └── VISUAL_BRAIN.md
│   ├── design/
│   │   └── WEBSITE_DESIGN_BRIEF.md
│   ├── guides/
│   │   ├── BOOK_COMIC_TECH.md
│   │   ├── CHARACTER_CONSISTENCY_IMPLEMENTATION.md
│   │   ├── DANBOORU_HF.md
│   │   ├── EDITING_PHASE.md
│   │   ├── HARDWARE.md
│   │   ├── HF_DATASET_SHORTLIST.md
│   │   ├── IMAGE_CAPTIONING.md
│   │   ├── INTEGRATION.md
│   │   ├── REPRODUCIBILITY.md
│   │   ├── SAMPLING.md
│   │   ├── SMOKE_TRAINING.md
│   │   └── TRAINING_TEXT_TO_PIXELS.md
│   ├── recipes/
│   │   ├── eval_baseline_prompts.md
│   │   ├── fast_training.md
│   │   ├── local_ci_mirror.md
│   │   └── quick_eval_holy_grail.md
│   ├── reference/
│   │   ├── DOMAINS.md
│   │   ├── FILES.md
│   │   ├── PRETRAINED_RECOMMENDED.md
│   │   ├── REGION_CAPTIONS.md
│   │   └── STYLE_ARTIST_TAGS.md
│   ├── releases/
│   │   ├── v0.1.0.md
│   │   ├── v0.2.0.md
│   │   ├── v10-github-release.md
│   │   ├── v10.md
│   │   ├── v11-github-release.md
│   │   ├── v11.md
│   │   ├── v12-github-release.md
│   │   ├── v12.md
│   │   ├── v3.md
│   │   ├── v4.md
│   │   ├── v5.md
│   │   ├── v6.md
│   │   ├── v7.md
│   │   ├── v8-github-release.md
│   │   ├── v8.md
│   │   ├── v9.md
│   │   └── VERSION_COMPARISON.md
│   ├── research/
│   │   ├── AGENTIC_RESEARCH_2026.md
│   │   ├── BLUEPRINTS.md
│   │   ├── DIFFUSION_LEVERAGE_ROADMAP.md
│   │   ├── IMAGE_QUALITY_LEVERS_2026.md
│   │   ├── IMPROVEMENTS.md
│   │   ├── INSPIRATION.md
│   │   ├── LANDSCAPE_2026.md
│   │   ├── MODERN_DIFFUSION.md
│   │   ├── NEXTGEN_SUPERMODEL_ARCHITECTURE.md
│   │   ├── README.md
│   │   ├── SAMPLING_EXPERIMENTS_BACKLOG.md
│   │   └── SUPERIOR_RESEARCH_2026.md
│   ├── AR.md
│   ├── CODEBASE.md
│   ├── GETTING_STARTED.md
│   ├── GLOSSARY.md
│   ├── HOLY_GRAIL_OVERVIEW.md
│   ├── HOW_GENERATION_WORKS.md
│   ├── MODEL_STACK.md
│   ├── NATIVE_AND_SYSTEM_LIBS.md
│   ├── PROMPT_COOKBOOK.md
│   ├── PROMPT_STACK.md
│   ├── QUALITY.md
│   ├── README.md
│   ├── SUPERIOR_STACK.md
│   └── TCIS.md
├── examples/
│   ├── __init__.py
│   ├── book_visual_memory.example.json
│   ├── box_layout.example.json
│   ├── box_layout_sketch.example.json
│   ├── eval_prompts_baseline.json
│   ├── example_character_consistency.py
│   ├── example_style_harmonization.py
│   ├── moodboard.example.json
│   ├── multi_character_scene.example.json
│   ├── prompt_layout.example.json
│   ├── prompt_layout_group_mansion_nsfw.json
│   ├── run_baseline_eval.py
│   ├── scene.example.json
│   ├── scene_continuity.example.json
│   ├── scene_frontier.example.json
│   ├── scene_i2v_control.example.json
│   ├── scene_studio.example.json
│   ├── scene_tier1.example.json
│   ├── style_references.example.json
│   └── video_plan.example.json
├── frontier/
│   ├── adherence/
│   │   ├── __init__.py
│   │   └── token_emphasis.py
│   ├── anatomy/
│   │   ├── __init__.py
│   │   └── body_planner.py
│   ├── archetype/
│   │   ├── __init__.py
│   │   └── symbol_map.py
│   ├── atmosphere/
│   │   ├── __init__.py
│   │   └── volumetric.py
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── dense_diffusion.py
│   │   └── layout_plan.py
│   ├── blend/
│   │   ├── __init__.py
│   │   └── style_dna.py
│   ├── causality/
│   │   ├── __init__.py
│   │   └── physical_plausibility.py
│   ├── chaos/
│   │   ├── __init__.py
│   │   ├── entropy_budget.py
│   │   └── serendipity.py
│   ├── cinema/
│   │   ├── __init__.py
│   │   ├── absence_pulse.py
│   │   ├── shot_grammar.py
│   │   └── video_bridge.py
│   ├── collective/
│   │   ├── __init__.py
│   │   └── crowd_grammar.py
│   ├── compose/
│   │   ├── __init__.py
│   │   └── multi_reference.py
│   ├── composition/
│   │   ├── __init__.py
│   │   └── framing.py
│   ├── constraint/
│   │   ├── __init__.py
│   │   └── creative_limits.py
│   ├── counterfactual/
│   │   ├── __init__.py
│   │   └── preserve_edit.py
│   ├── creatures/
│   │   ├── __init__.py
│   │   └── taxonomy.py
│   ├── economy/
│   │   ├── __init__.py
│   │   └── compute_budget.py
│   ├── era/
│   │   ├── __init__.py
│   │   └── period_accuracy.py
│   ├── focal/
│   │   ├── __init__.py
│   │   └── story_dof.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── genre_collision.py
│   ├── glitch/
│   │   ├── __init__.py
│   │   └── intentional_artifacts.py
│   ├── guidance/
│   │   ├── __init__.py
│   │   ├── dynamic_cfg.py
│   │   └── guidance_interval.py
│   ├── harmony/
│   │   ├── __init__.py
│   │   └── palette.py
│   ├── inverse/
│   │   ├── __init__.py
│   │   └── layout_sketch.py
│   ├── latent/
│   │   └── __init__.py
│   ├── layout/
│   │   ├── __init__.py
│   │   ├── coordinate_bind.py
│   │   ├── lamic_schedule.py
│   │   ├── layout_metrics.py
│   │   └── omost_canvas.py
│   ├── lighting/
│   │   ├── __init__.py
│   │   └── motivated_light.py
│   ├── logic/
│   │   ├── __init__.py
│   │   ├── absence.py
│   │   └── contradiction.py
│   ├── materials/
│   │   ├── __init__.py
│   │   └── surface_truth.py
│   ├── mature/
│   │   ├── __init__.py
│   │   └── mature_guidance.py
│   ├── medium/
│   │   ├── __init__.py
│   │   ├── brush_planner.py
│   │   └── extended_mediums.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── generation_echo.py
│   ├── motion/
│   │   ├── __init__.py
│   │   └── action_freeze.py
│   ├── multiview/
│   │   └── __init__.py
│   ├── mutation/
│   │   ├── __init__.py
│   │   └── prompt_mutator.py
│   ├── narrative/
│   │   ├── __init__.py
│   │   ├── chromatic_field.py
│   │   ├── moment.py
│   │   ├── tension_field.py
│   │   └── witness.py
│   ├── optics/
│   │   ├── __init__.py
│   │   └── lens_character.py
│   ├── paradox/
│   │   ├── __init__.py
│   │   └── beautiful_paradox.py
│   ├── provenance/
│   │   ├── __init__.py
│   │   └── audit_bundle.py
│   ├── realism/
│   │   ├── __init__.py
│   │   ├── anti_slop.py
│   │   └── photoreal_stack.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── fact_rag.py
│   ├── rhythm/
│   │   ├── __init__.py
│   │   └── visual_beat.py
│   ├── safety/
│   │   ├── __init__.py
│   │   └── content_policy.py
│   ├── scale/
│   │   ├── __init__.py
│   │   └── magnitude.py
│   ├── semantics/
│   │   ├── __init__.py
│   │   └── relation_graph.py
│   ├── surreal/
│   │   ├── __init__.py
│   │   └── dream_logic.py
│   ├── synesthesia/
│   │   ├── __init__.py
│   │   └── cross_modal.py
│   ├── temporal/
│   │   ├── __init__.py
│   │   └── storyboard.py
│   ├── typography/
│   │   ├── __init__.py
│   │   └── prompt_glyphs.py
│   ├── uncertainty/
│   │   ├── __init__.py
│   │   └── confidence_gate.py
│   ├── vibe/
│   │   ├── __init__.py
│   │   └── mood_physics.py
│   ├── weathering/
│   │   ├── __init__.py
│   │   └── patina.py
│   ├── world/
│   │   ├── __init__.py
│   │   └── world_bible.py
│   ├── __init__.py
│   ├── engine.py
│   ├── hooks.py
│   ├── imagination.py
│   ├── perfect.py
│   ├── README.md
│   ├── registry.py
│   ├── subject.py
│   └── synthesis.py
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
│   ├── dit_text.py
│   ├── dit_text_variants.py
│   ├── dynamic_patch.py
│   ├── enhanced_dit.py
│   ├── linear_attention.py
│   ├── long_prompt_encoder.py
│   ├── lora.py
│   ├── lora_train.py
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
│   └── vit_next_blocks.py
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
│   ├── video/
│   │   ├── scripts/
│   │   │   └── generate_video.py
│   │   ├── __init__.py
│   │   ├── animation_principles.py
│   │   ├── anticipation_windup.py
│   │   ├── audio_mux.py
│   │   ├── auto_rig.py
│   │   ├── breath_cadence.py
│   │   ├── camera_empathy.py
│   │   ├── camera_rig.py
│   │   ├── camera_stabilize.py
│   │   ├── causal_events.py
│   │   ├── character_memory.py
│   │   ├── chromatic_arc.py
│   │   ├── continuity_validators.py
│   │   ├── controls.py
│   │   ├── counterfactual_beats.py
│   │   ├── deflicker.py
│   │   ├── depth_interpolate.py
│   │   ├── diegetic_focus.py
│   │   ├── director_mode.py
│   │   ├── director_personalities.py
│   │   ├── drift_repair.py
│   │   ├── editor.py
│   │   ├── elements.py
│   │   ├── emotional_contagion.py
│   │   ├── flf2v.py
│   │   ├── flow_consistency.py
│   │   ├── frame_enhance.py
│   │   ├── frontier_compiler.py
│   │   ├── generation_router.py
│   │   ├── helpers.py
│   │   ├── i2v.py
│   │   ├── identity_lock.py
│   │   ├── interpolate.py
│   │   ├── keyframes.py
│   │   ├── kinetic_continuity.py
│   │   ├── layer_stack.py
│   │   ├── mask_propagate.py
│   │   ├── material_memory.py
│   │   ├── mise_en_scene.py
│   │   ├── motif_tracker.py
│   │   ├── motion.py
│   │   ├── motion_beats.py
│   │   ├── motion_brush.py
│   │   ├── motion_library.py
│   │   ├── motion_transfer.py
│   │   ├── narrative_debt.py
│   │   ├── narrative_tension.py
│   │   ├── offscreen_space.py
│   │   ├── parallax_budget.py
│   │   ├── parallel_segments.py
│   │   ├── pipeline.py
│   │   ├── pose_control.py
│   │   ├── post_grade.py
│   │   ├── process_options.py
│   │   ├── provenance.py
│   │   ├── quality.py
│   │   ├── README.md
│   │   ├── reference_sheet.py
│   │   ├── region_motion.py
│   │   ├── rehearsal_pipeline.py
│   │   ├── retrieval.py
│   │   ├── scar_timeline.py
│   │   ├── scene_graph.py
│   │   ├── scene_preflight.py
│   │   ├── screen_direction.py
│   │   ├── segment_processor.py
│   │   ├── segment_retry.py
│   │   ├── semantic_drift.py
│   │   ├── semantic_gravity.py
│   │   ├── shot_planner.py
│   │   ├── silence_map.py
│   │   ├── stinger_frames.py
│   │   ├── stitch.py
│   │   ├── storyboard.py
│   │   ├── studio_compiler.py
│   │   ├── style_engines.py
│   │   ├── t2v.py
│   │   ├── temporal.py
│   │   ├── temporal_echo.py
│   │   ├── thumbnail_rehearsal.py
│   │   ├── timeline.py
│   │   ├── transition_fx.py
│   │   ├── types.py
│   │   ├── velocity_curve.py
│   │   ├── video_io.py
│   │   ├── weather_inertia.py
│   │   ├── witness_lens.py
│   │   └── world_memory.py
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
│   ├── DC-AE-f32c32/
│   │   ├── config.json
│   │   ├── LICENSE.txt
│   │   └── README.md
│   ├── Depth-Anything-V3-Large/
│   │   ├── config.json
│   │   └── README.md
│   ├── DINOv3-ViT-B16/
│   │   ├── LICENSE.md
│   │   ├── README.md
│   │   └── SDX_HUB.json
│   ├── DINOv3-ViT-L16/
│   │   ├── LICENSE.md
│   │   ├── README.md
│   │   └── SDX_HUB.json
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
│   ├── FLUX.2-dev/
│   │   ├── scheduler/
│   │   ├── text_encoder/
│   │   ├── tokenizer/
│   │   ├── LICENSE.md
│   │   ├── README.md
│   │   └── SDX_HUB.json
│   ├── FLUX.2-klein-4B/
│   │   ├── scheduler/
│   │   │   └── scheduler_config.json
│   │   ├── text_encoder/
│   │   │   ├── config.json
│   │   │   ├── generation_config.json
│   │   │   └── model.safetensors.index.json
│   │   ├── tokenizer/
│   │   │   ├── added_tokens.json
│   │   │   ├── chat_template.jinja
│   │   │   ├── merges.txt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── transformer/
│   │   │   └── config.json
│   │   └── vae/
│   │       └── config.json
│   ├── FLUX.2-klein-9B/
│   │   ├── scheduler/
│   │   ├── text_encoder/
│   │   ├── tokenizer/
│   │   ├── transformer/
│   │   ├── LICENSE.md
│   │   ├── README.md
│   │   └── SDX_HUB.json
│   ├── Gemma-3-4B-IT/
│   │   ├── README.md
│   │   └── SDX_HUB.json
│   ├── GenSearcher-8B/
│   ├── GroundingDINO-Base/
│   ├── HPSv3/
│   │   ├── config.json
│   │   └── README.md
│   ├── ImageReward/
│   ├── InternVL3-8B/
│   │   ├── added_tokens.json
│   │   ├── chat_template.jinja
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors.index.json
│   │   ├── preprocessor_config.json
│   │   ├── processor_config.json
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
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
│   ├── moondream3-preview/
│   │   ├── config.json
│   │   ├── LICENSE.md
│   │   ├── model.safetensors.index.json
│   │   └── README.md
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
│   ├── Qwen-Image/
│   │   ├── scheduler/
│   │   │   └── scheduler_config.json
│   │   ├── text_encoder/
│   │   │   ├── config.json
│   │   │   ├── generation_config.json
│   │   │   └── model.safetensors.index.json
│   │   ├── tokenizer/
│   │   │   ├── added_tokens.json
│   │   │   ├── chat_template.jinja
│   │   │   ├── merges.txt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── transformer/
│   │   │   ├── config.json
│   │   │   └── diffusion_pytorch_model.safetensors.index.json
│   │   └── vae/
│   │       └── config.json
│   ├── Qwen-Image-2512/
│   │   ├── scheduler/
│   │   │   └── scheduler_config.json
│   │   ├── text_encoder/
│   │   │   ├── config.json
│   │   │   ├── generation_config.json
│   │   │   └── model.safetensors.index.json
│   │   ├── tokenizer/
│   │   │   ├── added_tokens.json
│   │   │   ├── chat_template.jinja
│   │   │   ├── merges.txt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── transformer/
│   │   │   ├── config.json
│   │   │   └── diffusion_pytorch_model.safetensors.index.json
│   │   └── vae/
│   │       └── config.json
│   ├── Qwen3-14B/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors.index.json
│   │   ├── README.md
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── Qwen3-8B/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors.index.json
│   │   ├── README.md
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── Qwen3-VL-8B-Instruct/
│   │   ├── chat_template.json
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors.index.json
│   │   ├── preprocessor_config.json
│   │   ├── README.md
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── video_preprocessor_config.json
│   │   └── vocab.json
│   ├── Real-ESRGAN/
│   ├── SAM2-Hiera-Large/
│   ├── SigLIP2-SO400M/
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer.model
│   │   └── tokenizer_config.json
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
│   ├── TrOCR-Large-Printed/
│   ├── UMT5-XXL/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model.bin.index.json
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── spiece.model
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── RECOMMENDED.md
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
│   ├── creature_character_guidance.py
│   └── physics_visual_guidance.py
├── results/
├── runpod/
│   ├── lib/
│   │   ├── ensure_repo.sh
│   │   ├── fix_shell.sh
│   │   ├── hf_sites.sh
│   │   ├── install_native.sh
│   │   ├── install_python_deps.sh
│   │   ├── install_scrape_secrets.sh
│   │   ├── install_system_deps.sh
│   │   ├── load_secrets.sh
│   │   ├── train_features.sh
│   │   ├── turbo_hf.sh
│   │   ├── turbo_scrape.sh
│   │   └── verify_env.sh
│   ├── bootstrap.sh
│   ├── budget.sh
│   ├── datasets.sh
│   ├── download.sh
│   ├── env.defaults
│   ├── IMAGE_GEN_PIPELINE.md
│   ├── README.md
│   ├── requirements-extra.txt
│   ├── requirements-runpod.txt
│   ├── run.ps1
│   ├── run.sh
│   ├── sample.sh
│   ├── scrape.sh
│   ├── scrape_stats.sh
│   ├── sdx.sh
│   ├── secret.txt
│   ├── secrets.example.txt
│   ├── setup.ps1
│   ├── setup.sh
│   ├── start.sh
│   ├── status.sh
│   ├── test.ps1
│   ├── test.sh
│   ├── train.sh
│   ├── train_h100.sh
│   ├── train_lora_bank.sh
│   ├── train_ultimate.sh
│   ├── ultimate.sh
│   └── update.sh
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
│   ├── scrape/
│   │   ├── __init__.py
│   │   ├── booru_client.py
│   │   ├── frame_split.py
│   │   ├── gelbooru_auth.py
│   │   ├── media_validate.py
│   │   ├── post_cap.py
│   │   ├── README.md
│   │   ├── rule34xyz_v2.py
│   │   ├── safety.py
│   │   ├── scrape_cli.py
│   │   ├── secrets_config.py
│   │   └── sites.py
│   ├── setup/
│   │   ├── clone_repos.ps1
│   │   ├── clone_repos.sh
│   │   └── runpod_setup.sh
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
│   │   │   ├── generate_sdx_architecture_diagram.py
│   │   │   ├── integration_smoke.py
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
│   │   │   ├── cascade_generate.py
│   │   │   ├── gen_searcher_bridge.py
│   │   │   ├── hybrid_dit_vit_generate.py
│   │   │   ├── model_soup.py
│   │   │   ├── op_preflight.py
│   │   │   ├── orchestrate_pipeline.py
│   │   │   ├── pretrained_status.py
│   │   │   ├── profile_image_cli.py
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
│   │   │   ├── prompt_compose.py
│   │   │   ├── prompt_lint.py
│   │   │   ├── research_image_prompt.py
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
│   │   ├── auto_refine.py
│   │   ├── benchmark_history.py
│   │   ├── benchmark_suite.py
│   │   ├── book_manifest_check.py
│   │   ├── book_prompt_audit.py
│   │   ├── book_scene_split.py
│   │   ├── character_session.py
│   │   ├── complex_prompt_coverage.py
│   │   ├── creative_explore.py
│   │   ├── dit_variant_compare.py
│   │   ├── download_all_danbooru_categorized_tags.py
│   │   ├── edit_inpaint.py
│   │   ├── editing_phase.py
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
│   │   ├── prompt_diff.py
│   │   ├── prompt_gap_scout.py
│   │   ├── README.md
│   │   ├── seed_explorer.py
│   │   ├── spatial_coverage.py
│   │   ├── split_danbooru_general_tags.py
│   │   ├── style_gallery.py
│   │   ├── taste_profile.py
│   │   ├── training_timestep_preview.py
│   │   ├── video_generate.py
│   │   ├── visual_memory_patch.py
│   │   └── vit_inspect.py
│   ├── training/
│   │   ├── hf_download_and_train.py
│   │   ├── hf_export_to_sdx_manifest.py
│   │   └── precompute_latents.py
│   ├── __init__.py
│   ├── cascade_generate.py
│   ├── integration_smoke.py
│   ├── profile_image_cli.py
│   ├── prompt_compose.py
│   ├── README.md
│   ├── research_image_prompt.py
│   └── run_pipeline.py
├── setup/
│   ├── build_artist_index.py
│   ├── build_lora_bank_index.py
│   ├── build_lora_subsets.py
│   ├── build_rag_corpus.py
│   ├── cleanup_scrape_media.py
│   ├── download_datasets.py
│   ├── download_hf_datasets.py
│   ├── download_pretrained.py
│   ├── enrich_manifest_captions.py
│   ├── ensure_repa_encoder.py
│   ├── ensure_t5_safetensors.py
│   ├── hf_dataset_packs.json
│   ├── merge_manifests.py
│   ├── preprocess_control_maps.py
│   ├── sanitize_manifest.py
│   └── tag_manifest_wd.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_agentic_stack.py
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
│   ├── test_editing_phase.py
│   ├── test_enrich_manifest.py
│   ├── test_error_handling_utils.py
│   ├── test_eval_prompt_pack.py
│   ├── test_eval_report.py
│   ├── test_frame_split.py
│   ├── test_frontier.py
│   ├── test_frontier_creative.py
│   ├── test_frontier_horizon.py
│   ├── test_frontier_ideas.py
│   ├── test_frontier_perfect.py
│   ├── test_frontier_subject.py
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
│   ├── test_image_profiler.py
│   ├── test_image_resize.py
│   ├── test_inference_research_hooks.py
│   ├── test_inference_stages.py
│   ├── test_jsonl_caption_hygiene_native.py
│   ├── test_jsonutil.py
│   ├── test_krea_controls.py
│   ├── test_latent_edit_helpers.py
│   ├── test_lora_train.py
│   ├── test_manifest_gate_tool.py
│   ├── test_media_validate.py
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
│   ├── test_post_cap.py
│   ├── test_prompt_breakdown.py
│   ├── test_prompt_composer_artists.py
│   ├── test_prompt_emphasis_import.py
│   ├── test_prompt_lexicon_artist_helpers.py
│   ├── test_prompt_ops_native.py
│   ├── test_prompt_research.py
│   ├── test_prompt_stack.py
│   ├── test_prompt_stack_exports.py
│   ├── test_prompt_train_sample_parity.py
│   ├── test_prompt_training_pkg_lazy.py
│   ├── test_rag_prompt_gen_searcher.py
│   ├── test_regional_box_prompting.py
│   ├── test_reverse_search.py
│   ├── test_rule34xyz_v2.py
│   ├── test_run_artifacts.py
│   ├── test_run_baseline_eval.py
│   ├── test_runtime_profiling.py
│   ├── test_sample_cli_passthrough.py
│   ├── test_sample_edit_runner.py
│   ├── test_sample_features.py
│   ├── test_sampling.py
│   ├── test_sampling_flex.py
│   ├── test_scene_graph.py
│   ├── test_scrape_safety.py
│   ├── test_scripts_tools_dispatcher.py
│   ├── test_segmentation_to_mask.py
│   ├── test_simple_latent_generate.py
│   ├── test_solvers_schedules.py
│   ├── test_startup_readiness_tool.py
│   ├── test_style_artists.py
│   ├── test_style_genome.py
│   ├── test_style_guidance.py
│   ├── test_style_native.py
│   ├── test_superior_extended.py
│   ├── test_superior_stack.py
│   ├── test_superior_waves.py
│   ├── test_superior_waves_torch.py
│   ├── test_test_time_pick.py
│   ├── test_text_encoder_penta.py
│   ├── test_text_encoder_stack.py
│   ├── test_timestep_curriculum.py
│   ├── test_training_wiring.py
│   ├── test_ultimate_pipeline.py
│   ├── test_validate_checkpoint.py
│   ├── test_video_continuity.py
│   ├── test_video_controls.py
│   ├── test_video_frontier.py
│   ├── test_video_perfect.py
│   ├── test_video_pipeline.py
│   ├── test_video_studio.py
│   ├── test_video_tier1.py
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
│   ├── agentic/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── experience.py
│   │   ├── planner.py
│   │   ├── reflector.py
│   │   ├── roles.py
│   │   ├── state.py
│   │   └── tools.py
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
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── image_search.py
│   │   ├── scene_brief.py
│   │   ├── understand.py
│   │   └── visual_brain.py
│   ├── caption/
│   │   ├── __init__.py
│   │   ├── api_keys.py
│   │   ├── danbooru_lookup.py
│   │   ├── e621_lookup.py
│   │   ├── image_profiler.py
│   │   ├── prompt_research.py
│   │   ├── reverse_search.py
│   │   └── wd_tagger.py
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
│   ├── data_quality/
│   │   ├── cleanup/
│   │   │   ├── __init__.py
│   │   │   └── dataset_cleaner.py
│   │   ├── __init__.py
│   │   └── pipeline.py
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
│   │   ├── comfy_export.py
│   │   ├── dit_ar_latent_compat.py
│   │   ├── edit_masks.py
│   │   ├── editing_phase.py
│   │   ├── eval_prompt_pack.py
│   │   ├── guidance_probe.py
│   │   ├── guidance_session.py
│   │   ├── guidance_stack.py
│   │   ├── image_dissection.py
│   │   ├── inference_research_hooks.py
│   │   ├── inference_stages.py
│   │   ├── krea_controls.py
│   │   ├── latent_edit_helpers.py
│   │   ├── micrograin_stabilizer.py
│   │   ├── multimodal_generation.py
│   │   ├── orchestration.py
│   │   ├── part_compositing.py
│   │   ├── per_region_cads.py
│   │   ├── precision_control.py
│   │   ├── rectified_cfgpp.py
│   │   ├── regional_box_prompting.py
│   │   ├── regional_box_sketch.py
│   │   ├── run_artifacts.py
│   │   ├── sample_cli_parser.py
│   │   ├── sample_cli_passthrough.py
│   │   ├── sample_edit_runner.py
│   │   ├── sample_features.py
│   │   ├── sample_helpers.py
│   │   ├── sample_main.py
│   │   ├── segmentation_to_mask.py
│   │   ├── simple_latent_generate.py
│   │   ├── slg_guidance.py
│   │   ├── speculative_denoise.py
│   │   ├── tcfg.py
│   │   ├── text_rendering.py
│   │   └── zeresfdg.py
│   ├── lora/
│   │   ├── __init__.py
│   │   └── lora_bank.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── autoencoder_loading.py
│   │   ├── ckpt_text_stack.py
│   │   ├── hf_control.py
│   │   ├── hf_index.py
│   │   ├── hf_loaders.py
│   │   ├── hf_reward.py
│   │   ├── hf_scaffold.py
│   │   ├── hf_upscale.py
│   │   ├── model_paths.py
│   │   ├── model_viz.py
│   │   ├── multi_encoder_encode.py
│   │   ├── nn_inspect.py
│   │   ├── t5_segmented_encode.py
│   │   ├── text_encoder_bundle.py
│   │   └── text_encoder_stack.py
│   ├── native/
│   │   └── __init__.py
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
│   │   ├── artist_registry.py
│   │   ├── artist_tag.py
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
│   │   ├── prompt_composer.py
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
│   │   └── test_time_pick.py
│   ├── quantization/
│   │   ├── __init__.py
│   │   └── nf4_codec.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── jsonutil.py
│   │   ├── plain_dict.py
│   │   └── profiling.py
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
│   │   ├── branch_grpo.py
│   │   ├── config_validator.py
│   │   ├── dense_grpo.py
│   │   ├── device_perf.py
│   │   ├── diffusion_dpo_loss.py
│   │   ├── dpo_advanced.py
│   │   ├── error_handling.py
│   │   ├── fast_dataloader.py
│   │   ├── flash_grpo.py
│   │   ├── flow_grpo.py
│   │   ├── grpo_guard.py
│   │   ├── ladd_distillation.py
│   │   ├── live_dashboard.py
│   │   ├── metrics.py
│   │   ├── ot_noise_pairing.py
│   │   ├── part_aware_training.py
│   │   ├── preference_image_dataset.py
│   │   ├── preference_jsonl.py
│   │   ├── throughput.py
│   │   ├── timestep_curriculum.py
│   │   └── turning_point_grpo.py
│   ├── visual_design/
│   │   ├── __init__.py
│   │   ├── argv.py
│   │   ├── compose.py
│   │   ├── negatives.py
│   │   ├── presets.py
│   │   ├── registry.py
│   │   ├── registry_core.py
│   │   ├── registry_extra.py
│   │   ├── sampling.py
│   │   └── validate.py
│   ├── __init__.py
│   ├── hf_secrets.py
│   ├── image_quality_metrics.py
│   ├── image_resize.py
│   ├── nt.py
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
├── workflows/
│   └── comfyui/
│       ├── custom_nodes/
│       │   ├── sdx_model_intelligence/
│       │   │   ├── __init__.py
│       │   │   ├── anima_ultimate_nodes.py
│       │   │   ├── correction_nodes.py
│       │   │   ├── nodes.py
│       │   │   ├── README.md
│       │   │   ├── reference_nodes.py
│       │   │   └── sdx_presets.py
│       │   └── sdx_unified_sampler/
│       │       ├── __init__.py
│       │       ├── dit_nodes.py
│       │       └── nodes.py
│       ├── presets/
│       │   ├── civitai_bulk_manifest.json
│       │   ├── civitai_nsfw_positions_manifest.json
│       │   ├── civitai_quality_downloads.json
│       │   ├── downloaded_lora_triggers.json
│       │   ├── inspo_index.json
│       │   ├── lora_compat.json
│       │   ├── model_profiles.json
│       │   ├── multi_character_roster.json
│       │   ├── prompt_adherence_stack.json
│       │   ├── prompt_packs.json
│       │   ├── quality_stack_v2.json
│       │   ├── sampler_bench_illustrious_eps.json
│       │   ├── sampler_scheduler_profiles.json
│       │   └── size_presets.json
│       ├── scripts/
│       │   ├── api_to_ui_workflow.py
│       │   ├── batch_anima_adult_characters.py
│       │   ├── batch_anima_corrected.py
│       │   ├── batch_anima_universal_matrix.py
│       │   ├── batch_anima_variety.py
│       │   ├── batch_anime_solid.py
│       │   ├── batch_anti_ai_style.py
│       │   ├── batch_hd_anime_sex.py
│       │   ├── batch_hd_best.py
│       │   ├── batch_hentai_anima.py
│       │   ├── batch_hentai_best.py
│       │   ├── batch_hot_style_variety.py
│       │   ├── batch_multi_model_generate.py
│       │   ├── batch_quality_first.py
│       │   ├── batch_sdx_corrected.py
│       │   ├── batch_section_generate.log.err
│       │   ├── batch_section_generate.py
│       │   ├── batch_section_generate_v2.py
│       │   ├── batch_v2.log.err
│       │   ├── batch_workflow_v3.py
│       │   ├── build_regional_multichar.py
│       │   ├── civitai_bulk_fetch.py
│       │   ├── civitai_download.ps1
│       │   ├── civitai_nsfw_positions_fetch.py
│       │   ├── collect_web_inspiration.py
│       │   ├── download_weights.ps1
│       │   ├── expand_moodboards.ps1
│       │   ├── fix_sdx_correction.py
│       │   ├── gen_illustrious_impact_hd4.py
│       │   ├── gen_illustrious_regional_hd4.py
│       │   ├── gen_noobai_regional_hd4.py
│       │   ├── install_quality_stack.ps1
│       │   ├── progress_sections.py
│       │   ├── promote_to_aether.py
│       │   ├── quality_research_bench.py
│       │   ├── run_api_workflow.py
│       │   ├── sampler_quality_bench.py
│       │   ├── validate_multi_char_nodes.py
│       │   └── validate_sdx_nodes.py
│       ├── florence2_caption_helper_api.json
│       ├── multi_character_no_bleed_maskbounds_api.json
│       ├── multi_character_no_bleed_stitch_api.json
│       ├── multi_character_no_bleed_v3_api.json
│       ├── multi_character_orgy_futa_furry_v2_api.json
│       ├── MULTI_CHARACTER_README.md
│       ├── multi_character_regional_illustrious_v3_api.json
│       ├── multi_character_regional_trio_v4_api.json
│       ├── multi_character_regional_v4_api.json
│       ├── orgy_positions_futa_furry_grid_v3_api.json
│       ├── orgy_single_canvas_4mask_api.json
│       ├── prompt_adherence_illustrious_api.json
│       ├── PROMPT_ADHERENCE_README.md
│       ├── README.md
│       ├── sdx_ntrmix_solid_api.json
│       ├── ultimate_anima_3d_api.json
│       ├── ultimate_anima_api.json
│       ├── ultimate_anima_dit_silvermoon.json
│       ├── ultimate_anima_dit_silvermoon_api.json
│       ├── ultimate_anima_silvermoon.json
│       ├── ultimate_anima_silvermoon_api.json
│       ├── ultimate_illustrious_api.json
│       ├── ultimate_vpred_api.json
│       ├── unified_ksampler_illustrious_hq_api.json
│       └── wd14_prompt_helper_api.json
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
- [docs/reference/FILES.md](docs/reference/FILES.md) — full file map

