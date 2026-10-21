# SDX project structure

> **Auto-generated** — do not edit by hand. Regenerate after moving files:
>
> ```bash
> python -m scripts.tools update_project_structure
> ```
>
> Generated: **2026-10-21 22:40:33 UTC** · max depth: **5** · repo root: `sdx/`
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
│   │   ├── _lexicon.py
│   │   ├── agentic_stack.py
│   │   ├── ai_image_shortcomings.py
│   │   ├── art_mediums.py
│   │   ├── creature_character_prompts.py
│   │   ├── model_presets.py
│   │   ├── physics_material_prompts.py
│   │   ├── prompt_domains.py
│   │   ├── quality_policy.py
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
│   ├── caption_utils.py
│   ├── enhanced_dataset.py
│   ├── manifest_utils.py
│   ├── t2i_dataset.py
│   └── video_catalog.json
├── datasets/
│   ├── train/
│   └── README.md
├── diffusion/
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── apex.py
│   │   ├── base.py
│   │   ├── dpm_solver_pp.py
│   │   ├── flow_ode.py
│   │   └── unipc.py
│   ├── __init__.py
│   ├── bridge_training.py
│   ├── cfg_schedulers.py
│   ├── flow_matching.py
│   ├── gaussian_diffusion.py
│   ├── inference_timesteps.py
│   ├── pixel_perfect.py
│   ├── README.md
│   ├── sampling.py
│   ├── sampling_utils.py
│   ├── schedules.py
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
│   │   ├── RSI_FEEDBACK.md
│   │   ├── SAMPLING.md
│   │   ├── SMOKE_TRAINING.md
│   │   ├── TRAINING_TEXT_TO_PIXELS.md
│   │   └── VIDEOMAX.md
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
│   │   ├── v13-github-release.md
│   │   ├── v13.md
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
│   │   ├── APEX_SAMPLER.md
│   │   ├── BLUEPRINTS.md
│   │   ├── CORPUS_AND_CAST.md
│   │   ├── DIFFUSION_LEVERAGE_ROADMAP.md
│   │   ├── IMAGE_QUALITY_LEVERS_2026.md
│   │   ├── IMPROVEMENTS.md
│   │   ├── INSPIRATION.md
│   │   ├── INVENTION_STACK.md
│   │   ├── LANDSCAPE_2026.md
│   │   ├── MODERN_DIFFUSION.md
│   │   ├── NEXTGEN_SUPERMODEL_ARCHITECTURE.md
│   │   ├── README.md
│   │   ├── SAMPLING_EXPERIMENTS_BACKLOG.md
│   │   └── SUPERIOR_RESEARCH_2026.md
│   ├── AR.md
│   ├── CODEBASE.md
│   ├── COMPETITIVE_ANALYSIS.md
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
│   ├── box_layout_library_hall.example.json
│   ├── box_layout_sketch.example.json
│   ├── creative_copilot.example.json
│   ├── eval_prompts_baseline.json
│   ├── example_character_consistency.py
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
│   ├── __init__.py
│   ├── attention.py
│   ├── engine.py
│   ├── hooks.py
│   ├── imagination.py
│   ├── layout.py
│   ├── perfect.py
│   ├── README.md
│   ├── registry.py
│   ├── subject.py
│   └── synthesis.py
├── models/
│   ├── __init__.py
│   ├── ar_masks_extended.py
│   ├── attention.py
│   ├── controlnet.py
│   ├── dit.py
│   ├── dit_text.py
│   ├── enhanced_dit.py
│   ├── hybrid_dit.py
│   ├── layout_attn_context.py
│   ├── linear_attention.py
│   ├── lora.py
│   ├── lora_train.py
│   ├── moe.py
│   ├── permanent_video_dit.py
│   ├── prompt_adherence.py
│   ├── rae_latent_bridge.py
│   ├── reference_token_projection.py
│   ├── rope2d.py
│   ├── superior_vit.py
│   ├── taca.py
│   └── video_dit.py
├── native/
│   ├── _experimental/
│   │   ├── c/
│   │   │   ├── include/
│   │   │   │   ├── sdx_c_buffer_stats.h
│   │   │   │   ├── sdx_c_cfg_combine.h
│   │   │   │   └── sdx_c_image_metrics.h
│   │   │   ├── src/
│   │   │   │   ├── sdx_c_buffer_stats.c
│   │   │   │   ├── sdx_c_cfg_combine.c
│   │   │   │   └── sdx_c_image_metrics.c
│   │   │   └── compile_flags.txt
│   │   ├── python/
│   │   │   ├── sdx_native/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── beta_schedules_native.py
│   │   │   │   ├── cuda_style_pick_native.py
│   │   │   │   ├── diffusion_math_native.py
│   │   │   │   ├── gaussian_blur_native.py
│   │   │   │   ├── image_metrics_native.py
│   │   │   │   ├── image_ops_native.py
│   │   │   │   ├── inference_timesteps_native.py
│   │   │   │   ├── jsonl_caption_hygiene.py
│   │   │   │   ├── latent_geometry.py
│   │   │   │   ├── mask_ops_native.py
│   │   │   │   ├── native_tools.py
│   │   │   │   ├── quality_scorers_native.py
│   │   │   │   ├── style_ops_native.py
│   │   │   │   └── text_hygiene.py
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
│   │   ├── build/
│   │   │   ├── ALL_BUILD.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── CMakeFiles/
│   │   │   │   ├── 4.2.3/
│   │   │   │   ├── a139b2be0e1c82f5f173f5b9c529a3a5/
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
│   │   │   │   ├── sdx_c_cfg_combine.exp
│   │   │   │   ├── sdx_c_cfg_combine.lib
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
│   │   │   │   ├── sdx_rmsnorm_rows_cpu.lib
│   │   │   │   ├── sdx_score_ops.exp
│   │   │   │   └── sdx_score_ops.lib
│   │   │   ├── sdx_beta_schedules.dir/
│   │   │   │   ├── Debug/
│   │   │   │   ├── MinSizeRel/
│   │   │   │   ├── Release/
│   │   │   │   └── RelWithDebInfo/
│   │   │   ├── sdx_c_cfg_combine.dir/
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
│   │   │   ├── sdx_score_ops.dir/
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
│   │   │   ├── sdx_c_cfg_combine.vcxproj
│   │   │   ├── sdx_c_cfg_combine.vcxproj.filters
│   │   │   ├── sdx_fnv64_file.vcxproj
│   │   │   ├── sdx_fnv64_file.vcxproj.filters
│   │   │   ├── sdx_image_metrics.vcxproj
│   │   │   ├── sdx_image_metrics.vcxproj.filters
│   │   │   ├── sdx_inference_timesteps.vcxproj
│   │   │   ├── sdx_inference_timesteps.vcxproj.filters
│   │   │   ├── sdx_latent.slnx
│   │   │   ├── sdx_latent.vcxproj
│   │   │   ├── sdx_latent.vcxproj.filters
│   │   │   ├── sdx_line_stats.vcxproj
│   │   │   ├── sdx_line_stats.vcxproj.filters
│   │   │   ├── sdx_mask_ops.vcxproj
│   │   │   ├── sdx_mask_ops.vcxproj.filters
│   │   │   ├── sdx_rmsnorm_rows_cpu.vcxproj
│   │   │   ├── sdx_rmsnorm_rows_cpu.vcxproj.filters
│   │   │   ├── sdx_score_ops.vcxproj
│   │   │   ├── sdx_score_ops.vcxproj.filters
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
│   │   ├── sdx-apex-schedule/
│   │   │   └── target/
│   │   │       ├── debug/
│   │   │       ├── release/
│   │   │       └── CACHEDIR.TAG
│   │   ├── sdx-canny-ops/
│   │   │   ├── src/
│   │   │   │   └── lib.rs
│   │   │   ├── target/
│   │   │   │   ├── debug/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── sdx-diffusion-math/
│   │   │   ├── src/
│   │   │   │   └── lib.rs
│   │   │   ├── target/
│   │   │   │   ├── debug/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── sdx-genesis-ops/
│   │   │   └── target/
│   │   │       ├── debug/
│   │   │       └── CACHEDIR.TAG
│   │   ├── sdx-image-metrics/
│   │   │   ├── src/
│   │   │   │   └── main.rs
│   │   │   ├── target/
│   │   │   │   ├── debug/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── sdx-image-ops/
│   │   │   ├── src/
│   │   │   │   └── lib.rs
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
│   │   ├── sdx-quality-scorers/
│   │   │   ├── src/
│   │   │   │   └── lib.rs
│   │   │   ├── target/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── sdx-style-embed/
│   │   │   ├── src/
│   │   │   │   └── lib.rs
│   │   │   ├── target/
│   │   │   │   ├── release/
│   │   │   │   └── CACHEDIR.TAG
│   │   │   ├── Cargo.lock
│   │   │   └── Cargo.toml
│   │   ├── src/
│   │   │   ├── advanced.rs
│   │   │   ├── lib.rs
│   │   │   ├── main.rs
│   │   │   └── py_module.rs
│   │   ├── target/
│   │   │   ├── debug/
│   │   │   │   ├── build/
│   │   │   │   ├── deps/
│   │   │   │   ├── examples/
│   │   │   │   └── incremental/
│   │   │   └── CACHEDIR.TAG
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
│   │   └── visual_memory.py
│   ├── image_gen/
│   │   └── README.md
│   ├── video/
│   │   ├── scripts/
│   │   │   └── generate_video.py
│   │   ├── __init__.py
│   │   ├── continuity_validators.py
│   │   ├── controls.py
│   │   ├── frontier_compiler.py
│   │   ├── interpolate.py
│   │   ├── motion.py
│   │   ├── motion_grammar.py
│   │   ├── neural_sample.py
│   │   ├── pipeline.py
│   │   ├── quality.py
│   │   ├── README.md
│   │   ├── retrieval.py
│   │   ├── scene_graph.py
│   │   ├── segment_processor.py
│   │   ├── shot_planner.py
│   │   ├── studio_compiler.py
│   │   ├── types.py
│   │   └── video_io.py
│   ├── __init__.py
│   └── README.md
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
│   │   └── download_revolutionary_stack.py
│   ├── enhanced/
│   │   ├── README.md
│   │   ├── sample_enhanced.py
│   │   ├── save_model_checkpoint.py
│   │   └── train_enhanced.py
│   ├── scrape/
│   │   ├── __init__.py
│   │   ├── booru_client.py
│   │   ├── frame_split.py
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
│   │   └── clone_repos.sh
│   ├── tools/
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── ar_tag_manifest.py
│   │   │   ├── caption_hygiene.py
│   │   │   ├── corpus_retrieve_cli.py
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
│   │   │   ├── smoke_imports.py
│   │   │   ├── strip_ai_contributors.py
│   │   │   ├── strip_ai_git_trailers.py
│   │   │   ├── test_style_native_stack.py
│   │   │   └── validate_config_json.py
│   │   ├── eval/
│   │   │   ├── __init__.py
│   │   │   ├── prompt_suite.py
│   │   │   ├── run_eval.py
│   │   │   └── scorers.py
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
│   │   │   ├── agentic.py
│   │   │   ├── agentic_stills.py
│   │   │   ├── auto_improve_loop.py
│   │   │   ├── cascade_generate.py
│   │   │   ├── creative_copilot.py
│   │   │   ├── feedback_cli.py
│   │   │   ├── gen_searcher_bridge.py
│   │   │   ├── hybrid_dit_vit_generate.py
│   │   │   ├── invention_lab.py
│   │   │   ├── model_soup.py
│   │   │   ├── op_preflight.py
│   │   │   ├── orchestrate_pipeline.py
│   │   │   ├── perfect_gen.py
│   │   │   ├── preference_flywheel.py
│   │   │   ├── pretrained_status.py
│   │   │   ├── profile_image_cli.py
│   │   │   ├── quality_inventory.py
│   │   │   ├── rsi_loop.py
│   │   │   ├── run_agentic.ps1
│   │   │   ├── scene_compile.py
│   │   │   ├── startup_readiness.py
│   │   │   ├── superior.py
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
│   │   │   ├── train_reference_adapter.py
│   │   │   ├── train_style_lora_oneshot.py
│   │   │   ├── train_video.py
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
│   ├── README.md
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
│   ├── test_agentic_stills.py
│   ├── test_ai.py
│   ├── test_apex_sampler.py
│   ├── test_ar_curriculum.py
│   ├── test_ar_masks_extended.py
│   ├── test_architecture_consistency.py
│   ├── test_art.py
│   ├── test_artist_composition_controls.py
│   ├── test_auto_improve_loop_tool.py
│   ├── test_auto_oc.py
│   ├── test_benchmark_suite_tool.py
│   ├── test_book.py
│   ├── test_camera_isp.py
│   ├── test_caption_truncate.py
│   ├── test_caption_utils.py
│   ├── test_cfg_batched.py
│   ├── test_checkpoint_analysis.py
│   ├── test_cli_entrypoints.py
│   ├── test_composition_brief.py
│   ├── test_consistency_text_continuity.py
│   ├── test_corpus_cast_edit.py
│   ├── test_creative_copilot.py
│   ├── test_creature_character_prompts.py
│   ├── test_data_pipeline.py
│   ├── test_data_quality_pipeline.py
│   ├── test_detailed_scene_entities.py
│   ├── test_device_perf.py
│   ├── test_diffusion.py
│   ├── test_diffusion_amp.py
│   ├── test_edit_masks.py
│   ├── test_editing_phase.py
│   ├── test_enhanced_spatial_control.py
│   ├── test_enrich_manifest.py
│   ├── test_error_handling_utils.py
│   ├── test_eval.py
│   ├── test_eval_harness.py
│   ├── test_feedback_rsi.py
│   ├── test_flow_matching.py
│   ├── test_frame_split.py
│   ├── test_frontier.py
│   ├── test_glyph_and_contact.py
│   ├── test_guidance_stack.py
│   ├── test_hf.py
│   ├── test_human_likeness.py
│   ├── test_human_made.py
│   ├── test_hybrid_dit.py
│   ├── test_hybrid_dit_vit_generate.py
│   ├── test_image.py
│   ├── test_invention_lab.py
│   ├── test_invention_wiring.py
│   ├── test_jsonl_caption_hygiene_native.py
│   ├── test_jsonutil.py
│   ├── test_krea_controls.py
│   ├── test_lora_train.py
│   ├── test_manifest_gate_tool.py
│   ├── test_media_validate.py
│   ├── test_mine_preference_pairs_tool.py
│   ├── test_minor_guard.py
│   ├── test_model.py
│   ├── test_models_regressions.py
│   ├── test_multi_encoder_encode.py
│   ├── test_multi_instance_scene.py
│   ├── test_naming_compat.py
│   ├── test_native_fast_paths.py
│   ├── test_part_compositing.py
│   ├── test_perfect_gen.py
│   ├── test_perfect_model_helpers.py
│   ├── test_photo_realism.py
│   ├── test_physics_material_prompts.py
│   ├── test_pixel_perfect.py
│   ├── test_plain_dict_snapshot.py
│   ├── test_post_cap.py
│   ├── test_preference_rl.py
│   ├── test_prompt.py
│   ├── test_prompt_tags.py
│   ├── test_quality_stack.py
│   ├── test_rag_prompt_gen_searcher.py
│   ├── test_regional_box_prompting.py
│   ├── test_resolution.py
│   ├── test_reverse_search.py
│   ├── test_rule34xyz_v2.py
│   ├── test_run_baseline_eval.py
│   ├── test_runtime_profiling.py
│   ├── test_sample.py
│   ├── test_samplers.py
│   ├── test_sampling.py
│   ├── test_sampling_flex.py
│   ├── test_scene_graph.py
│   ├── test_scene_physics.py
│   ├── test_scrape_safety.py
│   ├── test_scripts_tools_dispatcher.py
│   ├── test_segmentation_to_mask.py
│   ├── test_simple_latent_generate.py
│   ├── test_solvers_schedules.py
│   ├── test_srpo_reward.py
│   ├── test_startup_readiness_tool.py
│   ├── test_style.py
│   ├── test_superior.py
│   ├── test_superior_vit_runnable.py
│   ├── test_t2i_dataset.py
│   ├── test_test_time_pick.py
│   ├── test_text_encoder_penta.py
│   ├── test_text_encoder_stack.py
│   ├── test_timestep_curriculum.py
│   ├── test_training.py
│   ├── test_ultimate_pipeline.py
│   ├── test_variation_guard.py
│   ├── test_video.py
│   ├── test_video_perfect.py
│   ├── test_videomax.py
│   ├── test_visual_brain.py
│   ├── test_visual_design.py
│   ├── test_vit_scorer.py
│   └── test_vit_text_encoder.py
├── toolkit/
│   ├── extras/
│   │   └── requirements-suggested.txt
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
│   │   ├── gen_interview.py
│   │   ├── planner.py
│   │   ├── post_gen_critique.py
│   │   ├── reflector.py
│   │   ├── roles.py
│   │   ├── state.py
│   │   └── tools.py
│   ├── ai_accel/
│   │   └── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── llm_client.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── ar_block_conditioning.py
│   │   ├── architecture_map.py
│   │   └── dit_architecture.py
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── facet_ref_search.py
│   │   ├── image_search.py
│   │   ├── memory_box.py
│   │   ├── scene_brief.py
│   │   ├── understand.py
│   │   └── visual_brain.py
│   ├── caption/
│   │   ├── __init__.py
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
│   │   ├── consistency_losses.py
│   │   └── style_harmonization.py
│   ├── eval/
│   ├── generation/
│   │   ├── inventions/
│   │   │   ├── __init__.py
│   │   │   ├── bindlock.py
│   │   │   ├── countgate.py
│   │   │   ├── failure_oracle.py
│   │   │   ├── negatron.py
│   │   │   ├── priors.py
│   │   │   ├── stack.py
│   │   │   └── wire.py
│   │   ├── __init__.py
│   │   ├── agentic_stills.py
│   │   ├── cfg_batched.py
│   │   ├── clip_alignment.py
│   │   ├── comfy_export.py
│   │   ├── corpus_retrieve.py
│   │   ├── creative_copilot.py
│   │   ├── edit_masks.py
│   │   ├── edit_skills.py
│   │   ├── editing_phase.py
│   │   ├── frontier_consume.py
│   │   ├── guidance_probe.py
│   │   ├── guidance_stack.py
│   │   ├── image_dissection.py
│   │   ├── krea_controls.py
│   │   ├── latent_filters.py
│   │   ├── multi_char_cast.py
│   │   ├── part_compositing.py
│   │   ├── perfect_gen.py
│   │   ├── quality_stack.py
│   │   ├── regional_box_prompting.py
│   │   ├── regional_box_sketch.py
│   │   ├── sample_cli_parser.py
│   │   ├── sample_cli_passthrough.py
│   │   ├── sample_conditioning.py
│   │   ├── sample_edit_runner.py
│   │   ├── sample_features.py
│   │   ├── sample_helpers.py
│   │   ├── sample_main.py
│   │   ├── sample_output.py
│   │   ├── sample_prompt.py
│   │   ├── sample_sampling.py
│   │   ├── sample_setup.py
│   │   ├── segmentation_to_mask.py
│   │   ├── simple_latent_generate.py
│   │   ├── text_rendering.py
│   │   └── user_taste.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── autoencoder_loading.py
│   │   ├── hf_control.py
│   │   ├── hf_loaders.py
│   │   ├── hf_reward.py
│   │   ├── hf_scaffold.py
│   │   ├── model_paths.py
│   │   ├── model_viz.py
│   │   ├── multi_encoder_encode.py
│   │   ├── text_encoder_bundle.py
│   │   └── text_encoder_stack.py
│   ├── prompt/
│   │   ├── stack/
│   │   │   ├── stages/
│   │   │   │   ├── budget.py
│   │   │   │   ├── content.py
│   │   │   │   ├── finalize.py
│   │   │   │   ├── guidance.py
│   │   │   │   └── negative.py
│   │   │   ├── __init__.py
│   │   │   ├── clauses.py
│   │   │   ├── context.py
│   │   │   ├── controls.py
│   │   │   ├── intelligence.py
│   │   │   ├── runner.py
│   │   │   └── sample_bridge.py
│   │   ├── __init__.py
│   │   ├── advanced_prompting.py
│   │   ├── art_direction.py
│   │   ├── artist_registry.py
│   │   ├── auto_oc.py
│   │   ├── composition_brief.py
│   │   ├── content_control_tags_builtin.py
│   │   ├── content_controls.py
│   │   ├── creative_rag.py
│   │   ├── detailed_scene_entities.py
│   │   ├── facet_decompose.py
│   │   ├── multi_instance_scene.py
│   │   ├── multi_subject.py
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
│   │   ├── style_native.py
│   │   ├── subject_addons.py
│   │   └── tags.py
│   ├── quality/
│   │   ├── photoops/
│   │   ├── __init__.py
│   │   ├── ai_look.py
│   │   ├── architecture_consistency.py
│   │   ├── art_critic.py
│   │   ├── artistic_post_process.py
│   │   ├── camera_isp.py
│   │   ├── contact_shadow.py
│   │   ├── face_region_enhance.py
│   │   ├── hand_verifier.py
│   │   ├── human_likeness.py
│   │   ├── human_made.py
│   │   ├── quality.py
│   │   ├── scene_physics.py
│   │   ├── test_time_pick.py
│   │   └── variation_guard.py
│   ├── safety/
│   │   ├── __init__.py
│   │   └── minor_guard.py
│   ├── superior/
│   │   ├── __init__.py
│   │   ├── auto_loop.py
│   │   ├── caches.py
│   │   └── inference_pipeline.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ar_curriculum.py
│   │   ├── config_validator.py
│   │   ├── device_perf.py
│   │   ├── error_handling.py
│   │   ├── fast_dataloader.py
│   │   ├── feedback_bus.py
│   │   ├── flow_grpo.py
│   │   ├── grpo.py
│   │   ├── ladd_distillation.py
│   │   ├── live_dashboard.py
│   │   ├── metrics.py
│   │   ├── ot_noise_pairing.py
│   │   ├── part_aware_training.py
│   │   ├── preference.py
│   │   ├── throughput.py
│   │   └── timestep_curriculum.py
│   ├── __init__.py
│   ├── data_quality.py
│   ├── hf_secrets.py
│   ├── image_resize.py
│   ├── lora_bank.py
│   ├── native.py
│   ├── runtime.py
│   ├── terminal.py
│   └── visual_design.py
├── vit_quality/
│   ├── __init__.py
│   ├── checkpoint_utils.py
│   ├── dataset.py
│   ├── infer.py
│   ├── model.py
│   ├── text_encoder.py
│   └── train.py
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
├── train.py
└── uv.lock
```

## See also

- [docs/CODEBASE.md](docs/CODEBASE.md) — navigate the tree, `scripts/` layout, contribution rules
- [docs/reference/FILES.md](docs/reference/FILES.md) — full file map

