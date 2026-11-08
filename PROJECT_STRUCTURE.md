# SDX project structure

> **Auto-generated** — do not edit by hand. Regenerate after moving files:
>
> ```bash
> python -m scripts.tools update_project_structure
> ```
>
> Generated: **2026-11-08 02:44:14 UTC** · max depth: **5** · repo root: `sdx/`
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
│   ├── __init__.py
│   ├── presets.py
│   ├── prompts.py
│   ├── styles.py
│   └── train_config.py
├── data/
│   ├── danbooru/
│   │   └── README.md
│   ├── style_genomes/
│   │   └── explore_manifest.jsonl
│   ├── __init__.py
│   ├── caption_utils.py
│   ├── manifest_utils.py
│   ├── t2i_dataset.py
│   └── video_catalog.json
├── datasets/
│   ├── train/
│   └── README.md
├── diffusion/
│   ├── __init__.py
│   ├── flow_matching.py
│   ├── gaussian_diffusion.py
│   ├── README.md
│   ├── sampling.py
│   ├── schedules.py
│   └── solvers.py
├── docs/
│   ├── assets/
│   │   └── gallery/
│   ├── CHANGELOG.md
│   ├── CODEBASE.md
│   ├── FILES.md
│   ├── GETTING_STARTED.md
│   ├── GLOSSARY.md
│   ├── GUIDES.md
│   ├── HOW_GENERATION_WORKS.md
│   ├── MODEL_STACK.md
│   ├── PROMPT_STACK.md
│   ├── QUALITY.md
│   ├── README.md
│   └── RESEARCH.md
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
│   ├── engine.py
│   ├── imagination.py
│   ├── layout.py
│   ├── multiview.py
│   ├── planning.py
│   ├── README.md
│   └── registry.py
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── conditioning.py
│   ├── dit.py
│   ├── dit_text.py
│   ├── dit_variants.py
│   ├── lora.py
│   └── video_dit.py
├── native/
│   ├── _experimental/
│   │   ├── python/
│   │   │   └── sdx_native/
│   │   │       ├── __init__.py
│   │   │       ├── ffi.py
│   │   │       ├── native_tools.py
│   │   │       └── text_hygiene.py
│   │   └── zig/
│   │       ├── sdx-linecrc/
│   │       │   ├── src/
│   │       │   └── build.zig
│   │       └── sdx-pathstat/
│   │           ├── src/
│   │           └── build.zig
│   ├── cpp/
│   │   ├── include/
│   │   │   └── sdx.h
│   │   ├── src/
│   │   │   ├── sdx_cpu.cpp
│   │   │   └── sdx_cuda.cu
│   │   └── CMakeLists.txt
│   ├── go/
│   │   └── sdx-manifest/
│   │       ├── explore.go
│   │       ├── go.mod
│   │       └── main.go
│   ├── rust/
│   │   ├── sdx-apex-schedule/
│   │   ├── sdx-genesis-ops/
│   │   ├── src/
│   │   │   ├── bin/
│   │   │   │   ├── sdx-jsonl-tools.rs
│   │   │   │   └── sdx-noise-schedule.rs
│   │   │   └── lib.rs
│   │   ├── build.rs
│   │   ├── Cargo.lock
│   │   └── Cargo.toml
│   ├── .gitignore
│   └── README.md
├── pipelines/
│   ├── book_comic/
│   │   ├── scripts/
│   │   │   ├── generate_book.py
│   │   │   ├── prepare_and_train_book.py
│   │   │   └── train_book_model.py
│   │   ├── __init__.py
│   │   ├── prompt_lexicon.py
│   │   ├── README.md
│   │   ├── render.py
│   │   ├── story.py
│   │   └── training.py
│   ├── image_gen/
│   │   └── README.md
│   ├── video/
│   │   ├── scripts/
│   │   │   └── generate_video.py
│   │   ├── __init__.py
│   │   ├── controls.py
│   │   ├── frontier_compiler.py
│   │   ├── motion.py
│   │   ├── neural_sample.py
│   │   ├── pipeline.py
│   │   ├── quality.py
│   │   ├── README.md
│   │   ├── scene.py
│   │   ├── segment_processor.py
│   │   ├── studio.py
│   │   └── video_io.py
│   ├── __init__.py
│   └── README.md
├── runpod/
│   ├── bootstrap.sh
│   ├── datasets.sh
│   ├── download.sh
│   ├── env.defaults
│   ├── lib.sh
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
│   ├── train_lora_bank.sh
│   ├── ultimate.sh
│   ├── update.sh
│   └── verify_env.sh
├── runs/
│   ├── 000-EnhancedDiT-XL-2/
│   │   └── checkpoints/
│   ├── 001-EnhancedDiT-XL-2/
│   │   └── checkpoints/
│   └── 002-EnhancedDiT-XL-2/
│       └── checkpoints/
├── scripts/
│   ├── setup/
│   │   ├── clone_repos.ps1
│   │   └── clone_repos.sh
│   ├── tools/
│   │   ├── dev/
│   │   │   ├── __init__.py
│   │   │   ├── architecture_themes.py
│   │   │   ├── generate_sdx_architecture_diagram.py
│   │   │   ├── integration_smoke.py
│   │   │   ├── prepare-commit-msg
│   │   │   ├── quick_test.py
│   │   │   ├── smoke_imports.py
│   │   │   ├── strip_ai_contributors.py
│   │   │   ├── strip_ai_git_trailers.py
│   │   │   ├── test_style_native_stack.py
│   │   │   └── validate_config_json.py
│   │   ├── native/
│   │   │   ├── build_native.ps1
│   │   │   ├── build_native.sh
│   │   │   └── clean_native_builds.ps1
│   │   ├── ops/
│   │   │   └── __init__.py
│   │   ├── prompt/
│   │   │   └── __init__.py
│   │   ├── repo/
│   │   │   ├── __init__.py
│   │   │   ├── clean_repo_artifacts.py
│   │   │   ├── update_project_structure.py
│   │   │   └── verify_doc_links.py
│   │   ├── training/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── _repo_bootstrap.py
│   │   ├── agentic.py
│   │   ├── bench.py
│   │   ├── book.py
│   │   ├── captions.py
│   │   ├── ckpt.py
│   │   ├── danbooru.py
│   │   ├── enhanced.py
│   │   ├── evals.py
│   │   ├── generate.py
│   │   ├── manifest.py
│   │   ├── prompts.py
│   │   ├── README.md
│   │   ├── status.py
│   │   ├── superior.py
│   │   └── trainers.py
│   ├── training/
│   ├── __init__.py
│   ├── README.md
│   ├── run_pipeline.py
│   └── scrape.py
├── setup/
│   ├── download.py
│   ├── hf_dataset_packs.json
│   ├── indexes.py
│   └── manifests.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_agentic.py
│   ├── test_book.py
│   ├── test_data.py
│   ├── test_diffusion.py
│   ├── test_eval.py
│   ├── test_frontier.py
│   ├── test_image.py
│   ├── test_internal_imports.py
│   ├── test_minor_guard.py
│   ├── test_model.py
│   ├── test_prompt.py
│   ├── test_quality.py
│   ├── test_runtime.py
│   ├── test_sample.py
│   ├── test_sampling.py
│   ├── test_scrape.py
│   ├── test_style.py
│   ├── test_tools.py
│   ├── test_training.py
│   ├── test_video.py
│   └── test_vit.py
├── toolkit/
│   ├── __init__.py
│   └── env_health.py
├── training/
│   ├── __init__.py
│   ├── cli.py
│   ├── enhanced_trainer.py
│   └── loop.py
├── utils/
│   ├── agentic/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── interview.py
│   │   └── quality_loop.py
│   ├── eval/
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── editing.py
│   │   ├── guidance.py
│   │   ├── inventions.py
│   │   ├── planners.py
│   │   ├── regional.py
│   │   ├── sample_argv.py
│   │   ├── sample_cli_parser.py
│   │   ├── sample_core.py
│   │   ├── sample_prompt.py
│   │   └── sample_run.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── checkpoints.py
│   │   ├── encoders.py
│   │   ├── hf_loaders.py
│   │   └── model_paths.py
│   ├── prompt/
│   │   ├── __init__.py
│   │   ├── art_direction.py
│   │   ├── content_controls.py
│   │   ├── game.py
│   │   ├── rag.py
│   │   ├── realism.py
│   │   ├── scene.py
│   │   ├── special_prompt_helpers.py
│   │   ├── stack.py
│   │   ├── style.py
│   │   └── text.py
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── architecture_consistency.py
│   │   ├── critics.py
│   │   ├── finishing.py
│   │   ├── pixel_art.py
│   │   ├── realism.py
│   │   └── test_time_pick.py
│   ├── safety/
│   │   ├── __init__.py
│   │   └── minor_guard.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── monitoring.py
│   │   ├── objectives.py
│   │   ├── perf.py
│   │   └── preference.py
│   ├── __init__.py
│   ├── captioning.py
│   ├── consistency.py
│   ├── native.py
│   ├── runtime.py
│   └── visual_design.py
├── vit_quality/
│   ├── __init__.py
│   ├── infer.py
│   ├── model.py
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
- [docs/FILES.md](docs/FILES.md) — full file map

