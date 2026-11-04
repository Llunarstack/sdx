# File map: SDX project and external reference repos

> SDX section **generated** from the real tree (module docstring / first comment per file). Regenerate after moving files; the external-repos section is hand-written.

All **SDX project files** (what we run and edit) and **key files in external repos** (reference only; clone with `scripts/setup/clone_repos.ps1` or `scripts/setup/clone_repos.sh`). SDX does **not** import from `external/` at runtime.

---

## SDX project files (repo root: `sdx/`)

Run commands from repo root so `config`, `data`, `diffusion`, `models`, `utils` are on the path.

**Quick orientation:** [CODEBASE.md](CODEBASE.md) — layers, repo tree, entry points, `scripts/` layout, contribution rules.

### How the tree fits together (runtime)

| Area | Role | Consumed by |
|:-----|:-----|:------------|
| **config/** | `TrainConfig`, `get_dit_build_kwargs`, presets | `train.py`, `sample.py`, checkpoints |
| **data/** | `Text2ImageDataset`, captions | `train.py` |
| **diffusion/** | `GaussianDiffusion`, schedules, loss weights | `train.py`, `sample.py` |
| **models/** | DiT, ControlNet, MoE, RAE bridge, optional cascaded / multimodal **scaffolds** | `train.py`, `sample.py` |
| **utils/** | Checkpoint load, text-encoder bundle, REPA helpers, QC, metrics | `train.py`, `sample.py`, scripts |
| **vit_quality/** | ViT-style quality / adherence scoring and prompt tools (**not** the DiT generator) | CLI + optional dataset QA |
| **scripts/** | Download, tools, cascade stub | Ops & CI |
| **pipelines/** | **image_gen** vs **book_comic**: per–product-line docs; book script at `pipelines/book_comic/scripts/generate_book.py` | See [pipelines/README.md](../pipelines/README.md) |
| **native/** | Fast JSONL / manifest helpers (Rust, Go, Node, …) | Optional; see `native/README.md`, ecosystem map [MODEL_STACK.md § native libs](MODEL_STACK.md#native--lower-level-libraries--quality-training-prompt-adherence) |
| **toolkit/** | QoL: environment report | `python -m toolkit.env_health` |
| **datasets/** | **Your** datasets: `datasets/train/` + sidecar captions (see [datasets/README.md](../datasets/README.md)) | `train.py --data-path datasets/train` |
| **model/** | Downloaded HF weights (gitignored) | Resolved via `utils/modeling/model_paths.py` |

End-to-end flow: **manifest/images → train.py (T5/triple + VAE/RAE + DiT + diffusion) → checkpoint → sample.py → image**. See [README § Architecture and pipeline](../README.md#architecture-and-pipeline).

### Root

| File | Description |
|------|-------------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | PR checklist. |
| [demo.py](../demo.py) | SDX one-command demo — download weights and generate a sample image. |
| [DEPRECATIONS.md](../DEPRECATIONS.md) | Removed paths and migrations. |
| [inference.py](../inference.py) | Load a checkpoint and optionally smoke-test one latent refinement step (``--verify``, random latents). |
| [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | Generated ASCII tree (`python -m scripts.tools update_project_structure`). |
| [pyproject.toml](../pyproject.toml) | Package metadata, ruff + pytest config. |
| [README.md](../README.md) | Project overview, setup, data format, training, options. |
| [requirements.txt](../requirements.txt) | Pip dependencies. |
| [sample.py](../sample.py) | Generate an image from a text prompt with a trained checkpoint (``python sample.py --help`` for every option). |
| [SECURITY.md](../SECURITY.md) | Security policy. |
| [train.py](../train.py) | Train the text-conditioned DiT (VP diffusion, rectified flow or masked MDM), on one GPU or with DDP. |

### `config/` — Training config, presets, prompt/style registries

| File | Description |
|------|-------------|
| [config/__init__.py](../config/__init__.py) | Public config API: training dataclass, DiT build kwargs, shared prompt defaults. |
| [config/presets.py](../config/presets.py) | Sampler presets / OP modes, the prompt-routed quality policy, and the agentic + quality-flywheel stack defaults. |
| [config/prompts.py](../config/prompts.py) | Prompt packs and keyword helpers: tag merging, shared negatives/domains, physics, creature and failure-mode packs (stdlib only, plus the minor guard). |
| [config/styles.py](../config/styles.py) | Style/artist tags, artist-first medium guidance and style-domain guidance packs for sampling and training captions. |
| [config/train_config.py](../config/train_config.py) | Training configuration — single source of truth for all hyperparameters. |

### `data/` — Datasets and caption utilities

| File | Description |
|------|-------------|
| [data/__init__.py](../data/__init__.py) | Dataset loaders for text-to-image training. |
| [data/caption_utils.py](../data/caption_utils.py) | Training-caption helpers: tag emphasis, person-descriptor ordering, tag front-loading, anti-blending, |
| [data/manifest_utils.py](../data/manifest_utils.py) | Shared manifest helpers for scrape -> enrich -> train and the JSONL CLI tools. |
| [data/t2i_dataset.py](../data/t2i_dataset.py) | Text-to-image training dataset (image folders or JSONL manifests), its collate function, the aspect-bucket |

### `diffusion/` — Diffusion / flow objectives, schedules, solvers, sampling

| File | Description |
|------|-------------|
| [diffusion/__init__.py](../diffusion/__init__.py) | sdx.diffusion — VP and rectified-flow diffusion: noise schedules, ``GaussianDiffusion`` (training losses and the |
| [diffusion/flow_matching.py](../diffusion/flow_matching.py) | Linear **rectified-flow** training objective (Liu et al. 2022; SD3 recipe, Esser et al. 2024). |
| [diffusion/gaussian_diffusion.py](../diffusion/gaussian_diffusion.py) | VP Gaussian diffusion for the DiT: training losses (offset noise, min-SNR, eps/v/x0 prediction) and the |
| [diffusion/sampling.py](../diffusion/sampling.py) | The "holy grail" sampling recipe (``--holy-grail*``): coordinated per-step schedules instead of one fixed CFG and |
| [diffusion/schedules.py](../diffusion/schedules.py) | VP beta schedules for ``GaussianDiffusion`` (betas clipped to ``[1e-4, 0.999]`` for stability). |
| [diffusion/solvers.py](../diffusion/solvers.py) | Inference solvers and step grids for SDX. |

### `models/` — DiT variants, attention, conditioning, LoRA, video DiT

| File | Description |
|------|-------------|
| [models/__init__.py](../models/__init__.py) | DiT-family architectures. ``DiT_models_text`` maps every ``--model`` name to its constructor. |
| [models/attention.py](../models/attention.py) | Attention for the DiTs. |
| [models/conditioning.py](../models/conditioning.py) | Extra conditioning inputs for the DiTs. |
| [models/dit.py](../models/dit.py) | Class-conditional Diffusion Transformer (Peebles & Xie, "Scalable Diffusion Models with Transformers") and the |
| [models/dit_text.py](../models/dit_text.py) | DiT with text conditioning — the text-to-image denoiser (PixArt-style). |
| [models/dit_variants.py](../models/dit_variants.py) | Alternative DiT trunks. |
| [models/lora.py](../models/lora.py) | LoRA / DoRA / LyCORIS adapter loading for DiT-style Linear layers. |
| [models/video_dit.py](../models/video_dit.py) | Video denoiser for ``pipelines/video``: a factorized spatiotemporal DiT (STDiT, as in CogVideoX / Open-Sora). |

### `training/` — Training CLI and loop

| File | Description |
|------|-------------|
| [training/cli.py](../training/cli.py) | ``train.py`` command line: the argparse parser, its book/comic preset and the mapping to ``TrainConfig``. |
| [training/enhanced_trainer.py](../training/enhanced_trainer.py) | Enhanced training system for advanced DiT models. |
| [training/loop.py](../training/loop.py) | Helpers of the ``train.py`` loop: losses (VP / flow / MDM / REPA / bridge), conditioning kwargs, EMA, schedules, |

### `utils/` — Shared library: prompt, generation, quality, modeling, training, agentic

| File | Description |
|------|-------------|
| [utils/agentic/agent.py](../utils/agentic/agent.py) | Agentic image generation: plan -> tool -> verify -> reflect -> evolve (GenEvolve trajectories, VisionCreator UTPC planning, Act-Reflect-Think-Act)… |
| [utils/agentic/interview.py](../utils/agentic/interview.py) | Pre-generation interview (questions, reference requests, pose/place defaults, one-at-a-time flow) and post-generation critique into a targeted fix… |
| [utils/agentic/quality_loop.py](../utils/agentic/quality_loop.py) | Quality loops around sample.py: multi-candidate ensembles and composite ranking, TF-IDF prompt enrichment, self-correction and quality gates, and t… |
| [utils/captioning.py](../utils/captioning.py) | Image understanding for captions and references: WD tagger, OCR / VLM captions / control maps, SauceNAO + TinEye reverse search with booru tags, we… |
| [utils/consistency.py](../utils/consistency.py) | Character and style consistency: character profiles, face/body encoders and validation, the character database, consistency training losses, prompt… |
| [utils/generation/editing.py](../utils/generation/editing.py) | Image editing: heuristic and segmentation inpaint masks, image dissection, part compositing, text rendering, edit-skill plans and the post-generati… |
| [utils/generation/guidance.py](../utils/generation/guidance.py) | Sampling-time guidance: the composable CFG stack, batched CFG forward, CFG-rejection probe, latent filters, frontier kwargs and CLIP alignment. |
| [utils/generation/inventions.py](../utils/generation/inventions.py) | Invention stack: prompt passes for known T2I failure modes (counting, binding, negation, anatomy, plastic look), multi-character cast, corpus retri… |
| [utils/generation/planners.py](../utils/generation/planners.py) | Stills planners that build ``sample.py`` runs: user taste, perfect-gen, the agentic stills orchestrator and the creative co-pilot. |
| [utils/generation/regional.py](../utils/generation/regional.py) | Ideogram-style box + prompt regional generation: layout parsing, per-box sketches, masks and the regional CFG forward. |
| [utils/generation/sample_argv.py](../utils/generation/sample_argv.py) | ``sample.py`` as a subprocess: argv from argparse values, the OCR-repair passthrough, img2img / inpaint runs, ComfyUI export. |
| [utils/generation/sample_cli_parser.py](../utils/generation/sample_cli_parser.py) | Sample CLI parser builder (extracted from sample.py). |
| [utils/generation/sample_core.py](../utils/generation/sample_core.py) | ``sample.py`` models and conditioning: spec parsing, checkpoint / encoder loading, Krea controls, the quality policy, text embeddings and init late… |
| [utils/generation/sample_prompt.py](../utils/generation/sample_prompt.py) | ``sample.py`` prompt phases: feature planners (frontier, Krea, attention layout, auto-refine), the user's prompt, additions and the final encoder t… |
| [utils/generation/sample_run.py](../utils/generation/sample_run.py) | ``sample.py`` run: flag resolution, the denoising passes and latent refinement, decode / post / pick-best / save, and programmatic text-to-image. |
| [utils/modeling/architecture.py](../utils/modeling/architecture.py) | DiT architecture helpers: variant profiling, block-AR conditioning for the ViT scorer, training-free block caches and the 2026 architecture theme map. |
| [utils/modeling/checkpoints.py](../utils/modeling/checkpoints.py) | DiT checkpoint loading, checkpoint diffs / LoRA extraction, and the LoRA bank registry. |
| [utils/modeling/encoders.py](../utils/modeling/encoders.py) | Text-encoder bundles (T5 / triple / penta), their readiness checks and caption routing, and the training/sampling autoencoder. |
| [utils/modeling/hf_loaders.py](../utils/modeling/hf_loaders.py) | Lazy Hugging Face model loaders (captioners, OCR, depth, detectors, reward scorers), the reward panel and ControlNet control-map helpers. |
| [utils/modeling/model_paths.py](../utils/modeling/model_paths.py) | Where models live: local pretrained/ folders vs Hugging Face hub ids, the HF scaffold registry and config-only downloads, and the HF token lookup. |
| [utils/native.py](../utils/native.py) | Lazy bridge to ``sdx_native`` (``native/_experimental/python``). |
| [utils/prompt/art_direction.py](../utils/prompt/art_direction.py) | Art direction: prompt -> shot, lens, lighting and palette decisions, composition brief, photo-realism packs and originality tokens. |
| [utils/prompt/content_controls.py](../utils/prompt/content_controls.py) | Content-control tag packs for harder generation targets, with the built-in tag tables. |
| [utils/prompt/game.py](../utils/prompt/game.py) | Game imagery: which asset a prompt asks for, the conventions that asset needs, and its finishing defaults. |
| [utils/prompt/rag.py](../utils/prompt/rag.py) | Prompt grounding: RAG facts, the creative RAG engine and the advanced prompt analyzer / optimizer. |
| [utils/prompt/realism.py](../utils/prompt/realism.py) | Photorealism shot spec (camera, shutter, ISO chosen to agree with the planned lens and light), hard-prompt |
| [utils/prompt/scene.py](../utils/prompt/scene.py) | Scene and subject prompting: creature / physics add-ons, busy scenes, multi-instance and multi-subject hints, scene blueprints and shape scaffolds,… |
| [utils/prompt/special_prompt_helpers.py](../utils/prompt/special_prompt_helpers.py) | Special Prompt Helpers for AI Image Generation. |
| [utils/prompt/stack.py](../utils/prompt/stack.py) | Unified prompt stack: context types, intent clauses, prompt intelligence, content controls, the stages, the runner and the ``sample.py`` bridge. |
| [utils/prompt/style.py](../utils/prompt/style.py) | Style genomes: native fast paths, the genome type, chaos presets, the inventor, prompt mutation and the explore planner. |
| [utils/prompt/text.py](../utils/prompt/text.py) | Prompt text: comma tags, ``(emphasis)`` weights, the layered prompt layout, breakdown, i18n hints, JSONL lint, ``@artist`` mentions and the block c… |
| [utils/quality/architecture_consistency.py](../utils/quality/architecture_consistency.py) | Architecture / perspective **consistency lock**. |
| [utils/quality/critics.py](../utils/quality/critics.py) | Image critics: AI-look tells, art direction, human-likeness, scene physics, hands, counts, contact shadows, seed variation, and training-manifest c… |
| [utils/quality/finishing.py](../utils/quality/finishing.py) | Image finishing: deterministic sRGB helpers (contrast, saturation, grades, filters), the lens / sensor / film capture finish, artistic and human-ma… |
| [utils/quality/pixel_art.py](../utils/quality/pixel_art.py) | Exact-pixel finishing for game imagery: grid snap, palette, transparent background, seamless tiles, critics. |
| [utils/quality/realism.py](../utils/quality/realism.py) | Photoreal critics and finishing: plastic-skin and waxy-highlight detectors (pooled into |
| [utils/quality/test_time_pick.py](../utils/quality/test_time_pick.py) | Best-of-N candidate scoring for ``sample.py --pick-best`` (and every other ranker in the repo). |
| [utils/runtime.py](../utils/runtime.py) | Runtime helpers: UTF-8 console stdio on Windows, fast JSON (orjson when installed), plain-dict snapshots and an optional cProfile wrapper. |
| [utils/safety/minor_guard.py](../utils/safety/minor_guard.py) | Unconditional minor-sexualization guard for every text path into an SDX model. |
| [utils/training/monitoring.py](../utils/training/monitoring.py) | Training run checks and monitoring: config validation and memory estimates, the sdx logger, metrics tracking, progress bars, the live terminal dash… |
| [utils/training/objectives.py](../utils/training/objectives.py) | Per-step training schedules and objectives: block-AR and timestep curricula, OT noise pairing, part-aware captions and grounding losses. |
| [utils/training/perf.py](../utils/training/perf.py) | Training throughput: CUDA/CPU backend tuning, fast data loading, latent caching and text-encode batching. |
| [utils/training/preference.py](../utils/training/preference.py) | Preference learning: Diffusion-DPO data and losses, Flow-GRPO / BranchGRPO / DenseGRPO / TP-GRPO, LADD distillation and the user feedback bus. |
| [utils/visual_design.py](../utils/visual_design.py) | Visual-design prompt packs: UI, architecture, STEM, textbook, brand, infographic, packaging, wayfinding, |

### `vit_quality/` — ViT quality / adherence scorer (not the generator)

| File | Description |
|------|-------------|
| [vit_quality/__init__.py](../vit_quality/__init__.py) | Canonical ViT quality/adherence package. |
| [vit_quality/infer.py](../vit_quality/infer.py) | Score a JSONL manifest with a ViT quality/adherence checkpoint. |
| [vit_quality/model.py](../vit_quality/model.py) | ViT quality/adherence model, losses, text featurizers and checkpoint loading (ViTScorer is the one inference path). |
| [vit_quality/train.py](../vit_quality/train.py) | Train the ViT quality + adherence model on an SDX manifest (``python -m vit_quality.train``). |

### `frontier/` — Research modules: layout, multiview / 3D, imagination

| File | Description |
|------|-------------|
| [frontier/__init__.py](../frontier/__init__.py) | Frontier: generation ideas outside the usual prompt-latent-decode box (concept map in ``frontier/README.md``). |
| [frontier/engine.py](../frontier/engine.py) | Frontier engine: compose logic, narrative, chaos, and memory analyzers into one plan, and wire frontier plans |
| [frontier/imagination.py](../frontier/imagination.py) | Imagination engine (``--frontier-creative``): creative analyzers that emit prompt fragments AND diffusion knobs |
| [frontier/layout.py](../frontier/layout.py) | Layout control: Omost canvas, coordinate prompts, LAMIC schedules and metrics, Dense Diffusion-style |
| [frontier/multiview.py](../frontier/multiview.py) | SDX-Spatial: the 3D core. A tri-plane latent field that decodes to one consistent object from any camera. |
| [frontier/planning.py](../frontier/planning.py) | Prompt-side frontier planners. |
| [frontier/registry.py](../frontier/registry.py) | Catalog of research ideas that exist in SDX: source paper/project, module, and status. |

### `pipelines/` — Product lines: book/comic, image_gen, video

| File | Description |
|------|-------------|
| [pipelines/book_comic/prompt_lexicon.py](../pipelines/book_comic/prompt_lexicon.py) | Prompt snippets, negative add-ons, and aspect hints for book/comic/manga workflows. |
| [pipelines/book_comic/render.py](../pipelines/book_comic/render.py) | Book page rendering: sample.py flag assembly, page prompts, dual-model preflight, on-disk project |
| [pipelines/book_comic/scripts/generate_book.py](../pipelines/book_comic/scripts/generate_book.py) | Book / comic / manga workflow generator. |
| [pipelines/book_comic/scripts/prepare_and_train_book.py](../pipelines/book_comic/scripts/prepare_and_train_book.py) | End-to-end book dataset prep + training launcher. |
| [pipelines/book_comic/scripts/train_book_model.py](../pipelines/book_comic/scripts/train_book_model.py) | Book/comic/manga training launcher around root ``train.py``. |
| [pipelines/book_comic/story.py](../pipelines/book_comic/story.py) | Book prompt planning: text continuity, challenging content packs, style fusion/authenticity, |
| [pipelines/book_comic/training.py](../pipelines/book_comic/training.py) | Helpers for book/comic/manga model training orchestration. |
| [pipelines/video/controls.py](../pipelines/video/controls.py) | Conditioning controls: entity control modes, rigs, reference sheets, elements, multimodal refs, |
| [pipelines/video/frontier_compiler.py](../pipelines/video/frontier_compiler.py) | Frontier compiler — novel SDX video subsystems merged into shot enrichment. |
| [pipelines/video/motion.py](../pipelines/video/motion.py) | Motion analysis and temporal synthesis: optical flow, motion brushes/templates, stabilization, |
| [pipelines/video/neural_sample.py](../pipelines/video/neural_sample.py) | Neural VideoDiT sample path — denoise short latent clips without retrieve/interpolate. |
| [pipelines/video/pipeline.py](../pipelines/video/pipeline.py) | Retrieve -> transform -> compose video pipeline: clip retrieval, plan building, segment orchestration. |
| [pipelines/video/quality.py](../pipelines/video/quality.py) | Segment quality gates and post repairs: permanence, artifacts, identity, glyphs, drift, contact, |
| [pipelines/video/scene.py](../pipelines/video/scene.py) | Plan types, prompt shot planner, scene graph compiler and continuity validators. |
| [pipelines/video/scripts/generate_video.py](../pipelines/video/scripts/generate_video.py) | Generate video via retrieve → keyframe edit → interpolate → stitch pipeline. |
| [pipelines/video/segment_processor.py](../pipelines/video/segment_processor.py) | Process one segment: extract → keyframe edit → motion transfer → interpolate → temporal → quality. |
| [pipelines/video/studio.py](../pipelines/video/studio.py) | Studio routing: motion grammar per medium, engine presets, director storyboards, style router. |
| [pipelines/video/video_io.py](../pipelines/video/video_io.py) | Video I/O helpers: probe, extract frames, encode, retime. |

### `scripts/` — CLI tools (`python -m scripts.tools <name>`), scraping, pipeline runner

| File | Description |
|------|-------------|
| [scripts/run_pipeline.py](../scripts/run_pipeline.py) | Run the full SDX image-gen pipeline in order (setup → smoke → download → train). |
| [scripts/scrape.py](../scripts/scrape.py) | Booru dataset scraper (danbooru, e621, rule34.xxx, rule34.xyz) -> SDX JSONL manifests. |
| [scripts/setup/clone_repos.ps1](../scripts/setup/clone_repos.ps1) | Meta DiT (reference implementation cited in README) |
| [scripts/setup/clone_repos.sh](../scripts/setup/clone_repos.sh) | Clone reference repos used by or cited in SDX (optional; runtime deps are pip-only). |
| [scripts/tools/__init__.py](../scripts/tools/__init__.py) | Tools subpackage for SDX helper utilities. |
| [scripts/tools/__main__.py](../scripts/tools/__main__.py) | Run any tool script under ``scripts/tools/`` by its file name:: |
| [scripts/tools/_repo_bootstrap.py](../scripts/tools/_repo_bootstrap.py) | Ensure repo root and ``native/_experimental/python`` are on ``sys.path`` for tool scripts. |
| [scripts/tools/agentic.py](../scripts/tools/agentic.py) | Agentic image generation and planning: generate, roles, evolve, flywheel, stills, perfect, visual-brain. |
| [scripts/tools/bench.py](../scripts/tools/bench.py) | Checkpoint benchmarks and sample.py sweeps: suite, history, prompts, seeds, gallery. |
| [scripts/tools/book.py](../scripts/tools/book.py) | Book/comic run helpers: manifest check, prompt audit, scene split, visual-memory patch. |
| [scripts/tools/captions.py](../scripts/tools/captions.py) | Caption tools for JSONL training manifests: normalize, hygiene, lint, and coverage reports. |
| [scripts/tools/ckpt.py](../scripts/tools/ckpt.py) | Checkpoint / model inspection and export commands. |
| [scripts/tools/danbooru.py](../scripts/tools/danbooru.py) | Danbooru tag vocabulary: fetch -> split general tags into buckets -> merge one categorized TSV. |
| [scripts/tools/dev/architecture_themes.py](../scripts/tools/dev/architecture_themes.py) | Dump ``utils.modeling.architecture`` themes as JSON or Markdown. |
| [scripts/tools/dev/cursorfix.sh](../scripts/tools/dev/cursorfix.sh) | Rewrites commits authored/committed by Cursor Agent to your identity. |
| [scripts/tools/dev/generate_sdx_architecture_diagram.py](../scripts/tools/dev/generate_sdx_architecture_diagram.py) |  |
| [scripts/tools/dev/integration_smoke.py](../scripts/tools/dev/integration_smoke.py) | End-to-end integration smoke test before RunPod spend. |
| [scripts/tools/dev/quick_test.py](../scripts/tools/dev/quick_test.py) | Smoke test: one forward pass (no training). Verifies imports, config, and model run. |
| [scripts/tools/dev/smoke_imports.py](../scripts/tools/dev/smoke_imports.py) | Smoke test: import all *internal* SDX modules to catch broken links early. |
| [scripts/tools/dev/strip_ai_contributors.py](../scripts/tools/dev/strip_ai_contributors.py) | Strip AI Co-authored-by trailers from all commits in the current repo. |
| [scripts/tools/dev/strip_ai_git_trailers.py](../scripts/tools/dev/strip_ai_git_trailers.py) | Git --msg-filter helper: drop AI attribution lines from commit messages. |
| [scripts/tools/dev/test_style_native_stack.py](../scripts/tools/dev/test_style_native_stack.py) | Smoke-test Style Genome + native stack (Rust / CUDA / Go). |
| [scripts/tools/dev/validate_config_json.py](../scripts/tools/dev/validate_config_json.py) | Validate a JSON-serialized TrainConfig (or subset) against ``validate_train_config``. |
| [scripts/tools/enhanced.py](../scripts/tools/enhanced.py) | EnhancedDiT commands (optional path, not the default ``train.py`` / ``sample.py`` DiT-Text + T5 pipeline). |
| [scripts/tools/evals.py](../scripts/tools/evals.py) | Competitor-weakness eval harness: fixed prompt suite, graceful scorers, and the runner CLI. |
| [scripts/tools/generate.py](../scripts/tools/generate.py) | Generation wrappers around ``sample.py`` and side pipelines. |
| [scripts/tools/manifest.py](../scripts/tools/manifest.py) | JSONL training-manifest tools: filter, enrich, gate, merge, inspect, export from Hugging Face. |
| [scripts/tools/native/build_native.ps1](../scripts/tools/native/build_native.ps1) | .\scripts\tools\native\build_native.ps1 |
| [scripts/tools/native/build_native.sh](../scripts/tools/native/build_native.sh) | Build SDX native C++ (optional CUDA) and Rust tools when cargo exists. |
| [scripts/tools/native/clean_native_builds.ps1](../scripts/tools/native/clean_native_builds.ps1) | .\scripts\tools\native\clean_native_builds.ps1 |
| [scripts/tools/prompts.py](../scripts/tools/prompts.py) | Prompt tooling: preview, compose, diff, research, style invention, taste and scene planning. |
| [scripts/tools/repo/clean_repo_artifacts.py](../scripts/tools/repo/clean_repo_artifacts.py) | Delete local generated artifacts from the repository tree. |
| [scripts/tools/repo/update_project_structure.py](../scripts/tools/repo/update_project_structure.py) | Write PROJECT_STRUCTURE.md at the repo root: an ASCII tree of the SDX codebase. |
| [scripts/tools/repo/verify_doc_links.py](../scripts/tools/repo/verify_doc_links.py) | Verify relative markdown links in key docs point to existing files. |
| [scripts/tools/status.py](../scripts/tools/status.py) | Environment / model status reports: pretrained, readiness, inventory, gen-searcher. |
| [scripts/tools/superior.py](../scripts/tools/superior.py) | Self-improvement stack: Superior generation, preference mining, DPO loops, RSI and human feedback. |
| [scripts/tools/trainers.py](../scripts/tools/trainers.py) | Fine-tuning / distillation trainers and training-prep commands. |

### `setup/` — Download, manifest and index builders

| File | Description |
|------|-------------|
| [setup/download.py](../setup/download.py) | Download everything SDX needs: pretrained models, booru datasets, HF dataset packs. |
| [setup/indexes.py](../setup/indexes.py) | Build lookup indexes from scraped/trained data: artist aliases, RAG corpus, LoRA bank. |
| [setup/manifests.py](../setup/manifests.py) | Manifest prep for training: merge, sanitize, clean media, WD-tag, enrich captions, control maps. |

### `toolkit/` — Environment health

| File | Description |
|------|-------------|
| [toolkit/env_health.py](../toolkit/env_health.py) | Print environment summary for SDX training / inference debugging. |

### `native/` — Optional native accelerators (Rust / C++ / Zig / Go) and the `sdx_native` Python bridge

| File | Description |
|------|-------------|
| [native/_experimental/python/sdx_native/ffi.py](../native/_experimental/python/sdx_native/ffi.py) | ctypes bindings for the optional native libraries: ``sdx_cpu`` / ``sdx_cuda`` (``native/cpp``) and ``sdx_rs`` |
| [native/_experimental/python/sdx_native/native_tools.py](../native/_experimental/python/sdx_native/native_tools.py) | Optional ``native/`` helpers: discover built CLIs and shared libraries, run the Rust JSONL / noise-schedule |
| [native/_experimental/python/sdx_native/text_hygiene.py](../native/_experimental/python/sdx_native/text_hygiene.py) | Unicode and structure hygiene for training/inference captions and JSONL manifests (stdlib + optional xxhash): |

### `examples/` — Example scripts (JSON example inputs sit beside them)

| File | Description |
|------|-------------|
| [examples/example_character_consistency.py](../examples/example_character_consistency.py) | Character consistency example: build character profiles, score an image against one, and wire the |
| [examples/run_baseline_eval.py](../examples/run_baseline_eval.py) | Run `sample.py` over `examples/eval_prompts_baseline.json` (or another pack). |

### `runpod/` — RunPod deployment scripts

| File | Description |
|------|-------------|
| [runpod/bootstrap.sh](../runpod/bootstrap.sh) | Run from anywhere on the pod. Handles clone, update, then starts the full pipeline. |
| [runpod/datasets.sh](../runpod/datasets.sh) | Download training images from Hugging Face (danbooru, rule34, e621, gelbooru). |
| [runpod/download.sh](../runpod/download.sh) | Download pretrained models + Hugging Face training datasets + prep (WD tags, enrich). |
| [runpod/lib.sh](../runpod/lib.sh) | Shared RunPod helpers. Source from runpod/*.sh — defines functions only, runs nothing. |
| [runpod/run.ps1](../runpod/run.ps1) |  |
| [runpod/run.sh](../runpod/run.sh) | Full pipeline: setup → download → test → train (see scripts/run_pipeline.py --help). |
| [runpod/sample.sh](../runpod/sample.sh) | Generate an image from a trained checkpoint (RAG, @artist, box layout, prompt stack). |
| [runpod/sdx.sh](../runpod/sdx.sh) | SDX on RunPod — one script: |
| [runpod/setup.ps1](../runpod/setup.ps1) |  |
| [runpod/setup.sh](../runpod/setup.sh) | Bootstrap a RunPod (or any Ubuntu CUDA) pod for SDX training. |
| [runpod/start.sh](../runpod/start.sh) |  |
| [runpod/status.sh](../runpod/status.sh) | Live SDX pipeline monitor: HF export, tagging, enrich, training, LoRA bank. |
| [runpod/test.ps1](../runpod/test.ps1) |  |
| [runpod/test.sh](../runpod/test.sh) | Integration smoke test — scrape all sites + pipeline checks before spending on GPU time. |
| [runpod/train.sh](../runpod/train.sh) | Train image model — full DiT feature stack (no video/frontier). |
| [runpod/train_lora_bank.sh](../runpod/train_lora_bank.sh) | Train modular LoRA bank: per-artist + per-style adapters on a frozen base checkpoint. |
| [runpod/ultimate.sh](../runpod/ultimate.sh) | Ultimate SDX pipeline: pretrained → HF datasets → WD tags → enrich → train base → LoRA bank. |
| [runpod/update.sh](../runpod/update.sh) | Force-sync repo on a pod (discards local edits) and fix CRLF on shell scripts. |
| [runpod/verify_env.sh](../runpod/verify_env.sh) | Post-install sanity checks for RunPod (run by setup.sh and scripts/run_pipeline.py). |

### `tests/` — pytest suite (one file per package / concern)

| File | Description |
|------|-------------|
| [tests/conftest.py](../tests/conftest.py) |  |
| [tests/test_agentic.py](../tests/test_agentic.py) | Tests for utils.agentic: agentic stack, perfect-gen loop, visual brain, superior stack. |
| [tests/test_book.py](../tests/test_book.py) | Tests for book challenging-content prompt fragments. |
| [tests/test_data.py](../tests/test_data.py) | Tests for data/: T2I dataset, caption utils, manifests, reverse search. |
| [tests/test_diffusion.py](../tests/test_diffusion.py) | Unit tests for diffusion preference DPO surrogate loss. |
| [tests/test_eval.py](../tests/test_eval.py) | Also holds what used to be: test_eval_prompt_pack, test_eval_report, test_eval_harness, test_run_baseline_eval, test_benchmark_suite_tool, test_aut… |
| [tests/test_frontier.py](../tests/test_frontier.py) | Tests for frontier outside-the-box generation modules. |
| [tests/test_image.py](../tests/test_image.py) | Image layer of the minor guard (zero-shot CLIP age check) and the WD-tag manifest screen. |
| [tests/test_internal_imports.py](../tests/test_internal_imports.py) | Every ``from <internal module> import name`` (at any nesting level) names something that exists. |
| [tests/test_minor_guard.py](../tests/test_minor_guard.py) | Unconditional minor-sexualization guard (utils.safety.minor_guard) and its call sites. |
| [tests/test_model.py](../tests/test_model.py) | Tests for model construction and forward pass correctness. |
| [tests/test_prompt.py](../tests/test_prompt.py) | PromptParser extracts clothing, objects, people, and spatial relation tokens. |
| [tests/test_quality.py](../tests/test_quality.py) | Tests for utils.quality and utils.visual_design: scorers, realism, art/game-art, consistency locks. |
| [tests/test_runtime.py](../tests/test_runtime.py) | Tests for utils.runtime and utils.native fast paths. |
| [tests/test_sample.py](../tests/test_sample.py) | Tests for OCR repair argv forwarding. |
| [tests/test_sampling.py](../tests/test_sampling.py) | Tests for ``diffusion.sampling`` (holy-grail recipe) and the latent helpers in ``diffusion.sampling_utils``. |
| [tests/test_scrape.py](../tests/test_scrape.py) | Tests for scripts.scrape: safety filters, frame split, media validation, site adapters. |
| [tests/test_style.py](../tests/test_style.py) | Tests for config.styles extraction helpers. |
| [tests/test_tools.py](../tests/test_tools.py) | Tests for scripts.tools CLIs and the core CLI entry points. |
| [tests/test_training.py](../tests/test_training.py) | Training data conditioning: aspect-ratio buckets, bucket batches, caption tag front-loading, caption dropout. |
| [tests/test_video.py](../tests/test_video.py) | Tests for prompt/image understanding on T2V / I2V / V2V paths. |
| [tests/test_vit.py](../tests/test_vit.py) | Tests for vit_quality: scorer, text encoder, naming compat. |

## External repos (reference only)

Cloned into `external/` by [scripts/setup/clone_repos.ps1](../scripts/setup/clone_repos.ps1) or [scripts/setup/clone_repos.sh](../scripts/setup/clone_repos.sh). **Do not import from these at runtime.**

### 1. DiT (Meta) — base transformer

- **Clone:** `https://github.com/facebookresearch/DiT` → `external/DiT`
- **What we use:** Patch embed, timestep embed, adaLN blocks; architecture reference for our [models/dit.py](../models/dit.py) and [models/dit_text.py](../models/dit_text.py).

| Path under `external/DiT/` | Description |
|----------------------------|-------------|
| [models.py](../external/DiT/models.py) | DiT block, patchify, adaLN. |
| [diffusion/gaussian_diffusion.py](../external/DiT/diffusion/gaussian_diffusion.py) | Diffusion utils. |
| [diffusion/schedules.py](../external/DiT/diffusion/schedules.py) | Timestep respacing. |
| [train.py](../external/DiT/train.py) | Reference training loop. |
| [sample.py](../external/DiT/sample.py) | Reference sampling. |

---

### 2. ControlNet — structural conditioning

- **Clone:** `https://github.com/lllyasviel/ControlNet` → `external/ControlNet`
- **What we use:** Control image (depth/edge/pose) conditioning; reference for our [models/conditioning.py](../models/conditioning.py) (ControlNet) and control scale.

| Path under `external/ControlNet/` | Description |
|-----------------------------------|-------------|
| [cldm/model.py](../external/ControlNet/cldm/model.py) | ControlNet + SD backbone. |
| [ldm/models/diffusion/ddim.py](../external/ControlNet/ldm/models/diffusion/ddim.py) | DDIM sampling. |
| [ldm/models/diffusion/sampling_util.py](../external/ControlNet/ldm/models/diffusion/sampling_util.py) | Dynamic threshold (norm_thresholding). |
| [docs/train.md](../external/ControlNet/docs/train.md) | Training ControlNet. |

---

### 3. FLUX (Black Forest Labs) — modern diffusion

- **Clone:** `https://github.com/black-forest-labs/flux` → `external/flux`
- **What we use:** Sampling, guidance, img2img/fill; structural conditioning; resolution/aspect ideas.

| Path under `external/flux/` | Description |
|------------------------------|-------------|
| [src/flux/model.py](../external/flux/src/flux/model.py) | Transformer, guidance embed. |
| [src/flux/sampling.py](../external/flux/src/flux/sampling.py) | Denoise loop, timesteps, guidance. |
| [src/flux/modules/conditioner.py](../external/flux/src/flux/modules/conditioner.py) | CLIP/T5 conditioning. |
| [src/flux/modules/autoencoder.py](../external/flux/src/flux/modules/autoencoder.py) | VAE. |
| [docs/structural-conditioning.md](../external/flux/docs/structural-conditioning.md) | Control/docs. |
| [docs/fill.md](../external/flux/docs/fill.md) | Inpainting/fill. |

---

### 4. Stability generative-models — SD3 / MM-DiT

- **Clone:** `https://github.com/Stability-AI/generative-models` → `external/generative-models`
- **What we use:** SD3 / MM-DiT architecture and training reference.

| Path under `external/generative-models/` | Description |
|------------------------------------------|-------------|
| (repo structure varies) | Look for diffusion, DiT, or SD3 modules; official Stability reference. |

---

### 5. PixArt-alpha — PixArt-α (T5 + DiT)

- **Clone:** `https://github.com/PixArt-alpha/PixArt-alpha` → `external/PixArt-alpha`
- **What we use:** T5 text encoder + DiT; efficient T2I training reference.

---

### 6. PixArt-sigma — PixArt-Σ (4K T2I)

- **Clone:** `https://github.com/PixArt-alpha/PixArt-sigma` → `external/PixArt-sigma`
- **What we use:** 4K generation, weak-to-strong training, token compression attention.

---

### 7. Z-Image — S3-DiT (single-stream)

- **Clone:** `https://github.com/Tongyi-MAI/Z-Image` → `external/Z-Image`
- **What we use:** Single-stream DiT (text + image in one sequence); efficient scaling reference.

---

### 8. SiT — Scalable Interpolant Transformers

- **Clone:** `https://github.com/willisma/SiT` → `external/SiT`
- **What we use:** Flow matching + DiT backbone; interpolant framework; sampling/convergence ideas.

---

### 9. Lumina-T2X — Lumina-T2I / Next-DiT

- **Clone:** `https://github.com/Alpha-VLLM/Lumina-T2X` → `external/Lumina-T2X`
- **What we use:** Next-DiT scaling; multi-resolution; Rectified Flow; T2I pipeline reference.
