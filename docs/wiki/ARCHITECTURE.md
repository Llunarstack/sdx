# Architecture

This page explains how SDX components connect at runtime.

## Core layers

- `config/`: declarative defaults and train/inference argument surfaces.
- `data/`: manifest parsing, caption processing, and sample batching.
- `models/`: DiT backbone and auxiliary conditioning modules.
- `diffusion/`: schedulers, noise/timestep logic, objective math, and adaptive sampling controls.
- `training/`: CLI parser and argument mapping from command line to runtime config.
- `utils/`: cross-cutting helpers (prompt controls, model path routing, quality picking, native wrappers).

## Generation path

1. `sample.py` parses flags and composes final prompt/control state.
2. Text prompt is transformed through prompt-control utilities in `utils/prompt/`.
3. Model and optional adapters are loaded using path resolvers in `utils/modeling/model_paths.py`.
4. Diffusion sampler runs denoising in `diffusion/` with optional holy-grail/adaptive controls.
5. Candidate images are scored by `utils/quality/test_time_pick.py` for prompt adherence and quality dimensions.
6. Best candidate(s) are saved and optional metadata is emitted.

## Training path

1. `train.py` builds config from `training/train_cli_parser.py` and `training/train_args.py`.
2. Dataset pipeline (`data/t2i_dataset.py`) loads image/caption data and applies normalization.
3. Forward pass through DiT model in `models/`.
4. Loss and schedule logic from `diffusion/` drives optimization.
5. Checkpoints and logs are written to output directories.

## High-level pipelines

- `pipelines/book_comic/` wraps core train/sample into production-oriented scripts for sequential art:
  - `scripts/generate_book.py`
  - `scripts/train_book_model.py`
  - `scripts/prepare_and_train_book.py`

These wrappers expose profile/preset controls for quality, style, consistency, and AR behavior while still using core SDX internals.

## Most important integration points

- Model resolution: `utils/modeling/model_paths.py`
- Prompt assembly and grounding: `utils/prompt/`
- Quality scoring and picking: `utils/quality/test_time_pick.py`
- Continuous improvement loop: `scripts/tools/ops/auto_improve_loop.py`
