# Training Pipeline

How SDX training works from manifest to checkpoint.

## Main entry points

- Core trainer: `train.py`
- CLI mapping: `training/train_cli_parser.py`, `training/train_args.py`
- Dataset loader: `data/t2i_dataset.py`
- Caption prep: `data/caption_utils.py` and `scripts/tools/normalize_captions.py`

## Flow

1. Parse training flags into typed config.
2. Load data from JSONL/manifest and normalize captions.
3. Build model and diffusion schedule.
4. Run optimization loop with checkpointing.
5. Emit checkpoints and logs for later benchmark/eval.

## Book/comic training wrappers

For sequential-art workflows, use:

- `pipelines/book_comic/scripts/train_book_model.py`
- `pipelines/book_comic/scripts/prepare_and_train_book.py`

These add:

- presets (`fast`, `balanced`, `production`)
- AR profile controls (`--ar-profile`, `--num-ar-blocks`, `--ar-block-order`)
- optional preflight and caption normalization integration

## Quality-improvement loop (post-training)

1. Benchmark candidate checkpoints with `scripts/tools/benchmark_suite.py`.
2. Mine preference pairs with `scripts/tools/training/mine_preference_pairs.py`.
3. Train DPO with `scripts/tools/ops/auto_improve_loop.py`.
4. Re-benchmark and optionally promote best checkpoint.

## Before long runs

- Run readiness report:
  - `python -m scripts.tools startup_readiness --out-md startup_readiness.md`
- Validate pretrained routing:
  - `python -m scripts.tools pretrained_status`
