# SDX Wiki

Practical map of the repository: what each subsystem does, how data/weights flow, and which commands to use.

## Start here

- Core architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Pretrained and model wiring: [MODELS_AND_PRETRAINED.md](MODELS_AND_PRETRAINED.md)
- Training flow: [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)
- Inference and prompting: [INFERENCE_AND_PROMPTING.md](INFERENCE_AND_PROMPTING.md)
- Book/comic pipeline: [BOOK_COMIC_PIPELINE.md](BOOK_COMIC_PIPELINE.md)
- Tools and automation loops: [TOOLS_AND_AUTOMATION.md](TOOLS_AND_AUTOMATION.md)
- Native acceleration stack: [NATIVE_STACK.md](NATIVE_STACK.md)

## Repo in one screen

- `train.py`: main trainer entry point.
- `sample.py`: main image generation entry point.
- `data/`: dataset loading, caption normalization, batching.
- `models/`: DiT blocks, attention, adapters, prompt adherence modules.
- `diffusion/`: schedules, losses, samplers, holy grail logic.
- `utils/`: quality scoring, prompt controls, checkpoint/model helpers, native bridges.
- `pipelines/book_comic/`: high-level generation and training wrappers for sequential art.
- `scripts/tools/`: benchmarking, preference mining, auto-improve loop, ops checks.
- `native/`: Rust/C/C++/CUDA primitives for speed-critical quality and validation tasks.

## Typical lifecycle

1. Curate data and normalize captions.
2. Train or finetune with `train.py` (or `pipelines/book_comic/scripts/train_book_model.py`).
3. Generate candidates with `sample.py`.
4. Score/rank outputs with `utils.quality.test_time_pick`.
5. Benchmark checkpoints with `scripts/tools/benchmark_suite.py`.
6. Mine preferences and run DPO loops with `scripts/tools/ops/auto_improve_loop.py`.

## Fast command cheatsheet

```bash
# Environment and stack sanity check (no training)
python -m scripts.tools startup_readiness --out-md startup_readiness.md

# Check pretrained model wiring/resolution
python -m scripts.tools pretrained_status --out-json pretrained_status.json

# Generate one image
python sample.py --prompt "cinematic portrait, natural skin texture" --num 4 --pick-best auto

# Benchmark one or more checkpoints
python scripts/tools/benchmark_suite.py --suite-pack top_contender_proxy_v1 --compare-to-dir results/

# Run one full improvement loop
python -m scripts.tools auto_improve_loop --iterations 1 --suite-pack top_contender_proxy_v1
```

## Related docs

- Main docs index: [../README.md](../README.md)
- Tooling index: [../../scripts/tools/README.md](../../scripts/tools/README.md)
- Model weaknesses: [../MODEL_WEAKNESSES.md](../MODEL_WEAKNESSES.md)
- v3 release notes: [../releases/v3.md](../releases/v3.md)
