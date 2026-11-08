# Pipeline: general image generation

**Scope:** Single-image (or batched) text-to-image—photography, illustration, game assets, etc.  
**Not** multi-page book workflows; those live under **[book_comic/](../book_comic/README.md)**.

## Shared engine (repo root)

| Piece | Role |
|-------|------|
| `train.py` | DiT + diffusion training |
| `sample.py` | Inference, CFG, img2img, pick-best |
| `config/`, `diffusion/`, `models/`, `data/`, `utils/` | Shared library code |

## Training paths

| Goal | Doc / script |
|------|----------------|
| Minimal sanity check | [docs/GETTING_STARTED.md](../../docs/GETTING_STARTED.md#smoke-training-prove-the-pipeline-works), `scripts/tools/manifest.py smoke-dataset` |
| Hugging Face → JSONL | [docs/GUIDES.md](../../docs/GUIDES.md#training-on-danbooru-style-data-from-hugging-face), `scripts/tools/manifest.py hf-export` / `hf-train` |
| Hardware / VRAM | [docs/GETTING_STARTED.md](../../docs/GETTING_STARTED.md#pc-specs-for-training-sdx-consumer-hardware-no-datacenter-gpus) |

## Conventions

- Put checkpoints under something like `results/<your_run>/` (e.g. `results/general_baseline/`).
- Use JSONL + `data_path` as in the main [README](../../README.md#data-format).
- For **regional / layout** captions without switching pipelines, see [docs/PROMPT_STACK.md](../../docs/PROMPT_STACK.md#regional--layout-captions-jsonl).
