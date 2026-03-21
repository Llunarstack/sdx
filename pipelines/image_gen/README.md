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
| Minimal sanity check | [docs/SMOKE_TRAINING.md](../../docs/SMOKE_TRAINING.md), `scripts/tools/make_smoke_dataset.py` |
| Hugging Face → JSONL | [docs/DANBOORU_HF.md](../../docs/DANBOORU_HF.md), `scripts/training/hf_export_to_sdx_manifest.py`, `scripts/training/hf_download_and_train.py` |
| Hardware / VRAM | [docs/HARDWARE.md](../../docs/HARDWARE.md) |

## Conventions

- Put checkpoints under something like `results/<your_run>/` (e.g. `results/general_baseline/`).
- Use JSONL + `data_path` as in the main [README](../../README.md#data-format).
- For **regional / layout** captions without switching pipelines, see [docs/REGION_CAPTIONS.md](../../docs/REGION_CAPTIONS.md).
