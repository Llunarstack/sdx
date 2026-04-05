# Pipelines (two product lines)

SDX trains **one shared diffusion core** (`diffusion/`, `models/`, `data/`, `utils/`, `train.py`, `sample.py`).  
We split **workflows and documentation** by **target use case**, not by duplicating the engine.

| Pipeline | Purpose | Typical training focus | Entry docs |
|----------|---------|------------------------|------------|
| **[image_gen/](image_gen/README.md)** | General text-to-image | Photos, art, broad domains; HF export, smoke runs | [README](image_gen/README.md) |
| **[book_comic/](book_comic/README.md)** | Books, comics, manga, storyboards | Multi-page coherence, text-in-panel, line art, OCR loops | [README](book_comic/README.md) |

**Rule of thumb:** both pipelines use the **same** `train.py` / checkpoints; you may train **two separate checkpoints** (e.g. `results/general/` vs `results/book_comic/`) with different datasets and caption styles. The split here is **where to look** for scripts and conventions—not a second copy of DiT code.

See also: [docs/REGION_CAPTIONS.md](../docs/REGION_CAPTIONS.md) (layout-friendly JSONL), [docs/LANDSCAPE_2026.md](../docs/LANDSCAPE_2026.md) (industry context).
