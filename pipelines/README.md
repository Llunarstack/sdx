# Pipelines

SDX trains **one shared diffusion core** (`diffusion/`, `models/`, `data/`, `utils/`, `train.py`, `sample.py`).  
We split **workflows and documentation** by **target use case**, not by duplicating the engine.

| Pipeline | Purpose | Typical training focus | Entry docs |
|----------|---------|------------------------|------------|
| **[image_gen/](image_gen/README.md)** | General text-to-image | Photos, art, broad domains; HF export, smoke runs | [README](image_gen/README.md) |
| **[book_comic/](book_comic/README.md)** | Books, comics, manga, storyboards | Multi-page coherence, text-in-panel, line art, OCR loops | [README](book_comic/README.md) |
| **[video/](video/)** | Text/image → video (retrieve, edit, stitch) | Reference clips, keyframe SDX edits, temporal harmonize | `python -m scripts.tools video_generate` |

**Rule of thumb:** image and book/comic workflows share `train.py` and the image model code, but can use separate checkpoints and datasets. The video pipeline can use image outputs and also has an optional VideoDiT path. Spatial/3D remains experimental under `frontier/multiview.py`. These modules share utilities rather than duplicating the core.

See [model lines and release gates](../docs/MODEL_LINES_AND_RELEASES.md) for branch and release plans.

See also: [docs/PROMPT_STACK.md](../docs/PROMPT_STACK.md#regional--layout-captions-jsonl) (layout-friendly JSONL), [docs/RESEARCH.md](../docs/RESEARCH.md#image-generation-landscape-architecture-and-workflows-2026) (industry context).
