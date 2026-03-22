"""
Atlas pipeline tags + human-readable summaries for the Codebase Explorer.

Maps each indexed path to:
- ``atlas_tags``: how the file relates to **image_gen**, **book_comic**, **training**, **sampling**, **ViT**, etc.
- ``atlas_summary``: short paragraph tying the file to prompt→image and training (complements docstrings).
"""

from __future__ import annotations

# Tag vocabulary (also used by the website filter)
TAG_IMAGE_GEN = "image_gen"
TAG_BOOK_COMIC = "book_comic"
TAG_SHARED_CORE = "shared_core"
TAG_TRAINING = "training"
TAG_SAMPLING = "sampling"
TAG_VIT = "vit_scorer"
TAG_DATA = "dataset"
TAG_TEXT = "text_encode"
TAG_DIFFUSION = "diffusion"
TAG_DIT = "dit"
TAG_UTILS = "utilities"
TAG_OTHER = "other"

# Exact-path blurbs (highest priority)
_EXACT: dict[str, str] = {
    "train.py": (
        "Trains the shared DiT + diffusion stack on your dataset. Checkpoints power **both** general "
        "text-to-image (`sample.py`, `pipelines/image_gen`) and book/comic generation "
        "(`pipelines/book_comic/scripts/generate_book.py`) — same weights, different data and prompts."
    ),
    "sample.py": (
        "End-to-end **single-image** generation: prompt → text encoder(s) → denoising loop in latent space → "
        "VAE decode. This is the usual **image_gen** path; book workflows call the same primitives inside "
        "multi-page scripts."
    ),
    "inference.py": (
        "Programmatic API for loading a checkpoint and running the **sampling** pipeline (same core as "
        "`sample.py`) for integrations and tools."
    ),
    "config/train_config.py": (
        "Central `TrainConfig` and `get_dit_build_kwargs`: every training hyperparameter and DiT variant "
        "used when learning weights for **both** product lines."
    ),
    "data/t2i_dataset.py": (
        "Dataset and collation: images + captions (JSONL or folders) fed into **training**; caption format "
        "also drives what the model sees for general vs book-style data."
    ),
    "diffusion/gaussian_diffusion.py": (
        "Noise schedule, forward diffusion, training losses, and schedulers — the **mathematical core** "
        "shared by all training and sampling."
    ),
    "models/dit_text.py": (
        "T5-conditioned DiT: patch embed, cross-attention to captions, self-attention — the **generator** "
        "trained by `train.py` and run during **sampling** for both pipelines."
    ),
    "models/attention.py": (
        "Attention implementations (xformers / SDPA) used by DiT blocks in **training** and **inference**."
    ),
    "utils/text_encoder_bundle.py": (
        "Loads T5 and optional CLIP fusion for **prompt → conditioning**; frozen or partially trained "
        "during DiT training, then reused at sample time."
    ),
    "utils/checkpoint_loading.py": (
        "Loads DiT checkpoints for resume and **sampling**; keeps config and optional fusion weights aligned."
    ),
    "utils/test_time_pick.py": (
        "CLIP / edge / OCR **best-of-N** scoring after generation — improves perceived quality for "
        "**sampling** (including book presets that call pick-best)."
    ),
    "pipelines/book_comic/scripts/generate_book.py": (
        "**Book/comic image-gen orchestration**: multi-page prompts, optional OCR and panel logic, calling "
        "into the same checkpoint and `sample.py`-style decoding as general image gen."
    ),
    "pipelines/book_comic/book_helpers.py": (
        "Book workflow helpers: CFG / pick-best / post-process wiring for **book_comic** runs."
    ),
    "pipelines/book_comic/prompt_lexicon.py": (
        "Comic/manga style snippets and merged negatives for **book_comic** prompts — augments the same "
        "text→DiT path as general generation."
    ),
    "ViT/train.py": (
        "Trains the **ViT scorer** (quality + adherence) on manifests — **not** the DiT. Used to filter or "
        "rank data and outputs; improves datasets and best-of-N for **both** product lines indirectly."
    ),
    "ViT/infer.py": (
        "Scores images in a JSONL manifest with a trained ViT — **downstream QA**, not prompt→latent generation."
    ),
}

# Prefix templates: (prefix, must_be_dir or file), blurb
_PREFIX_BLURBS: list[tuple[str, str]] = [
    (
        "pipelines/image_gen/",
        "Docs and conventions for **general text-to-image**; training still uses root `train.py` and shared `config/`, `data/`, `models/`.",
    ),
    (
        "pipelines/book_comic/",
        "**Book/comic/manga** layer: scripts and helpers on top of the same DiT checkpoint; focuses on multi-page and layout-oriented generation.",
    ),
    (
        "scripts/training/",
        "Training-adjacent scripts (HF export, latent precompute, etc.) that feed **`train.py`** and dataset prep.",
    ),
    (
        "scripts/download/",
        "Downloads base weights (T5, VAE, CLIP, …) into `model/` so **training** and **sampling** can run.",
    ),
    (
        "tests/",
        "Unit and smoke tests — guard **training** and **sampling** behavior across refactors.",
    ),
    (
        "native/",
        "Optional native helpers (JSONL, speed) for data tooling around **training** pipelines.",
    ),
]


def _tags_for_path(rel: str) -> set[str]:
    t: set[str] = set()

    if rel == "utils/ar_dit_vit.py":
        t.update({TAG_UTILS, TAG_VIT, TAG_TRAINING, TAG_SAMPLING, TAG_SHARED_CORE})
        return t

    if rel.startswith("ViT/"):
        t.add(TAG_VIT)
        if "train.py" in rel or rel.endswith("losses.py") or "dataset" in rel:
            t.add(TAG_TRAINING)
        return t

    if rel.startswith("pipelines/book_comic") or rel.startswith("scripts/book/"):
        t.add(TAG_BOOK_COMIC)
        if "generate_book" in rel or "book_helpers" in rel or rel.endswith("generate_book.py"):
            t.add(TAG_SAMPLING)
        return t

    if rel.startswith("pipelines/image_gen"):
        t.add(TAG_IMAGE_GEN)
        return t

    if rel in ("train.py",) or rel.startswith("training/") or rel.startswith("scripts/training/"):
        t.add(TAG_TRAINING)

    if rel in ("sample.py", "inference.py") or rel.startswith("scripts/enhanced/sample"):
        t.add(TAG_SAMPLING)

    if rel.startswith("data/"):
        t.add(TAG_DATA)
        t.add(TAG_TRAINING)

    if rel.startswith("config/"):
        t.add(TAG_TRAINING)
        t.add(TAG_SHARED_CORE)

    if rel.startswith("diffusion/"):
        t.add(TAG_DIFFUSION)
        t.add(TAG_TRAINING)
        t.add(TAG_SAMPLING)
        t.add(TAG_SHARED_CORE)

    if rel.startswith("models/"):
        t.add(TAG_DIT)
        t.add(TAG_TRAINING)
        t.add(TAG_SAMPLING)
        t.add(TAG_SHARED_CORE)

    if rel.startswith("utils/text_encoder") or "text_encoder" in rel:
        t.add(TAG_TEXT)
        t.add(TAG_TRAINING)
        t.add(TAG_SAMPLING)
        t.add(TAG_SHARED_CORE)

    if rel.startswith("utils/"):
        t.add(TAG_UTILS)
        if any(
            x in rel
            for x in (
                "checkpoint",
                "model_paths",
                "metrics",
                "config_validator",
                "error_handling",
            )
        ):
            t.add(TAG_TRAINING)
        if any(x in rel for x in ("checkpoint_loading", "test_time_pick", "quality")):
            t.add(TAG_SAMPLING)
        if "checkpoint" in rel or "config_validator" in rel:
            t.add(TAG_SHARED_CORE)

    if rel.startswith("scripts/tools/") or rel.startswith("scripts/cli"):
        t.add(TAG_UTILS)

    if rel.startswith("pipelines/") and not t:
        t.add(TAG_SHARED_CORE)

    if TAG_SHARED_CORE not in t and not t.isdisjoint({TAG_DIT, TAG_DIFFUSION, TAG_DATA, TAG_TEXT}):
        t.add(TAG_SHARED_CORE)

    if rel in ("train.py", "sample.py", "inference.py"):
        t.add(TAG_SHARED_CORE)

    if not t:
        t.add(TAG_OTHER)

    return t


def _blurb_from_tags(rel: str, tags: set[str], doc_summary: str) -> str:
    """Fallback when no exact/prefix blurb: stitch tag semantics + first line."""
    bits: list[str] = []
    if TAG_BOOK_COMIC in tags:
        bits.append("Part of the **book/comic** workflow layer (multi-page orchestration on top of the shared DiT).")
    elif TAG_IMAGE_GEN in tags:
        bits.append("Relates to **general image generation** docs or presets (same engine as book paths).")
    if TAG_TRAINING in tags and TAG_SAMPLING in tags:
        bits.append("Used in **training** and when **generating** images from a trained checkpoint.")
    elif TAG_TRAINING in tags:
        bits.append("Supports **training** the diffusion model or preparing training data.")
    elif TAG_SAMPLING in tags:
        bits.append("Used on the **sampling** side (prompt → image) after a checkpoint exists.")
    if TAG_VIT in tags:
        bits.append("**ViT module**: scores (image, caption) pairs — improves datasets and ranking, not latent generation.")

    if not bits:
        bits.append("Supporting code in the SDX tree; see imports and docstring for precise use.")

    extra = " ".join(bits)
    base = doc_summary.strip() if doc_summary else "Source file in the SDX codebase."
    if len(base) > 200:
        base = base[:197] + "…"
    return f"{base} {extra}"


def enrich_atlas_entry(rel: str, doc_summary: str, _role: str) -> dict[str, object]:
    tags = _tags_for_path(rel)
    tags_list = sorted(tags)

    if rel in _EXACT:
        atlas_summary = _EXACT[rel]
    else:
        atlas_summary = None
        for prefix, blurb in _PREFIX_BLURBS:
            if rel.startswith(prefix):
                atlas_summary = blurb
                break
        if atlas_summary is None:
            atlas_summary = _blurb_from_tags(rel, tags, doc_summary)

    return {
        "atlas_tags": tags_list,
        "atlas_summary": atlas_summary.strip(),
    }
