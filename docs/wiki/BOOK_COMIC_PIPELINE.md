# Book/Comic Pipeline

This pipeline wraps SDX for multi-page visual storytelling workflows.

## Entry scripts

- Generation: `pipelines/book_comic/scripts/generate_book.py`
- Training wrapper: `pipelines/book_comic/scripts/train_book_model.py`
- End-to-end prepare+train: `pipelines/book_comic/scripts/prepare_and_train_book.py`

Compatibility wrappers also exist under `scripts/book/`.

## What this pipeline adds

- Style and medium lexicon packs
- OC (original character) helpers and identity consistency hints
- Artist craft controls (camera language, lettering, values, pacing)
- Humanization helpers to reduce synthetic look
- Auto-humanize profile inference based on book/style/safety mode
- AR controls for layout consistency in training wrappers

## Core helper modules

- `pipelines/book_comic/prompt_lexicon.py`
- `pipelines/book_comic/book_helpers.py`
- `pipelines/book_comic/book_training_helpers.py`
- `pipelines/book_comic/consistency_helpers.py`

## Typical flow

1. Build prompt/control presets for the project style.
2. Generate pages with `generate_book.py` and pick-best settings.
3. Curate outputs and normalize captions.
4. Train or finetune with `train_book_model.py`.
5. Re-run generation and benchmark to verify gains.

## Recommended docs

- `pipelines/book_comic/README.md`
- `docs/BOOK_COMIC_TECH.md`
