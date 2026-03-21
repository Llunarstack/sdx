# Pipeline: books, comics, manga, storyboards

**Scope:** Multi-page workflows, **text in panels**, speech bubbles, optional **face/edge anchoring** across pages, **OCR**-guided fixes.  
Uses the **same** `sample.py` / `train.py` as general image gen; this folder owns **book-specific orchestration**.

## Script (canonical path)

| Script | Description |
|--------|-------------|
| [scripts/generate_book.py](scripts/generate_book.py) | Multi-page generator: inpaint chains, MDM masks, OCR repair |
| [book_helpers.py](book_helpers.py) | Presets (`--book-accuracy`), pick-best wiring, post-process (sharpen / naturalize) |
| [prompt_lexicon.py](prompt_lexicon.py) | Style snippets (shonen/webtoon/…), merged negatives, aspect presets, tategaki/SFX hints |
| [../../docs/BOOK_COMIC_TECH.md](../../docs/BOOK_COMIC_TECH.md) | Tech survey + SDX mapping (consistency, layout, OCR) |
| [../../scripts/tools/book_scene_split.py](../../scripts/tools/book_scene_split.py) | Split `## Page N` script → one line per page for `--prompts-file` |

**Legacy entry (wrapper):** `scripts/book/generate_book.py` forwards to the path above so old commands keep working.

### Accuracy & consistency (uses shared repo tools)

- **`utils/test_time_pick.py`** — via `sample.py --pick-best clip|edge|ocr|combo` and `--num` candidates per page.
- **`data/caption_utils.prepend_quality_if_short`** — when `--book-accuracy balanced|maximum|production` or `--prepend-quality-if-short`.
- **`utils/quality.py`** — optional post-pass: sharpen + naturalize (film grain / micro-contrast) after each PNG.
- **`sample.py`** — `--boost-quality`, `--subject-first`, `--save-prompt`, `--cfg-scale` / `--cfg-rescale`, `--vae-tiling`, `--text-in-image`.

Recommended starting point:

```bash
python pipelines/book_comic/scripts/generate_book.py --ckpt path/to/best.pt --output-dir out_book \
  --book-type manga --prompts-file pages.txt --book-accuracy balanced --text-in-image
```

Use `--book-accuracy maximum` for 4 candidates, or **`production`** for 6 candidates + stricter merged negatives (`prompt_lexicon.PRODUCTION_TIER_NEGATIVE_ADDON`). See **[../../docs/BOOK_MODEL_EXCELLENCE.md](../../docs/BOOK_MODEL_EXCELLENCE.md)** for a full quality checklist.

## Shared engine (repo root)

Training still uses root `train.py` with your **book/comic/manga** dataset (line art, panels, rendered text in captions).  
Use JSONL fields such as `parts` / `region_captions` when you need layout-aware text—see [docs/REGION_CAPTIONS.md](../../docs/REGION_CAPTIONS.md).

## Related utilities

| Area | Location |
|------|----------|
| Text rendering / OCR hooks | `utils/text_rendering.py` |
| Prompt + typography hints | `config/prompt_domains.py`, [docs/PROMPT_COOKBOOK.md](../../docs/PROMPT_COOKBOOK.md) |

## Checkpoint layout

Prefer a dedicated results dir, e.g. `results/book_comic_run/`, separate from general `image_gen` experiments.
