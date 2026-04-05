# Pipeline: books, comics, manga, storyboards

**Scope:** Multi-page workflows, **text in panels**, speech bubbles, optional **face/edge anchoring** across pages, **OCR**-guided fixes.  
Uses the **same** `sample.py` / `train.py` as general image gen; this folder owns **book-specific orchestration**.

## Script (canonical path)

| Script | Description |
|--------|-------------|
| [scripts/generate_book.py](scripts/generate_book.py) | Multi-page generator: inpaint chains, MDM masks, OCR repair |
| [scripts/train_book_model.py](scripts/train_book_model.py) | Book-specific trainer wrapper around `train.py` with presets + native manifest preflight |
| [scripts/prepare_and_train_book.py](scripts/prepare_and_train_book.py) | One command: optional HF export -> optional caption normalization -> book trainer |
| [book_helpers.py](book_helpers.py) | Presets (`--book-accuracy`), pick-best wiring, post-process (sharpen / naturalize) |
| [book_training_helpers.py](book_training_helpers.py) | Training presets, train command builder, Rust/Zig manifest diagnostics |
| [consistency_helpers.py](consistency_helpers.py) | Character / prop / vehicle / setting / lettering prompt cues + optional JSON spec |
| [prompt_lexicon.py](prompt_lexicon.py) | Style snippets (shonen/webtoon/…), merged negatives, aspect presets, tategaki/SFX hints |
| [../../docs/BOOK_COMIC_TECH.md](../../docs/BOOK_COMIC_TECH.md) | Tech survey + SDX mapping + **best output checklist** (data, training, production tier) |
| [../../scripts/tools/book_scene_split.py](../../scripts/tools/book_scene_split.py) | Split `## Page N` script → one line per page for `--prompts-file` |

**Legacy entry (wrapper):** `scripts/book/generate_book.py` forwards to the path above so old commands keep working.

### Accuracy & consistency (uses shared repo tools)

- **`utils/quality/test_time_pick.py`** — via `sample.py --pick-best clip|edge|ocr|combo` and `--num` candidates per page.
- **`data/caption_utils.prepend_quality_if_short`** — when `--book-accuracy balanced|maximum|production` or `--prepend-quality-if-short`.
- **`config/defaults/ai_image_shortcomings.py`** — wired through `generate_book.py` to sample.py (`--shortcomings-mitigation`, `--shortcomings-2d`) for photoreal, digital art, CG, and optional 2D-anime mitigation packs.
- **`config/defaults/art_mediums.py`** — wired through `generate_book.py` to sample.py (`--art-guidance-mode`, `--anatomy-guidance`) for artist-medium + anatomy/proportion guidance.
- **`config/defaults/style_guidance.py`** — wired through `generate_book.py` to sample.py (`--style-guidance-mode`) for anime/comic/editorial/concept/game/photo style-language guidance.
- **`utils/quality/quality.py`** — optional post-pass: sharpen + naturalize (film grain / micro-contrast) after each PNG.
- **`sample.py`** — `--boost-quality`, `--subject-first`, `--save-prompt`, `--cfg-scale` / `--cfg-rescale`, `--vae-tiling`, `--text-in-image`.

Recommended starting point:

```bash
python pipelines/book_comic/scripts/generate_book.py --ckpt path/to/best.pt --output-dir out_book \
  --book-type manga --prompts-file pages.txt --book-accuracy balanced --text-in-image
```

Use `--book-accuracy maximum` for 4 candidates, or **`production`** for 6 candidates + stricter merged negatives (`prompt_lexicon.PRODUCTION_TIER_NEGATIVE_ADDON`). See **[../../docs/BOOK_COMIC_TECH.md](../../docs/BOOK_COMIC_TECH.md)** (*Best output checklist*) for data, training, and inference quality notes.

### Extra CLI (continuity, layout, manifest)

| Flag | Effect |
|------|--------|
| `--narration-prefix "…"` | Prepended to every page prompt (cover uses narration only). |
| `--panel-layout NAME` | Adds a layout hint (`single`, `two_panel_horizontal`, `four_koma`, `splash`, …) via `prompt_lexicon`. |
| `--page-context-previous N` | Injects a short rolling summary of the last *N* page lines (total length capped by `--page-context-max-chars`). |
| `--chapter-break-every K` | Every *K* pages, drops the inpaint chain so the next page starts fresh (story context in prompts can still roll). |
| `--start-page INDEX` | 0-based first page to generate; earlier prompts still feed rolling context; existing PNGs can seed the chain. |
| `--skip-existing` | Skip `sample.py` when `page_NNN.png` already exists; still updates chain/manifest when applicable. |
| `--write-book-manifest` | Writes `book_manifest.json` in `--output-dir` (paths, composed prompts, seeds, `skipped`). |
| `--include-print-finish` / `--include-cover-spotlight` | Forwarded into book prefix enhancement when supported. |
| `--artist-craft-profile` / `--shot-language` / `--pacing-plan` | Artist-facing craft bundles for panel flow, shot grammar, and pacing rhythm. |
| `--lettering-craft` / `--value-plan` / `--screentone-plan` | Add practical lettering placement, value hierarchy, and screentone discipline hints. |
| `--artist-pack` | One-flag preset for artist craft controls (`manga_cinematic`, `comic_dialogue`, `webtoon_scroll`, `storyboard_fast`). |
| `--oc-name` / `--oc-archetype` / `--oc-traits` / `--oc-wardrobe` | Build an original-character consistency block from artist-facing anchors. |
| `--oc-pack` | One-flag OC design preset (`heroine_scifi`, `rival_dark`, `mentor_classic`), overridable by explicit `--oc-*` fields. |
| `--book-style-pack` | Higher-level bundle that sets artist pack + OC pack + safety/nsfw defaults (`manga_nsfw_action`, `webtoon_nsfw_romance`, `comic_dialogue_safe`, `oc_launch_safe`). |
| `--humanize-pack` / `--humanize-profile` / `--humanize-imperfection` | Humanization helpers to push handcrafted variance and reduce synthetic/sterile artifacts. |
| `--auto-humanize` | Autopilot for humanization defaults inferred from `--book-type`, `--lexicon-style`, and effective safety mode; explicit `--humanize-*` flags still win. |
| `--safety-mode` / `--nsfw-pack` / `--nsfw-civitai-pack` / `--civitai-trigger-bank` | Forward sample.py content-control scaffolding (including NSFW modes) into page and OCR-repair passes. |
| `--sample-originality` / `--sample-creativity` | Passed through to `sample.py` when set. |
| `--shortcomings-mitigation` / `--shortcomings-2d` | Forward mitigation packs to `sample.py` for digital/realism consistency (OCR repair passes included). |
| `--art-guidance-mode` / `--anatomy-guidance` | Forward artist-medium and anatomy/proportion packs to `sample.py` (OCR repair passes included). |
| `--style-guidance-mode` | Forward style-domain and artist/game-name stabilization packs to `sample.py` (OCR repair passes included). |
| `--resize-mode` / `--resize-saliency-face-bias` | Aspect fit when width/height differ: stretch, center-crop, or saliency crop (helps composition/cropping). |
| `--expected-count` / `--expected-count-target` / `--expected-count-object` | Optional count-verifier controls for `--pick-best combo_count` (people or simple repeated objects). |

### Entity & lettering consistency (prompt layer)

| Flag | Effect |
|------|--------|
| `--consistency-json PATH` | JSON object: `character` (string or trait map), `costume`, `props` / `objects`, `vehicle`, `setting`, `creature`, `palette_lighting`, `lettering`, `negative_level`, … |
| `--consistency-character` / `--consistency-costume` | Freeform protagonist + locked outfit (merged into every page after narration). |
| `--consistency-props` | Semicolon-separated important props (each gets a “same object” cue). |
| `--consistency-vehicle` / `--consistency-setting` | Recurring vehicle and environment continuity. |
| `--consistency-creature` | Recurring pet / mascot / non-human companion. |
| `--consistency-palette` / `--consistency-lighting` | Color script + light direction (reduces visual drift). |
| `--consistency-visual-extra` | Extra tokens appended to the consistency block. |
| `--consistency-lettering-hard` | Strong legibility positives (pair with `--ocr-fix` and expected text for dialogue). |
| `--consistency-negative {none,light,strong}` | Appends anti-drift negatives; omit to use JSON `negative_level` if present. |

Combine with **`--anchor-face`**, **`--edge-anchor`**, **`--anchor-speech-bubbles`**, and **`--character-sheet`** / **`--character-prompt-extra`** for stronger real consistency than prompts alone.

Helpers: `book_helpers.compose_book_page_prompt` (narration → **consistency** → panel → rolling → page line), `build_rolling_page_context`; layout: `prompt_lexicon.PANEL_LAYOUT_HINTS`.

## Artist craft references

The artist-facing prompt helpers are informed by common production guidance around panel flow, pacing, shot language, and lettering readability:
- [Scott McCloud — Making Comics](https://www.scottmccloud.com/makingcomics/)
- [Scott McCloud — Understanding Comics](https://scottmccloud.com/2-print/1-uc/index.html)
- [Todd Klein — Balloon placement tips](https://kleinletters.com/BalloonPlacement.html)
- [StudioBinder — Shot size language](https://www.studiobinder.com/blog/types-of-camera-shots-sizes-in-film/)

## Shared engine (repo root)

Training still uses root `train.py` with your **book/comic/manga** dataset (line art, panels, rendered text in captions).  
You can use `pipelines/book_comic/scripts/train_book_model.py` to apply book-oriented training presets and optional native preflight (`Rust sdx-jsonl-tools` + `Zig sdx-linecrc` when built) before launching `train.py`.
For fully automated runs, `pipelines/book_comic/scripts/prepare_and_train_book.py` can export from Hugging Face, normalize captions with book guidance packs, then launch the trainer in one flow.
When using caption normalization in that flow, `--train-humanize-pack` can apply anti-synthetic training presets (`lite`, `balanced`, `strong`).
Book trainers now expose AR controls directly: `--ar-profile auto|none|layout|strong|zorder`, plus explicit `--num-ar-blocks` / `--ar-block-order` overrides (see [../../docs/AR.md](../../docs/AR.md)).
Use JSONL fields such as `parts` / `region_captions` when you need layout-aware text—see [docs/REGION_CAPTIONS.md](../../docs/REGION_CAPTIONS.md).

## Related utilities

| Area | Location |
|------|----------|
| Text rendering / OCR hooks | `utils/generation/text_rendering.py` |
| Prompt + typography hints | `config/prompt_domains.py`, [docs/PROMPT_COOKBOOK.md](../../docs/PROMPT_COOKBOOK.md) |

## Checkpoint layout

Prefer a dedicated results dir, e.g. `results/book_comic_run/`, separate from general `image_gen` experiments.
