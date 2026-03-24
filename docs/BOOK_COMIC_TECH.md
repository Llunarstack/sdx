# Book / comic / manga: techniques and how SDX maps to them

This note ties **common 2024–2026 product/research themes** in sequential art generation to **this repo**—without claiming parity with any commercial tool.

---

## 1. Multi-page character consistency

**Industry pattern:** reusable **character references** (turnaround, palette, outfit locks) and **cross-page conditioning** so faces/outfits don’t drift.

**SDX today**

- JSONL **`character`** / **`character_id`** patterns and **`--character-sheet`** on `sample.py` (see README sampling section).
- **`generate_book.py`**: `--anchor-face`, `--edge-anchor`, `--anchor-speech-bubbles`, optional **`--character-sheet`**, **`--character-prompt-extra`**.
- **Training:** same identity tags in captions + optional **`--style-embed-dim`** / style fields.

**Next steps (see [IMPROVEMENTS.md](IMPROVEMENTS.md))**

- Stronger **reference conditioning** (extra image tokens, IP-Adapter-style—roadmap).
- **LoRA** per character for sampling (`sample.py --lora`).

---

## 2. Layout, panels, and “where things go”

**Research / products:** **GLIGEN-style** grounding (boxes + phrases), **ControlNet** (edges, depth, scribbles), **layout tokens** in JSONL.

**SDX today**

- **Regional text:** JSONL **`parts`** / **`region_captions`** → merged captions for T5 ([REGION_CAPTIONS.md](REGION_CAPTIONS.md)).
- **Control:** `sample.py` **`--control-image`** + training **`control_cond`** when you train with control pairs.
- **Panel-ish structure:** prompt lexicon + aspect presets ([`pipelines/book_comic/prompt_lexicon.py`](../pipelines/book_comic/prompt_lexicon.py)).

**Next steps**

- Explicit **box + label** conditioning in data (GLIGEN-like) if you extend the DiT cross-attn inputs ([IMPROVEMENTS.md](IMPROVEMENTS.md) §11.4).

---

## 3. Speech bubbles, lettering, and vertical text

**Industry pattern:** SVG bubbles, **tategaki** (vertical JP), SFX glyphs, OCR pass for legibility.

**SDX today**

- **`--text-in-image`**, **`--expected-text`**, **`--ocr-fix`** in `sample.py` / **`generate_book.py`**.
- **`utils/generation/text_rendering.py`** for OCR-aware masks and validation.
- **Test-time pick:** **`--pick-best combo`** uses OCR alignment when expected text is set ([`utils/quality/test_time_pick.py`](../utils/quality/test_time_pick.py)).

**Prompt hints:** see **`prompt_lexicon.tategaki_hint`**, **`lettering_negative_addon`**.

---

## 4. Story rhythm and pacing

**Pattern:** page = beat; **gutter** and **panel size** imply time jumps—usually **layout authoring**, not one giant diffusion forward.

**SDX today**

- **You** split story into lines in **`pages.txt`** (or use **`scripts/tools/book_scene_split.py`**).
- Per-page **`--book-type`** / style presets adjust the global look (manga vs comic vs cover).

---

## 5. Art direction presets (shonen / shoujo / webtoon)

**Pattern:** template styles for ink weight, screentone, color vs B&W, **vertical scroll** vs print page.

**SDX today**

- **`--lexicon-style`** on **`generate_book.py`** merges snippets from [`prompt_lexicon.py`](../pipelines/book_comic/prompt_lexicon.py).
- **`--aspect-preset`** suggests width/height for **webtoon** vs **print**-ish frames.

---

## 6. Quality and “anti-AI look” for ink

**Pattern:** avoid plastic gradients; favor **halftone**, **ink bleed**, paper grain.

**SDX today**

- **`--book-accuracy`** (`balanced` → `maximum` → **`production`**: more candidates + stricter merged negatives) + **`utils/quality.naturalize`** post-pass in the book pipeline.
- Dataset captions that mention **screentone**, **ink**, **paper** (see lexicon).

---

## References (external; not dependencies)

- GLIGEN (grounded generation) — spatial phrase grounding.
- ControlNet — edge/depth/sketch control for structure.
- Commercial manga tools — character bibles, bubble editors, vertical type (workflow ideas only).

---

## See also

- [BOOK_MODEL_EXCELLENCE.md](BOOK_MODEL_EXCELLENCE.md) — full checklist for strong book/comic output.
- [pipelines/book_comic/README.md](../pipelines/book_comic/README.md)
- [LANDSCAPE_2026.md](LANDSCAPE_2026.md) — authenticity, multi-stage pipelines
- [IMPROVEMENTS.md](IMPROVEMENTS.md) §11–12
