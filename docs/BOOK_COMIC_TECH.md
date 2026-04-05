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

## Best output checklist

There is no single checkpoint that is “the best book model” for every art style and language. **Quality comes from the stack:** data, training, inference workflow, and test-time selection.

### Data (what the model actually learns)

- **Panels & gutters:** Dataset should include real sequential layouts (not only single illustrations).
- **Legible text in-image:** If you need speech bubbles, train with **rendered text** in the image and captions that describe placement (see [REGION_CAPTIONS.md](REGION_CAPTIONS.md)).
- **Consistent characters:** Same identity tags + reference crops / character sheets in captions when possible.
- **Negatives in training:** Optionally pair `negative_caption` with borderline failure cases.

### Training (repo)

- Use root `train.py` with a **book/comic-focused** JSONL; keep resolution and aspect mix aligned with your target format (manga B/W, webtoon tall, US print, etc.).
- For layout-aware captions, use `parts` / `region_captions` so the model sees **where** text and figures sit.

### Inference workflow (`generate_book.py`)

| Goal | Knob |
|------|------|
| More chances to get a good page | `--book-accuracy production` or `--sample-candidates N` + `--pick-best combo` |
| Stricter anti-failure negatives | `--book-accuracy production` (adds `PRODUCTION_TIER_NEGATIVE_ADDON` in `prompt_lexicon`) or edit `--negative-prompt` |
| Style tone | `--lexicon-style` (e.g. `graphic_novel`, `yonkoma`, `webtoon`) |
| Canvas shape | `--aspect-preset` (e.g. `double_page_spread`, `print_us_comic`, `webtoon_tall`) |
| Print/cover polish (prompt hints) | `--include-print-finish`, `--include-cover-spotlight` |
| Readable quoted text | `--expected-text`, `--text-in-image`, optional `--ocr-fix` |

### Test-time pick-best

- **`combo`** uses CLIP + edge (+ OCR when expected text is set). More candidates (`production` = 6 by default) cost time but improve worst-page outcomes.
- Enable **`--save-prompt`** (or use `balanced`+ presets) to audit what was actually sent to the sampler.

### Post-process

- `book_helpers` applies optional **sharpen + naturalize** after each PNG; `production` uses stronger defaults than `maximum`.

### Character consistency across pages

- Use **`--character-sheet`**, **`--anchor-face`**, **`--edge-anchor`**, and speech-bubble / OCR options as documented in **[pipelines/book_comic/README.md](../pipelines/book_comic/README.md)** and the **SDX mapping sections above** (§1–3).

**Summary:** Train on real sequential art + in-panel text, then use **`--book-accuracy production`**, lexicon/aspect flags, pick-best, and OCR/anchoring where needed—not a single magic `.pt` name.

---

## See also

- [pipelines/book_comic/README.md](../pipelines/book_comic/README.md)
- [LANDSCAPE_2026.md](LANDSCAPE_2026.md) — authenticity, multi-stage pipelines
- [IMPROVEMENTS.md](IMPROVEMENTS.md) §11–12
