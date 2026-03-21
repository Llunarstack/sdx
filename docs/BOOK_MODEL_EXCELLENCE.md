# Book / comic / manga “best model” checklist

There is no single checkpoint that is “the best book model” for every art style and language. **Quality comes from the stack:** data, training, inference workflow, and test-time selection.

## 1. Data (what the model actually learns)

- **Panels & gutters:** Dataset should include real sequential layouts (not only single illustrations).
- **Legible text in-image:** If you need speech bubbles, train with **rendered text** in the image and captions that describe placement (see `docs/REGION_CAPTIONS.md`).
- **Consistent characters:** Same identity tags + reference crops / character sheets in captions when possible.
- **Negatives in training:** Optionally pair `negative_caption` with borderline failure cases.

## 2. Training (repo)

- Use root `train.py` with a **book/comic-focused** JSONL; keep resolution and aspect mix aligned with your target format (manga B/W, webtoon tall, US print, etc.).
- For layout-aware captions, use `parts` / `region_captions` so the model sees **where** text and figures sit.

## 3. Inference workflow (`generate_book.py`)

| Goal | Knob |
|------|------|
| More chances to get a good page | `--book-accuracy production` or `--sample-candidates N` + `--pick-best combo` |
| Stricter anti-failure negatives | `--book-accuracy production` (adds `PRODUCTION_TIER_NEGATIVE_ADDON` in `prompt_lexicon`) or edit `--negative-prompt` |
| Style tone | `--lexicon-style` (e.g. `graphic_novel`, `yonkoma`, `webtoon`) |
| Canvas shape | `--aspect-preset` (e.g. `double_page_spread`, `print_us_comic`, `webtoon_tall`) |
| Print/cover polish (prompt hints) | `--include-print-finish`, `--include-cover-spotlight` |
| Readable quoted text | `--expected-text`, `--text-in-image`, optional `--ocr-fix` |

## 4. Test-time pick-best

- **`combo`** uses CLIP + edge (+ OCR when expected text is set). More candidates (`production` = 6 by default) cost time but improve worst-page outcomes.
- Enable **`--save-prompt`** (or use `balanced`+ presets) to audit what was actually sent to the sampler.

## 5. Post-process

- `book_helpers` applies optional **sharpen + naturalize** after each PNG; `production` uses stronger defaults than `maximum`.

## 6. Character consistency across pages

- Use **`--character-sheet`**, **`--anchor-face`**, **`--edge-anchor`**, and speech-bubble / OCR options as documented in **`pipelines/book_comic/README.md`** and **`docs/BOOK_COMIC_TECH.md`**.

---

**Summary:** Train on real sequential art + in-panel text, then use **`--book-accuracy production`**, lexicon/aspect flags, pick-best, and OCR/anchoring where needed—not a single magic `.pt` name.
