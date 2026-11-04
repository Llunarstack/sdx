# Prompt stack — how text becomes conditioning

This doc ties together the **Python modules** and **CLI flags** that shape the positive and negative strings before T5 (or triple-encoder) runs. For **copy-paste recipes**, see [PROMPT_COOKBOOK.md](#prompt-cookbook-copypaste-recipes-using-sdx-features).

## PromptStack v2 (unified pipeline)

`sample.py` post-checkpoint prompt assembly runs through **`utils.prompt.stack`**:

| Stage | Role |
|-------|------|
| Intelligence | Analyze complexity/domains; optional light quality hints |
| Guidance | Shortcomings, art medium, style guidance, photo realism |
| Negative bootstrap | Default neg, staged layout/MI/VD negs, flag negs |
| Content controls | Resolved infer + `apply_content_controls` |
| Clauses | Named bundles (`--prompt-clauses uncensored.fidelity,hands.stable`) |
| Post-enrich | Character/scene/scale negatives |
| Prompt breakdown | Optional T5 section ordering |
| Neg filter | Remove pos/neg token conflicts |

**Preview without GPU:**

```bash
python -m scripts.tools preview_prompt_stack --prompt "1girl, portrait" --quality-pack top
SDX_PROMPT_STACK_TRACE=1 python -m scripts.tools preview_prompt_stack --prompt "..." --json
```

**Training:** `merge_guidance_for_training_caption()` reuses the guidance stage on JSONL captions.

**API:** `from utils.prompt.stack import run_prompt_stack, PromptContext, apply_sample_prompt_stack`

---

## End-to-end path (inference)

1. **User input** — `--prompt`, optional `--negative-prompt`, `--tags` / `--tags-file`, `--lora-trigger`, character sheet / scene blueprint JSON, emphasis `(word)` / `[word]` in `sample.py`.
2. **Defaults** — If negative is empty, `config.prompts` supplies `DEFAULT_NEGATIVE_PROMPT` (or text-in-image variants when `--text-in-image` or detected phrases).
3. **Optional prefixes** — `--hard-style`, `--naturalize` / `--naturalize-deep`, `--anti-bleed`, `--diversity`, originality tokens, etc. (see `sample.py` after the prompt is assembled).
4. **`utils.prompt.content_controls.apply_content_controls`** — Quality packs (including **`micro_detail`** texture/material), **`adherence_pack`** (standard/strict literalism), pose/view/domain, Civitai trigger banks, **one-shot** scaffolding, **anti-AI** / **human-media** / **LoRA scaffold** packs, composition guards, etc. (`--less-ai`, `--auto-content-fix`, …). Tag packs are built-in (no CSV). With **`--auto-content-fix`** (default on), long prompts can auto-pick **`standard`** / **`strict`** adherence from length + keywords. **`--uncensored-mode`** (default on) disables character-sheet safety sanitization.
5. **Extra negatives** — Scale distortion, character/scene negatives, **multi-LoRA** `LORA_STACK_NEGATIVE` when two+ `--lora` paths.
6. **Art direction + token budget** — `utils.prompt.art_direction` plans shot, lens, light and palette from your own words (`--art-direction auto|off`); `utils/prompt/stack.py` rebuilds the positive as your tags, then the plan caption, then pack text that fits `--prompt-budget` tokens (default 96) and the T5 limit (300). Pack text the plan contradicts (skin/hands with nobody in frame, camera terms on a painting) is dropped; your words never are. The plan is saved as `art_direction` in `--pick-report-json`.
7. **Conflict filter** — `utils.prompt.text.filter_negative_by_positive` drops whole negative tags that name what you asked for (multi-word defect tags such as `bad hands` stay) unless `--no-neg-filter`.
8. **Encode** — T5 (or triple bundle) → `encoder_hidden_states` / CFG uncond.

**Debug:** if `apply_content_controls` fails, `sample.py` prints a warning; set **`SDX_DEBUG=1`** for a full traceback.

**Training:** emphasis in JSONL captions is only applied to the DiT forward if you pass **`--train-prompt-emphasis`** (strips brackets for T5 and passes `token_weights`; see [TRAINING_TEXT_TO_PIXELS.md](GUIDES.md)).

---

## Key files

| Piece | Path | Role |
|-------|------|------|
| Content controls | [`utils/prompt/content_controls.py`](../utils/prompt/content_controls.py) | `apply_content_controls`, `infer_content_controls_from_prompt` |
| Art direction planner | [`utils/prompt/art_direction.py`](../utils/prompt/art_direction.py) | `plan_art_direction`, `ArtDirection.caption` / `to_dict`, `gate_tag` |
| Token budget | [`utils/prompt/stack.py`](../utils/prompt/stack.py) | `stage_token_budget`, `fit_tags` |
| Pos/neg conflict filter | [`utils/prompt/text.py`](../utils/prompt/text.py) | `filter_negative_by_positive` |
| Domain / anti-AI strings | [`config/prompts.py`](../config/prompts.py) | Defaults, `ANTI_AI_*`, `NATURAL_LOOK_*`, `LORA_STACK_NEGATIVE` |
| Sampling CLI | [`sample.py`](../sample.py) | Orchestrates the chain above |
| Emphasis → token weights | [`utils/prompt/text.py`](../utils/prompt/text.py) | `( )` / `[ ]` parsing + T5 `offset_mapping` weights; **`train.py --train-prompt-emphasis`** for training–inference parity |
| Originality injection | [`utils/prompt/art_direction.py`](../utils/prompt/art_direction.py) | **`sample.py --originality`** / **`train.py --train-originality-prob`** — composition tokens after subject tags |
| Refinement smoke | [`inference.py`](../inference.py) | Load ckpt + diffusion; optional `--verify` |

---

## Preview without a GPU

```bash
python -m scripts.tools preview_generation_prompt --prompt "1girl, red dress" --quality-pack top --less-ai
```

Prints **effective positive / negative** after content controls + the same conflict filter as `sample.py` (subset of flags; no checkpoint). See [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md).

---

## Flag cheat sheet (sampling)

| Intent | Flags (examples) |
|--------|------------------|
| Uncensored / no sheet sanitization | `--uncensored-mode` (default on) / `--no-uncensored-mode` |
| Stronger first-try composition | `--one-shot-boost` (default on) / `--no-one-shot-boost` |
| Keyword inference | `--auto-content-fix` (default on) / `--no-auto-content-fix` |
| Less “AI render” look | `--less-ai`, `--anti-ai-pack lite\|strong`, `--human-media photographic\|dslr\|film`, `--naturalize` |
| LoRA fusion | `--lora-scaffold blend\|…`, `--lora-scaffold-auto` |
| Quality ladder | `--quality-pack top\|one_shot\|micro_detail\|…`, `--boost-quality` |
| Prompt literalism | `--adherence-pack standard\|strict` (optional auto via `--auto-content-fix`) |
| Presets | `--preset`, `--op-mode`, `--hard-style` |

Full list: `python sample.py --help`.

---

## Training vs inference

- **Training** uses captions from folders / JSONL (emphasis, regional captions, optional negative fields). Enable **`--boost-adherence-caption`** on `train.py` to prepend adherence tags from [`data/caption_utils.py`](../data/caption_utils.py) (`prepend_adherence_boost`) for stronger literal conditioning. The **same** quality/domain ideas are documented in [MODEL_WEAKNESSES.md](QUALITY.md) and [prompt_domains.py](../config/prompts.py) for caption writing.
- **Inference** adds the dynamic stack above so one checkpoint can be steered without retraining.

---

## See also

- [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) — diffusion loop + T5 + VAE
- [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) — config ↔ checkpoint ↔ sample (§13)
- [QUALITY.md](QUALITY.md) — Civitai tips + community issue matrix
- [scripts/tools/README.md](../scripts/tools/README.md) — `preview_generation_prompt.py`
- [config/prompts.py](../config/prompts.py) — default and anti-AI negatives

---

## Prompt cookbook: copy‑paste recipes using SDX features

Short, opinionated recipes that use the flags and presets you’ve added (`--preset`, `--op-mode`, `--hard-style`, `--naturalize`, `--anti-bleed`, `--diversity`, etc.).

**How the prompt is built (modules and order):** see **[PROMPT_STACK.md](#prompt-stack--how-text-becomes-conditioning)**. **Preview without sampling:** `python -m scripts.tools preview_generation_prompt --prompt "..."`.

All examples assume:

```bash
cd sdx
```

---

### 1. Photorealistic portrait (SDXL‑style, minimal AI slop)

**Goal:** Natural, realistic portrait with strong adherence and minimal plastic look.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "portrait of a woman, soft window light, shallow depth of field, sharp eyes, 50mm lens" \
  --negative-prompt "" \
  --preset sdxl \
  --op-mode portrait \
  --width 768 --height 1024 \
  --out portrait_sdxl.png
```

What this does:
- `--preset sdxl`: photorealistic defaults (cfg-scale ~6.5, cfg-rescale, steps, hard-style realistic, naturalize on).
- `--op-mode portrait`: turns on diversity, anti-artifacts, naturalize for faces.

---

### 2. Flux‑style gritty photo (safer CFG, no grid)

**Goal:** FLUX‑like realism but with explicit safeguards against CFG burn / grid artifact.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "street photo of a man walking in the rain, city lights, cinematic, moody" \
  --preset flux \
  --op-mode portrait \
  --width 896 --height 1152 \
  --out flux_style.png
```

Preset `flux`:
- Uses CFG ≈ 3.5 and cfg-rescale to avoid FLUX‑style grid artifacts.
- Enables `--naturalize`, `--anti-bleed`, `--diversity`, `--anti-artifacts`, `--strong-watermark`.

---

### 3. 2.5D / semi‑realistic anime character

**Goal:** Anime‑leaning character with semi‑realistic shading and strong tag control.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "1girl, school uniform, medium shot, dynamic lighting, detailed hair" \
  --preset anime \
  --op-mode anime_char \
  --tags "1girl, teen, medium chest, long hair, dynamic lighting" \
  --width 768 --height 1024 \
  --out anime_25d.png
```

Preset `anime` + `anime_char`:
- Hard-style `style_mix` (2.5D / semi‑realistic).
- Anti‑bleed + diversity + anti‑artifacts + strong watermark negative.

---

### 4. Hard 3D/CG shot (no bleed, strong structure)

**Goal:** 3D render that doesn’t mush colors/objects together.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "3d render of a red car parked next to a blue house, sunset, detailed reflections" \
  --preset sdxl \
  --hard-style 3d \
  --anti-bleed \
  --anti-artifacts \
  --width 1024 --height 768 \
  --out hard_3d.png
```

Key pieces:
- `--hard-style 3d`: prepends 3D‑specific tags (octane, 3d render, etc.).
- `--anti-bleed`: adds distinct-colors positive + color‑bleed negative.
- `--anti-artifacts`: adds artifact negative (white dots, spiky, pixel-stretch).

---

### 5. Full‑body character without two‑head

**Goal:** Standing full‑body character in portrait aspect, avoiding two‑head and missing legs.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "1girl, standing, long dress, legs, shoes, full body, outdoors, sunset" \
  --preset anime \
  --op-mode fullbody \
  --width 768 --height 1152 \
  --out fullbody.png
```

Tips:
- Describe lower body explicitly: `standing, long dress, legs, shoes`.
- Use portrait aspect for full body; 1:1 for head/shoulder.

---

### 6. Seed exploration (find your god seeds)

Use the seed explorer script to rapidly try many seeds with presets:

```bash
python -m scripts.tools seed_explorer \
  --ckpt results/.../best.pt \
  --prompt "portrait of a woman, studio lighting" \
  --preset sdxl \
  --op-mode portrait \
  --rows 2 --cols 4 \
  --out-dir seed_explorer/portrait
```

This will:
- Generate 8 seeds (2×4 grid), each saved to `seed_explorer/portrait/seed_*.png`.
- Save a grid image and `seeds.json` listing the seeds.

Pick the best seed(s) and re‑run `sample.py` with your favorite settings.

---

### 7. Checkpoint regression suite (eval prompts)

Run a fixed prompt set so you can compare checkpoints/presets quickly:

```bash
python -m scripts.tools eval_prompts \
  --ckpt results/.../best.pt \
  --preset sdxl \
  --op-mode portrait \
  --out-dir eval_prompts/sdxl_v1
```

This writes `eval_prompts/sdxl_v1/index.json` plus PNGs for each prompt.

---

### 8. Normalize/OP-ify captions before training

If you have a big JSONL manifest, normalize tag order and apply hard-style/quality boosting in one pass:

```bash
python -m scripts.tools normalize_captions \
  --in manifest.jsonl \
  --out manifest_normalized.jsonl
```

The script updates:
- `caption` (subject/age/height/build/anatomy ordering + hard-style/quality/domain boosts)
- `negative_caption` / `negative_prompt` (when present)

---

### 9. More novel results (Originality)

If you notice your generations feel too templated / repetitive, use `--originality`.

It:
- injects deterministic “unique composition” tokens near the start of the prompt
- auto-sets `--creativity` (when supported) for extra diversity
- slightly lowers CFG so the model explores beyond the most literal token match

Example:

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "portrait of a woman, studio lighting" \
  --preset sdxl \
  --op-mode portrait \
  --originality 0.6 \
  --out novel.png
```

---

## Style and artist tags (PixAI, Danbooru, Gelbooru)

Make the model **really good at styles** by using artist and style tags from tag-based image boards. The pipeline extracts style from captions and prompts and feeds it into the style conditioning head.

---

### How it works

1. **Training**  
   - JSONL can include a `"style"` field (e.g. `"style": "oil painting, miyazaki"`).  
   - If `style` is **empty**, the dataset **auto-fills** it from the caption using patterns like "by X", "style of X", `artist:name`, and a list of known artist/style tags (`config/styles.py`).  
   - Style is encoded with T5 and blended via `style_embed_dim` / `style_strength`, so the model learns to follow it without overpowering the main prompt.

2. **Inference**  
   - Use `--style "oil painting, vivid"` as usual.  
   - Or put the style in the prompt and use **`--auto-style-from-prompt`**: the script extracts a style string from the prompt (same patterns as in training) and uses it as the style conditioning.

---

### Supported patterns (extraction)

- **Phrases:** `by artist_name`, `art by X`, `drawn by X`, `in the style of X`, `style of X`, `like X`, `similar to X`  
- **Tag-board:** `artist:name`, `style:name` (Danbooru/Gelbooru style)  
- **Known tags:** If the caption contains a tag from `ARTIST_STYLE_TAGS` in `config/styles.py` (e.g. `ghibli`, `oil_painting`, `makoto_shinkai`), that tag is used as the style when no explicit style is set.

---

### Training with artist/style tags

1. **Explicit style in JSONL**  
   ```json
   {"image_path": "img.png", "caption": "1girl, forest", "style": "studio ghibli, miyazaki"}
   ```

2. **Style only in caption (auto-extraction)**  
   - Use captions like: `1girl, forest, drawn by makoto shinkai` or `1girl, artist:miyazaki`.  
   - Leave `"style": ""` or omit it; the dataset will fill `style` from the caption when `extract_style_from_caption=True` (default).

3. **Add your own tags**  
   - Edit `config/styles.py` and extend `ARTIST_STYLE_TAGS` with artist names or style tags from your dataset (e.g. from Danbooru/PixAI exports).

---

### Inference

```bash
# Explicit style
python sample.py --ckpt .../best.pt --prompt "1girl in a forest" --style "studio ghibli, miyazaki" --style-strength 0.7 --out out.png

# Style from prompt (no --style needed)
python sample.py --ckpt .../best.pt --prompt "1girl in a forest, drawn by makoto shinkai" --auto-style-from-prompt --out out.png
```

Requires a model trained with `--style-embed-dim 4096` (or your text_dim).

---

### Files

| File | Role |
|------|------|
| `config/styles.py` | Patterns and `ARTIST_STYLE_TAGS`; `extract_style_from_text()`. |
| `data/caption_utils.py` | `DOMAIN_TAGS["style_artist"]`: tags boosted for style learning. |
| `data/t2i_dataset.py` | Auto-fill `style` from caption when empty (`extract_style_from_caption=True`). |
| `sample.py` | `--auto-style-from-prompt`: extract style from prompt and use as style conditioning. |

---

### Tips

- Use **consistent** artist/style tags in training data (same spelling and format as on your tag board).  
- For strong style, keep **style_strength** in the 0.6–0.8 range so style doesn’t overwhelm the subject.  
- Add artist names from your dataset to `ARTIST_STYLE_TAGS` so they are recognized even without "by" or "artist:" in every caption.

---

## Regional & layout captions (JSONL)

Train the text encoder + DiT on **where things are**, not only a single global tag string — **without** changing model architecture.

### Idea

1. From each training image you (or a VLM/segmentation pipeline) produce **short labels per region**: subject, clothing, hands, background, etc.
2. Store them in the manifest as **`parts`** (dict) and/or **`region_captions`** (list).
3. At load time, `Text2ImageDataset` merges them into one T5 string using a fixed **`[layout]`** prefix so the model learns a consistent “parts + global scene” pattern.

Optional: combine with **`--crop-mode random`** so crops + layout text encourage local detail.

### JSONL fields

| Field | Type | Description |
|:------|:-----|:------------|
| `caption` | string | Global prompt (tags or sentence) — **required** as today |
| `parts` | object | e.g. `{"subject": "1girl, long hair", "clothing": "red dress", "background": "forest"}` |
| `region_captions` | list | Strings and/or `{"label": "...", "text": "..."}` |
| `segments` | alias | Same as `region_captions` |

You may send **both** `parts` and `region_captions`; they are merged (`parts` first, then list).

#### Example

```json
{
  "image_path": "/data/img001.png",
  "caption": "1girl, masterpiece, best quality, full body",
  "negative_caption": "lowres, bad anatomy",
  "parts": {
    "subject": "young woman, long black hair, looking at viewer",
    "clothing": "white blouse, pleated skirt",
    "hands": "visible hands, five fingers",
    "background": "city street, sunset, bokeh"
  },
  "region_captions": [
    {"label": "foreground", "text": "character centered"},
    {"label": "lighting", "text": "warm rim light"}
  ]
}
```

Merged into training (default **`--region-caption-mode append`**):

`… global caption after boosts … . [layout] subject: … | clothing: … | … | foreground: … | lighting: …`

### CLI (`train.py`)

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `--region-caption-mode` | `append` | `append` — after global caption; `prefix` — layout before global; `off` — ignore regions |
| `--region-layout-tag` | `[layout]` | Token(s) before the regional block; use `""` to disable |

### Inference

At sample time, write prompts in the **same style** your data used, e.g. end with  
`. [layout] subject: … | background: …`  
or rely on a strong global caption only — the model is not required to see `[layout]` at inference if you never trained with it (use `--region-caption-mode off` to A/B).

### See also

- [docs/FILES.md](FILES.md) — data module entry points  
- [README § Data format](../README.md#data-format) — manifest overview

---

## Generating 3D, realistic, interior/exterior and other hard domains

Many models struggle with **3D-rendered** looks, **photorealistic** images, **interior** and **exterior** scenes (architecture, rooms, landscapes), and **style mixes** (2.5D, semi-realistic, photorealistic anime). SDX is set up so these domains are learned well.

### How we handle them

1. **Hard-style boosting** — 3D, photorealistic, and style-mix tags are boosted **first** (stronger than general domains) so the model anchors on the right look. See `data/caption_utils.py`: `HARD_STYLE_TAGS`, `boost_hard_style_tags`. Training applies this before quality/domain boosts.
2. **Domain tag boosting** — When your training captions (or inference prompts) contain tags like `interior design`, `exterior`, the data pipeline moves them to the **front** of the caption (once, deduplicated) via `DOMAIN_TAGS`, `boost_domain_tags`.
3. **Recommended prefixes** — `config/prompts.py` lists `HARD_STYLE_RECOMMENDED_PROMPTS`. At inference use `--hard-style 3d | realistic | 3d_realistic | style_mix` to prepend the right tags, with a matching `--negative-prompt` (table below).

### Training data

- **3D**: Include images with captions like `3d render, octane render, ...` or `blender 3d, isometric, ...`.
- **Realistic**: Use `photorealistic`, `realistic`, `raw photo`, `natural lighting`, etc.
- **Interior**: Use `interior design`, `indoor`, `room`, `living room`, `modern interior`, etc.
- **Exterior**: Use `exterior`, `outdoor`, `architecture`, `building`, `street view`, etc.

The pipeline automatically boosts these when present, so the model gets a strong signal for these domains.

### Inference

**Hard styles (3D, realistic, style mixes):** Use `--hard-style 3d | realistic | 3d_realistic | style_mix` to prepend recommended tags. Optionally add the matching `--negative-prompt` below.

| Hard style   | Example (prepended by --hard-style)                    | Example negative |
|--------------|--------------------------------------------------------|------------------|
| **3d**       | `3d render, octane render, masterpiece, best quality, 8k` | flat, 2d, blurry, bad proportions |
| **realistic**| `photorealistic, raw photo, natural lighting, masterpiece, 8k uhd` | unrealistic, cartoon, blurry |
| **3d_realistic** | `3d render, photorealistic, octane render, natural lighting, masterpiece` | flat, wrong lighting, plastic look |
| **style_mix** | `2.5d, semi-realistic, masterpiece, best quality, detailed` | blurry, incoherent style, messy |

**Other domains (interior, exterior):** Use the same tags in your prompt and negatives like these:

| Domain    | Example prompt prefix                                      | Example negative |
|-----------|------------------------------------------------------------|------------------|
| **Interior**  | `interior design, indoor, room, modern interior, masterpiece` | empty room, distorted perspective, blurry |
| **Exterior**  | `exterior, outdoor, architecture, building, masterpiece`   | flat, wrong perspective, blurry |

Full prefix lists: `config/prompts.py` → `HARD_STYLE_RECOMMENDED_PROMPTS`. **Style mixing and LoRAs:** see [QUALITY.md](QUALITY.md).

### Other hard areas

The same mechanism supports **hands**, **correct anatomy**, **perspective**, **symmetry**, **text/lettering**, **complex composition**, etc. — see `data/caption_utils.py` → `DOMAIN_TAGS["other_hard"]`, `DOMAIN_TAGS["anatomy"]`, `DOMAIN_TAGS["avoid_failures"]`. Use those tags in captions so they get boosted and the model learns them. For **what other models fail at** (hands, faces, double head, text, multiple subjects) and **concrete fixes**, see [QUALITY.md](QUALITY.md) (Part 2 — model weaknesses).
