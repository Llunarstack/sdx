# Quality guide

Merged: sampling fixes, community issues, model weaknesses, and failure-mode reference.

---



---

## Part 1 — Sampling fixes and community issues

Sampling fixes (Civitai-style) and a **community issue matrix** (SDXL, Flux, Z-Image, etc.) with sample.py flags and config/prompt_domains.py references.

---

## Civitai-style quality tips

Based on common problems reported in Civitai checkpoint comments and SD community guides (oversaturation, blur, bad hands, wrong resolution). Use these with `sample.py` and training to get more reliable, high-quality output.

---

### 1. Oversaturation / “burned” colors (high CFG)

**Problem:** Images look overexposed or oversaturated when using strong guidance.

**Fixes:**

- **`--cfg-rescale 0.7`** — ComfyUI-style CFG rescale; keeps prompt adherence while reducing saturation (use with `--cfg-scale` 7–10).
- **Lower `--cfg-scale`** — Try 5–7 for realistic/SDXL-style models instead of 7.5–10.
- **Dynamic threshold** — `--dynamic-threshold-percentile 99.5 --dynamic-threshold-type percentile` to clamp extreme activations.

Example:
```bash
python sample.py --ckpt .../best.pt --prompt "..." --cfg-scale 7 --cfg-rescale 0.7 --out out.png
```

---

### 2. Blurry or soft output

**Causes:** Non-native resolution, too few steps, VAE/quantization, or scaling artifacts.

**Fixes:**

- **Use native resolution** — Prefer the model’s training size (e.g. 256 or 512). If you set `--width` / `--height` far from that, you’ll see a note; use `--vae-tiling` for large decodes to reduce VRAM and artifacts.
- **More steps** — 25–35 steps often improve clarity; we default to 50.
- **Post-process** — `--sharpen 0.3` and `--contrast 1.05` can help (see `utils/quality/quality.py`).
- **VAE** — Train/sample with a good VAE (e.g. sd-vae-ft-mse or sdxl-vae); avoid heavily quantized VAEs for final decode.

---

### 3. Bad hands / anatomy

**Problem:** Extra fingers, fused fingers, deformed hands or limbs.

**Fixes:**

- **Default negative prompt** — If you leave `--negative-prompt` empty, we use a Civitai-style default: `low quality, worst quality, blurry, bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, deformed, duplicate`. Override with `--negative-prompt "..."` if your model prefers minimal negatives (e.g. some SDXL-style models).
- **Positive cues** — In the prompt, describe hands in context: e.g. “holding a cup”, “hands on keyboard”, “correct hands, five fingers”.
- **Simpler poses** — “Arms crossed”, “hands in pockets” tend to work better than complex hand poses.
- **Inpainting** — For a single image, use `--init-image` + `--mask` to inpaint only the hand/face area at 0.4–0.5 strength.

See also `config/prompt_domains.py`: `ANATOMY_NEGATIVES`, `HAND_FIX_PROMPT_TIPS`.

---

### 4. Positive vs negative prompt conflict

**Problem:** You want “cat, dog, portrait” but your negative prompt says “dog, blurry”. CFG would then push away from *dog* as well, contradicting the positive prompt.

**Fix (default):** `sample.py` **filters the negative prompt** by removing any token that also appears in the positive (split on comma/space, case-insensitive). So with pos “cat, dog, portrait” and neg “dog, blurry”, the effective negative becomes “blurry” only. The model is guided away from blur without fighting your request for a dog.

- To **disable** this and use the raw negative prompt, pass **`--no-neg-filter`**.

---

### 5. Text in image (signs, lettering, legible text)

**Problem:** Diffusion models often render garbled or wrong text. The default negative includes “text”, which discourages *any* text in the image, so desired signs/lettering get suppressed.

**Fixes:**

- **Be explicit in the prompt** — Use phrases like “sign that says OPEN”, “text reading Hello in bold”, “label that says 24h”. Quote the exact text: “sign that says \"CAFE\"”.
- **Add quality cues** — “legible text”, “clear lettering”, “sharp text”, “readable”.
- **Use a text-friendly negative** — If you leave `--negative-prompt` empty and your prompt suggests text (e.g. “sign that says”, “lettering”), `sample.py` automatically uses a negative that avoids *bad* text (garbled, misspelled, watermark) but does **not** suppress desired text.
- **Force it** — Pass **`--text-in-image`** to always use the text-friendly default negative when you didn’t set a custom negative.

Example:
```bash
python sample.py --ckpt .../best.pt --prompt "storefront sign that says COFFEE in bold letters, neon style" --out sign.png
# Or explicitly: --text-in-image
```

**Training:** Include many examples with correct text in images and captions that describe the exact text (e.g. “sign that says OPEN”). The data pipeline boosts “text_in_image” domain tags (see `data/caption_utils.py`, `config/prompt_domains.py`: `TEXT_IN_IMAGE_PROMPT_TIPS`).

---

### 6. Tags and LoRAs

**Tag-based prompts:** Use `--tags "1girl, long hair, outdoors, sunset"` or `--tags-file path/to/tags.txt` (one tag per line or comma-separated). Tags are normalized and reordered for better adherence: **subject** (1girl, 1boy, etc.) → **age** (teen, young, adult, elderly) → **height** (tall, short, petite) → **build/size** (slim, muscular, curvy) → **anatomy/framing** (full body, portrait, bust) → **body parts** (hands, long hair, etc.) → rest. Use `--subject-first` with a plain `--prompt` to get the same ordering. You can omit `--prompt` when using `--tags` or `--tags-file`.

**LoRAs:** `sample.py` loads `.pt` and `.safetensors` LoRAs (`--lora path.safetensors path2.pt:0.6`). Many style/character LoRAs expect a **trigger word** in the prompt. Use `--lora-trigger "style name"` (or the trigger from the LoRA's model card); it is prepended to the prompt automatically when any LoRA is applied. Example:

```bash
python sample.py --ckpt .../best.pt --tags "1girl, smile" --lora my_style.safetensors --lora-trigger "MyStyle" --out out.png
```

This produces a prompt like `MyStyle, 1girl, smile` so the LoRA activates correctly while keeping tag order.

**People: height, age, size, anatomy:** The tag order puts **subject → age → height → build/size → anatomy/body parts** before other tags. So prompts like `long hair, 1girl, tall, teen, slim, full body` are reordered to `1girl, teen, tall, slim, full body, long hair`. Training data uses the same ordering and boosts these person-descriptor tags so the model learns height, age, body size, and anatomy (e.g. full body, portrait, hands, feet) reliably.

---

### 7. Complex and long prompts
**Problem:** Long or multi-part prompts get truncated (T5 max 300 tokens); the model may ignore later parts or lose adherence.

**Fixes:**
- **Put the main subject first** — e.g. "1girl, red dress, in a forest, sunset, ..." so the most important part is not truncated.
- **Use comma-separated tags** — clearer than long sentences for the model.
- **`--boost-quality`** — Prepends "masterpiece, best quality" to the prompt for stronger adherence (helpful for complex or challenging prompts).
- If the prompt has **>250 tokens**, `sample.py` prints a note; shorten or reorder so key elements are early.
- Use **(word)** to emphasize critical terms and **[word]** to de-emphasize.

See `config/prompt_domains.py`: `COMPLEX_PROMPT_TIPS`.

---

### 8. Challenging content (NSFW, surreal, abstract, weird)
**Problem:** Model under-adheres or produces generic output for mature, surreal, or unusual prompts.

**Fixes:**
- **Do not censor** — Use clear, descriptive tags in training data and at inference; the model learns what you show.
- **Quality tags help** — Add "masterpiece, best quality, detailed, sharp focus" so any challenging prompt gets stronger conditioning.
- **Be concrete** — For surreal/abstract, describe colors, layout, and mood; put subject first, then setting, then style.
- **`--boost-quality`** — Use for maximum adherence on challenging prompts.
- **Training:** Include diverse examples with consistent tags; the data pipeline boosts "complex" and "challenging" domain tags (see `data/caption_utils.py`, `config/prompt_domains.py`: `CHALLENGING_PROMPT_TIPS`).

---

### 9. Hard styles (3D, photorealistic, style mixes) and LoRA mixing

**Problem:** Many models blur 3D vs 2D, realistic vs illustrated, and mixed styles (2.5D, semi-realistic, photorealistic anime). Combining multiple LoRAs (e.g. style + character) can give muddy or oversaturated results.

**Fixes:**

- **`--hard-style`** — Use `--hard-style 3d`, `realistic`, `3d_realistic`, or `style_mix` to prepend recommended tags so the model anchors on the right look. Set `--negative-prompt` from `config/prompt_domains.py` → `HARD_STYLE_NEGATIVES` for that style (e.g. for 3d: "flat, 2d, blurry, bad proportions").
- **Training:** Captions that contain 3D, photorealistic, or style-mix phrases (e.g. "2.5d", "semi-realistic", "photorealistic anime") are boosted **first** in the pipeline (`data/caption_utils.py` → `HARD_STYLE_TAGS`, `boost_hard_style_tags`) so the model learns these hard styles well.
- **Style mixing:** Put the dominant style first in the prompt; use explicit phrases like "2.5d", "semi-realistic", "photorealistic anime". See `config/prompt_domains.py` → `STYLE_MIX_TIPS`.
- **Multiple LoRAs:** Use lower scales (0.5–0.6 each) so they blend; put the dominant LoRA first and its trigger at the start of the prompt. See `LORA_MIX_TIPS` in the same file. If output is muddy, try `--cfg-scale 5` or `--cfg-rescale 0.7`.

See [docs/DOMAINS.md](DOMAINS.md) for full hard-style prompts and negatives.

---

### 10. Wrong resolution warning

If you pass `--width` / `--height` much larger or smaller than the model’s native size (e.g. 1024 when trained at 256), `sample.py` prints a note. For best quality:

- Prefer the model’s native resolution, or
- Use a small multiple (e.g. 1.5×) and optionally `--vae-tiling` for large decode.

---

### 11. Quick reference: sample.py flags that address these issues

| Issue              | Flags to try |
|--------------------|--------------|
| Oversaturation     | `--cfg-rescale 0.7`, `--cfg-scale 5` or `7` |
| Blur               | Native resolution, `--steps 35`, `--sharpen 0.3`, `--vae-tiling` for large size |
| Bad hands/anatomy  | Default negative (or add `bad hands, extra fingers`), describe hands in prompt |
| Pos/neg conflict   | Default: neg is filtered so tokens in pos are removed from neg; use `--no-neg-filter` to disable |
| Text in image      | Describe exact text in prompt (“sign that says X”); use `--text-in-image` or leave neg empty so text-friendly negative is used |
| Tags / LoRAs        | `--tags "tag1, tag2"` or `--tags-file path`; `--lora path.safetensors --lora-trigger "word"` for style/character LoRAs |
| Reproducibility     | `--save-prompt` writes prompt, seed, steps to a `.txt` next to the image; `--subject-first` reorders comma-separated prompt (subject first) |
| Complexamples/long prompts | Put subject first; use `--boost-quality`; keep key elements in first ~250 tokens |
| Challenging (NSFW, surreal) | No censorship; add quality tags; use `--boost-quality`; see CHALLENGING_PROMPT_TIPS |
| Hard styles (3D, realistic, 2.5D) | `--hard-style 3d | realistic | 3d_realistic | style_mix`; set negative from `HARD_STYLE_NEGATIVES`; see STYLE_MIX_TIPS, LORA_MIX_TIPS |
| Multiple LoRAs / style mix | Lower LoRA scales (0.5–0.6); dominant LoRA first; `--cfg-rescale 0.7` if muddy |
| Large output OOM   | `--vae-tiling` |
| Softer / more natural | `--cfg-scale 5` |
| Less AI-looking / plastic | `--naturalize` (negative + prompt prefix + film grain); tune `--naturalize-grain 0.02` |
| Small artifacts polish | Refinement is on by default; disable with `--no-refine` (or tune `--refine-t`) |
| Color/concept bleeding (SDXL) | `--anti-bleed`; put spatial relations early; see [Community model issues](#community-model-issues) |
| Repetitive / same face (Flux) | `--diversity`; see `EMOTION_PROMPT_TIPS`, `DIVERSITY_POSITIVE` |
| Less templated / more novel | `--originality 0.3-0.8` (injects unique composition tokens + tunes creativity/CFG) |
| White dots / speckles / spiky artifacts | `--anti-artifacts`; “particle” in negative |
| Stubborn watermarks (e.g. Illustrious) | `--strong-watermark` |
| V-pred burn (NoobAI) | `--cfg-scale 4.5`, `--cfg-rescale 0.7`; see `CFG_BURN_TIPS` |
| FLUX grid artifact | CFG ≤ 3.5, LoRA strength ≤ 1.20; see `FLUX_GRID_ARTIFACT_TIPS` |
| Full body / two-head | Add standing, legs, long dress, shoes; 1:1 for head shot; see `FULL_BODY_AND_TWO_HEAD_TIPS` |
| Negative prompt ineffective | Be specific (e.g. extra fingers), not vague; see `NEGATIVE_PROMPT_BEST_PRACTICES` |

---

### 12. FLUX grid artifact and full-body / two-head (community fixes)

**FLUX grid:** Visible grid pattern in dark areas (worse with Depth/Canny and upscaling). From [FLUX GitHub #406](https://github.com/black-forest-labs/flux/issues/406): keep **CFG at 3.5 or lower** and **LoRA strength at or below 1.20**; avoid overtrained LoRAs. See `config/prompt_domains.py` → `FLUX_GRID_ARTIFACT_TIPS`.

**Full body / two-head:** From [Stable Diffusion Art](https://stable-diffusion-art.com/common-problems-in-ai-images-and-how-to-fix-them/): (1) For full body, add **standing, long dress, legs, shoes** (describe what should appear) rather than only “full body portrait”. (2) Use **1:1** for head/shoulder shots to avoid two-head; use portrait aspect for full-body so one body fills the frame. (3) Garbled faces often need more pixels: use close-up, Hi-Res Fix, better VAE, or face restorer/inpainting. See `FULL_BODY_AND_TWO_HEAD_TIPS`, `GARBLED_FACE_TIPS`.

---

### 13. Making images look less AI-generated

**Problem:** Outputs look too smooth, plastic, waxy, or “obviously generated.”

**Fixes:**

- **`--naturalize`** — One flag that: (1) adds an **anti-AI-look negative** (plastic skin, oversmooth, airbrushed, waxy, doll-like, synthetic, CGI, uncanny, etc.) so the model is steered away from those traits; (2) prepends **natural-look positive** hints (film grain, natural skin texture, subtle imperfections, raw photo, natural lighting) to the prompt; (3) runs a **post-process**: subtle film grain + slight micro-contrast on the decoded image to break up the plastic smoothness.
- **`--naturalize-grain 0.02`** — Adjust grain strength (0 = no grain, 0.01–0.03 typical). Use 0 if you only want the prompt/negative changes.
- **Lower CFG** — Try `--cfg-scale 5` or `6` for a less “pushed” look.
- **Training:** Include diverse, non-perfect references and captions that mention “natural skin texture,” “film grain,” “imperfections,” so the model learns a less polished look. You can add `ANTI_AI_LOOK_NEGATIVE`-style terms to your training negatives (see `config/prompt_domains.py`).

---

### 14. Training-side tips (fewer issues at inference)

- **Negative prompt in data** — Include quality/anatomy negatives in captions or the dataset so the model learns to avoid them (see `RECOMMENDED_NEGATIVE_BY_DOMAIN`, `ANATOMY_NEGATIVES` in `config/prompt_domains.py`).
- **Min-SNR loss** — `--min-snr-gamma 5` (or similar) stabilizes training and can improve clarity.
- **Native resolution** — Train at the resolution you plan to use for inference to avoid resolution mismatch blur.

These tips align with common Civitai and A1111/ComfyUI recommendations; adjust per checkpoint if a model card suggests different settings.

---

## Community model issues

Community-reported problems (SDXL, Flux/Klein, Illustrious/NoobAI, Z-Image, Civitai) and the mitigations available in SDX: **config/prompt_domains.py** tips/negatives, **sample.py** flags, and **training** practices.

**Sources:** Reddit r/StableDiffusion, Civitai articles & comments, FLUX GitHub (#406 grid artifact), [Stable Diffusion Art](https://stable-diffusion-art.com/common-problems-in-ai-images-and-how-to-fix-them/), ComfyUI prompt-engineering docs, Black Forest Labs FLUX.

---

### Quick reference: issue → mitigation

| Issue | Models | Mitigation |
|-------|--------|------------|
| **Concept/color bleeding** (red shirt + blue pants → purple) | SDXL | `--anti-bleed`; prompt: `CONCEPT_BLEEDING_POSITIVE`; negative: `CONCEPT_BLEEDING_NEGATIVE`; training: boost `DOMAIN_TAGS["concept_bleed"]` |
| **Poor spatial awareness** (behind, next to, under wrong) | SDXL | Put spatial phrases early; see `SPATIAL_AWARENESS_TIPS` |
| **Plastic/waxy skin** | SDXL, base Flux | `--naturalize`; `ANTI_AI_LOOK_NEGATIVE`, `NATURAL_LOOK_POSITIVE` |
| **Inconsistent prompt following** (ignores end of long prompt) | SDXL | Put key details first; `--subject-first`; `--boost-quality`; keep &lt;250 tokens; see `COMPLEX_PROMPT_TIPS` |
| **Repetitive / “default” face** (Flux face) | Flux, Klein | `--diversity`; negative: `FLUX_FACE_DIVERSITY_NEGATIVE`; positive: `DIVERSITY_POSITIVE` |
| **Over-polished / “too AI”** | Flux, Klein | `--naturalize`; lower CFG; `ANTI_AI_LOOK_NEGATIVE`, film grain |
| **Rigidity / no “creative accidents”** | Flux | `--naturalize` (grit, grain); vary seeds; training: diverse styles |
| **Poor emotion control** (smug, terrified → neutral) | Flux | Put emotion early; explicit phrases; see `EMOTION_PROMPT_TIPS` |
| **Catastrophic forgetting** (real-world objects wrong) | Illustrious, NoobAI | Train on mixed data; explicit object tags; domain prompts |
| **Quality tag dependency** (no masterpiece → 2022 look) | Anime models | Always use quality tags or `--boost-quality`; see `QUALITY_TAG_DEPENDENCY_TIPS` |
| **White dots / speckles / artifacts** | Illustrious, SDXL | `--anti-artifacts`; negative: `ARTIFACT_NEGATIVES`; “particle” in negative |
| **Flatness / no depth** | Anime base | Lighting tags; “dynamic lighting”, “depth”; lighting LoRAs |
| **Low seed variance** (same composition per prompt) | Z-Image, some Flux | `--creativity`; multiple seeds; vary prompt slightly; see `SEED_VARIANCE_TIPS` |
| **Vocabulary/language gaps** (Qwen, niche tags) | Z-Image | Common words; synonyms; see `VOCABULARY_TIPS` |
| **Character bleeding** (multi-char colors blend) | Z-Image, SDXL | Anti-blending in dataset; explicit “distinct”; `--anti-bleed`; see MODEL_WEAKNESSES §6 |
| **Stubborn watermarks** | Illustrious | `--strong-watermark`; negative: `WATERMARK_NEGATIVE_STRONG` |
| **V-pred “burn”** (CFG too high → burnt colors) | NoobAI v-pred | CFG 3–5.5; `--cfg-rescale 0.7`; see `CFG_BURN_TIPS` |
| **Background amnesia** (blur, impossible geometry) | NoobAI | Describe background explicitly; see `BACKGROUND_TIPS` |
| **Centering bias** (always middle, passport feel) | Klein | “Off-center”, “rule of thirds”; see `CENTERING_TIPS` |
| **“Flux mouth” / same mouth shape** | Flux + Person LoRAs | Vary prompts; different LoRA scales; diversity in training |
| **Loss of artistic “soul”** (too smooth, no grit) | Flux | `--naturalize`; film grain; rough/sketch in prompt; training: textured data |
| **Distant face meltdown** (smear beyond medium shot) | SDXL | Close-up/medium for faces; face restorer (ADetailer); see `DISTANT_FACE_TIPS` |
| **Spiky / pixel-stretch artifacts** | SDXL | `--anti-artifacts`; try different steps/seed; see `ARTIFACT_NEGATIVES` |
| **Resolution inflexibility** (double-head off buckets) | SDXL | Native resolutions; see `RESOLUTION_TIPS`; 1:1 or full-body cues: `FULL_BODY_AND_TWO_HEAD_TIPS` |
| **FLUX grid artifact** (grid in dark areas, with ControlNet/upscale) | FLUX | CFG ≤ 3.5, LoRA strength ≤ 1.20; avoid overtrained LoRAs; see `FLUX_GRID_ARTIFACT_TIPS` |
| **Full body not showing / two-head** | SDXL | Describe lower body: standing, legs, long dress, shoes; 1:1 for head shot; see `FULL_BODY_AND_TWO_HEAD_TIPS` |
| **Garbled faces** (not enough pixels) | SDXL | Hi-Res Fix, better VAE, face restorer, inpainting; see `GARBLED_FACE_TIPS` |
| **Negative prompt not working** | General | Be specific, avoid vague terms; test per checkpoint; see `NEGATIVE_PROMPT_BEST_PRACTICES` |
| **LoRA too strong / grid / overcooked** | FLUX, Civitai | Strength ~1.0, FLUX ≤ 1.20; see `LORA_STRENGTH_TIPS` |
| **Orange/green tint** (e.g. Civitai online) | Platform | Negative: `COLOR_TINT_NEGATIVE`; export PNG locally to avoid compression |
| **Compression artifacts** | Civitai generator | Generate locally; save PNG; avoid re-encoding |

---

### sample.py flags that map to these issues

| Flag | What it does |
|------|----------------|
| `--naturalize` | Anti-AI look: negative + natural-look prefix + film grain post-process |
| `--naturalize-grain 0.015` | Grain amount (0 = no grain) |
| `--anti-bleed` | Concept/color bleeding: distinct-colors positive + color-bleed negative |
| `--diversity` | Repetitive face: diversity positive + same-face negative |
| `--anti-artifacts` | White dots, speckles, spiky: append artifact negative |
| `--strong-watermark` | Stronger watermark/logo negative |
| `--boost-quality` | Prepend masterpiece, best quality |
| `--subject-first` | Reorder prompt (subject → age → height → …) |
| `--cfg-rescale 0.7` | Reduce oversaturation / burn (use with CFG 5–7) |
| `--cfg-scale 4.5` | Lower CFG for v-pred / burn-prone models |

---

### Config reference

All tips and negatives live in **config/prompt_domains.py**:

- **Concept bleeding:** `CONCEPT_BLEEDING_NEGATIVE`, `CONCEPT_BLEEDING_POSITIVE`
- **Artifacts:** `ARTIFACT_NEGATIVES`
- **Watermark:** `WATERMARK_NEGATIVE_STRONG`
- **Face diversity:** `FLUX_FACE_DIVERSITY_NEGATIVE`, `DIVERSITY_POSITIVE`
- **Anti-AI look:** `ANTI_AI_LOOK_NEGATIVE`, `NATURAL_LOOK_POSITIVE`
- **Tips (prompting):** `SPATIAL_AWARENESS_TIPS`, `EMOTION_PROMPT_TIPS`, `CFG_BURN_TIPS`, `BACKGROUND_TIPS`, `CENTERING_TIPS`, `DISTANT_FACE_TIPS`, `RESOLUTION_TIPS`, `SEED_VARIANCE_TIPS`, `VOCABULARY_TIPS`, `QUALITY_TAG_DEPENDENCY_TIPS`
- **Color tint:** `COLOR_TINT_NEGATIVE`
- **FLUX grid:** `FLUX_GRID_ARTIFACT_TIPS` (CFG ≤ 3.5, LoRA ≤ 1.20)
- **Negative prompts:** `NEGATIVE_PROMPT_BEST_PRACTICES`
- **Full body / two-head:** `FULL_BODY_AND_TWO_HEAD_TIPS`, `GARBLED_FACE_TIPS`
- **LoRA strength:** `LORA_STRENGTH_TIPS`
- **Prompt structure:** `PROMPT_STRUCTURE_TIPS` (Who/What/Where/When)

Training: **data/caption_utils.py** boosts `DOMAIN_TAGS["concept_bleed"]` (distinct colors, no color bleed) when present in captions. Use anti-blending for multiple characters (§6 in QUALITY.md).

---

## Part 2 — Model weaknesses and SDX mitigations

Based on common failures of Stable Diffusion, SDXL, FLUX, and similar models (2024–2025). We address these via **training data**, **caption boosting**, **recommended prompts/negatives**, and **documented workarounds**.

---

## 1. Hands and fingers

**What goes wrong:** Deformed hands, extra or fused fingers, impossible poses. Caused by low hand visibility in training data, anatomical complexity (many joints), and latent-space smoothing.

**How we prepare:**
- **Domain tag boosting:** Captions containing `hands`, `correct anatomy`, `five fingers`, `visible hands`, `natural pose` are boosted so the model learns them. See `data/caption_utils.py` → `DOMAIN_TAGS["anatomy"]`, `boost_domain_tags`.
- **Training data:** Include images with clear, well-captioned hands (e.g. “holding a cup”, “hands on keyboard”). Describe hands in relation to objects (object anchoring) and use simple poses (“hands clasped”, “arms crossed”).

**Fixes at inference:**
- Use **positive** tags: `correct hands`, `five fingers`, `natural hands`, `visible hands`.
- Use **negative:** `config/prompt_domains.py` → `ANATOMY_NEGATIVES`, or `data/caption_utils.py` → `NEGATIVE_ANATOMY`.
- **Prompt tips:** `config/prompt_domains.py` → `HAND_FIX_PROMPT_TIPS` (object anchoring, simple poses, avoid vague “no deformed hands”).

---

## 2. Faces and eyes

**What goes wrong:** Garbled or distorted faces, asymmetric or wrong eyes.

**How we prepare:**
- Quality and anatomy tags are boosted; include “detailed face”, “clear eyes”, “symmetrical” in training captions where relevant.

**Fixes at inference:**
- Negative: `data/caption_utils.py` → `NEGATIVE_FACE` (bad face, deformed face, distorted eyes, etc.).
- Use quality tags: `masterpiece`, `best quality`, `sharp focus`, `detailed`.

---

## 3. Full body and limbs

**What goes wrong:** Model ignores “full body”, crops limbs, or generates incomplete figures.

**How we prepare:**
- Boost anatomy tags: `full body`, `standing`, `legs`, `feet`, `visible hands`, `natural pose` (in `DOMAIN_TAGS["anatomy"]`).
- Training data: Explicitly caption full-body images with “full body”, “standing”, “legs”, “feet”, “shoes”.

**Fixes at inference:**
- Add “full body”, “standing”, “long dress”, “legs”, “shoes” (or similar) so the model fills the frame with one complete body.

---

## 4. Double head / portrait aspect ratio

**What goes wrong:** In portrait aspect (e.g. 9:16), models often output two heads or merged subjects (“two-head problem”).

**How we prepare:**
- Boost “single subject”, “no duplicate”, “clear composition” (`DOMAIN_TAGS["avoid_failures"]`).
- Train on multiple aspect ratios if possible (see `docs/IMPROVEMENTS.md` multi-resolution).

**Fixes at inference:**
- Use **1:1** for head/shoulder shots, or for portrait aspect add **full-body cues** so one body fills the frame: “standing”, “long dress”, “legs”, “shoes”. See `config/prompt_domains.py` → `PORTRAIT_ASPECT_TIPS`.

---

## 5. Text and spelling

**What goes wrong:** Unreadable or wrong text; models treat text as texture, not symbols.

**How we prepare:**
- Boost “text”, “lettering” when present; include clearly captioned images with readable text if you need text generation.
- **Practical approach:** Prefer generating images without text and adding typography in post (design software). Document this in your pipeline.

**Fixes at inference:**
- Negative: `data/caption_utils.py` → `NEGATIVE_TEXT` (garbled text, misspelled, watermark).
- For readable text, use short, simple words and quality tags; or add text in post.

---

## 6. Multiple subjects / wrong count / blending

**What goes wrong:** Wrong number of people/objects; characters blend into one.

**How we prepare:**
- **Anti-blending:** For multi-person phrases (`2girls`, `crowd`, etc.), the dataset adds positive tags (distinct characters, no blending) and negative (character blending, merged figures). See `data/caption_utils.py` → `add_anti_blending_and_count`.
- Boost “correct number of”, “distinct”, “multiple objects” (`DOMAIN_TAGS["avoid_failures"]`, `other_hard`).

**Fixes at inference:**
- Be explicit: “2 girls”, “three people”, “room full of people”. Use the same negative terms (blending, merged) if needed.

---

## 7. Composition and framing

**What goes wrong:** Cropped, out of frame, bad composition, cut-off body parts.

**How we prepare:**
- Boost “complex composition”, “depth of field”, “perspective”, “clear composition”, “well proportioned”.
- Negatives available: `NEGATIVE_COMPOSITION` (duplicate, merged subjects, cropped, cut off).

**Fixes at inference:**
- Use `config/prompt_domains.py` → `ANATOMY_NEGATIVES` or `data/caption_utils.py` → `NEGATIVE_COMPOSITION` in negative prompt.

---

## 8. Resolution and aspect ratio

**What goes wrong:** Quality drop at resolutions or aspect ratios the model wasn’t trained on.

**How we prepare:**
- Train at your target resolution (`--image-size`). For multiple aspect ratios, see `docs/IMPROVEMENTS.md` (multi-resolution / bucketing).

**Fixes at inference:**
- Use `--width` and `--height` that match or are close to training; upscale in post if needed.

---

## Quick reference: our tools

| Problem           | Training / prep                    | Inference / config reference                    |
|-------------------|------------------------------------|-------------------------------------------------|
| Hands             | Boost anatomy tags; object anchoring, simple poses | `ANATOMY_NEGATIVES`, `HAND_FIX_PROMPT_TIPS`, `NEGATIVE_ANATOMY` |
| Faces/eyes        | Quality + anatomy in captions      | `NEGATIVE_FACE`                                 |
| Full body         | “full body”, “standing”, “legs” in captions | Same tags in prompt                             |
| Double head       | “single subject”, “no duplicate”   | 1:1 or full-body cues; `PORTRAIT_ASPECT_TIPS`   |
| Text              | “text”, “lettering” boost; or add text in post | `NEGATIVE_TEXT`; add text in post recommended  |
| Multiple subjects | Anti-blending in dataset           | Explicit count; negative: blending, merged      |
| Composition       | “clear composition”, “well proportioned” | `NEGATIVE_COMPOSITION`, `ANATOMY_NEGATIVES`     |

**Code/config:**
- `config/defaults/ai_image_shortcomings.py` — taxonomy + `mitigation_fragments` for `sample.py --shortcomings-mitigation` and `train.py --train-shortcomings-mitigation` (photoreal, **digital painting / concept / pixel / vector / game textures**, 3D render, plus optional 2D-anime packs with `--shortcomings-2d` / `--train-shortcomings-2d`; see [QUALITY.md](QUALITY.md)).
- `config/defaults/art_mediums.py` — artist-first medium + anatomy/proportion guidance for `sample.py --art-guidance-mode --anatomy-guidance`, `train.py --train-art-guidance-mode --train-anatomy-guidance`, and book pipeline forwarding.
- `config/defaults/style_guidance.py` — style-domain + artist/game-name guidance for `sample.py --style-guidance-mode`, `train.py --train-style-guidance-mode`, normalize-captions tooling, and book pipeline forwarding.
- `data/caption_utils.py` — `DOMAIN_TAGS`, `boost_domain_tags`, `apply_shortcomings_to_caption_pair`, `NEGATIVE_ANATOMY`, `NEGATIVE_FACE`, `NEGATIVE_COMPOSITION`, `NEGATIVE_QUALITY`, `NEGATIVE_TEXT`, `NEGATIVE_ANATOMY_FULL`
- `config/prompt_domains.py` — `ANATOMY_NEGATIVES`, `HAND_FIX_PROMPT_TIPS`, `PORTRAIT_ASPECT_TIPS`, `RECOMMENDED_NEGATIVE_BY_DOMAIN`

---

## 9. Community-wide issues (SDXL, Flux, Illustrious, NoobAI, Z-Image)

Concept bleeding, plastic skin, repetitive faces, artifacts, watermark stubbornness, CFG burn, centering bias, distant face meltdown, resolution lock, seed variance, and vocabulary gaps are all documented with **mitigations** (prompt tips, negatives, and sample.py flags) in **[docs/QUALITY.md](QUALITY.md)** (*Community model issues*).

**sample.py flags that address these:**
- `--naturalize` — plastic/AI look (negative + natural prefix + film grain)
- `--anti-bleed` — color/object bleeding (distinct colors positive + bleed negative)
- `--diversity` — repetitive/default face (diversity positive + same-face negative)
- `--anti-artifacts` — white dots, speckles, spiky artifacts
- `--strong-watermark` — stubborn logos/watermarks
- `--boost-quality` — prompt adherence; `--subject-first` — tag order
- `--cfg-rescale 0.7`, lower `--cfg-scale` — v-pred burn, oversaturation

**Config:** All related negatives and tip lists are in `config/prompt_domains.py` (e.g. `CONCEPT_BLEEDING_NEGATIVE`, `ARTIFACT_NEGATIVES`, `EMOTION_PROMPT_TIPS`, `SPATIAL_AWARENESS_TIPS`). See [QUALITY.md](QUALITY.md) (*Community model issues*) for the full issue → mitigation table.

---

## 10. Gaps: what frontier models still struggle with — and what SDX does **not** fully solve

This section is the honest complement to §1–9 and [QUALITY.md](QUALITY.md): **community mitigations** (prompts, negatives, flags) are not the same as a **guaranteed fix**. Below, **“no in-repo fix”** means there is no integrated module that *enforces* correctness; you rely on training data, luck, or external tools.

### A. Still fundamentally hard (diffusion + CLIP/T5 blind spots)

| Problem | Why models fail | What SDX has | What we **don’t** have |
| :--- | :--- | :--- | :--- |
| **Arbitrary spelled text** | Text is learned as texture; long strings are unstable. | OCR-guided **repair** when you know the target string (`sample.py` / `generate_book.py` + `utils/generation/text_rendering.py`); lettering negatives; book pipeline expected text. | General “render this exact paragraph” without **known** expected text; no diffusers-style dedicated text renderer layer in-core. |
| **Exact counts** (“exactly 7 coins”) | No discrete counter in the denoiser. | Dataset-side `add_anti_blending_and_count`, explicit prompt counts, negatives for merging, and inference pick-best `--pick-best combo_count` (+ `--expected-count`, `--expected-count-target`, `--expected-count-object`) as lightweight people/simple-object count verification. | No hard cardinality guarantee for arbitrary object classes; still heuristic without constrained decoding/segmentation. |
| **Fine spatial logic** (“left hand holds blue cup, right waves”) | Conditioning is global; relations are fuzzy. | `SPATIAL_AWARENESS_TIPS`, `--subject-first`, early prompt ordering. | Scene-graph / layout-conditioned attention (ControlNet-class conditioning is only partially exposed via `control_cond_dim` on `DiT_Text` — not a full layout stack). |
| **Physical plausibility** | No simulation; only statistics of pixels. | Tips, pick-best metrics (`utils/quality/test_time_pick.py`), orchestration hooks. | Physics engine, 3D consistency, reflection law enforcement ([`architecture_map`](../utils/architecture/architecture_map.py): `physical_grounding` = not in repo). |
| **Identity across unrelated images** | No persistent memory per user. | `--character-sheet`, **`--reference-image`** + CLIP → extra cross-attn tokens (`DiT_Text`), book anchors, `consistency_helpers`. | Strong identity still needs **trained** `--reference-adapter-pt` or LoRA; default projector is randomly initialized. |
| **Video / temporal coherence** | Out of scope for single-image DiT. | — | Native video diffusion or frame consistency training. |

### B. Partially mitigated (prompts + data — not architectural guarantees)

| Problem | SDX mitigation | Remaining gap |
| :--- | :--- | :--- |
| **Hands / anatomy** | `DOMAIN_TAGS`, `NEGATIVE_ANATOMY`, `HAND_FIX_PROMPT_TIPS` | No **auxiliary anatomy loss** or hand keypoint head on the denoiser ([`auxiliary_structure_supervision`](../utils/architecture/architecture_map.py)). |
| **Distant / small faces** | `DISTANT_FACE_TIPS`, `GARBLED_FACE_TIPS` in `config/prompt_domains.py` | In `sample.py`: **`--face-enhance`** (Haar + local sharpen/contrast), **`--face-restore-shell`** for external GFPGAN/ADetailer CLIs; not a full in-repo face restoration model. |
| **Composition / cropping** | `NEGATIVE_COMPOSITION`, quality tags, inference `--resize-mode center_crop|saliency_crop` (+ `--resize-saliency-face-bias`) | Saliency crop is heuristic (not semantic segmentation/subject detector); complex multi-subject scenes may still need manual framing. |
| **Style / concept bleeding** | `--anti-bleed`, dataset boosts | No token-level cross-attention steering (SAG / per-token scale) — see [IMPROVEMENTS.md](IMPROVEMENTS.md) §2.2. |

### C. Training / inference tooling gaps (listed elsewhere, summarized)

These are called out in [IMPROVEMENTS.md](IMPROVEMENTS.md) and [DIFFUSION_LEVERAGE_ROADMAP.md](DIFFUSION_LEVERAGE_ROADMAP.md):

- **Few-step distilled student** (consistency / DMD) — no trainer in repo.  
- **Full flow-matching / rectified-flow training** — not drop-in for current `GaussianDiffusion` ([MODERN_DIFFUSION.md](MODERN_DIFFUSION.md)).  
- **Extra samplers** (DPM++ / UniPC as first-class flags) — DDIM/Euler-style path is primary; more schedulers are “add” items.  
- **Inpaint-aware *training*** (random latent masks) — not wired; inpainting is inference/workflow (`sample.py`, book pipeline).  
- **DDP + resolution buckets** — buckets exist; multi-GPU + buckets is constrained.  
- **WebDataset / giant-scale streaming** — optional future in IMPROVEMENTS.  
- **Live web retrieval** — facts via `utils/prompt/rag_prompt.py` only; no built-in crawl.

### D. Where to invest if you want to *close* gaps (not just document them)

1. **Text:** double down on OCR + expected-text workflows **or** train with readable text in-domain; consider external typography for production UI.  
2. **Structure:** auxiliary heads or ControlNet-style conditioning (see [DIFFUSION_LEVERAGE_ROADMAP.md](DIFFUSION_LEVERAGE_ROADMAP.md) §3–4).  
3. **Identity:** reference-image adapter or stronger character-sheet conditioning in `DiT_Text`.  
4. **Inference control:** SAG / cross-attn hooks in `sample.py` ([IMPROVEMENTS.md](IMPROVEMENTS.md) §2.2).  
5. **Faces at distance:** optional post-pass script wrapping GFPGAN/CodeFormer or face-region inpaint — **bridge** the doc↔code gap ADetailer users expect.

When you ship a new fix, update **§1–9** or [QUALITY.md](QUALITY.md) for users, and **`architecture_map.py`** for theme status.

---

## Part 3 — General AI image failure modes

A reference guide to typical failure modes in current image diffusion and generative models—useful for training goals, caption design, and evaluation. For SDX-specific mitigations and config hooks, see [QUALITY.md](QUALITY.md) and [QUALITY.md](QUALITY.md).

---

## Photorealism and general image quality

### Surface detail, skin, and tangents

Models often excel at basic color theory and broad object recognition (shapes, designs, characteristics of nouns) but miss finer cues: realistic skin texture (pores, subtle freckles, veins, micro-imperfections, natural translucency), which can read as plastic or overly smooth.

**Tangents** are a persistent composition problem: insufficient overlap or separation between objects hurts clarity. Elements may meet awkwardly at edges, flattening depth and causing confusing mergers instead of deliberate overlaps or clean separations that guide the eye and reinforce 3D space.

### Spatial relationships, support, and interaction

Models lack a reliable sense of **weight, contact, and support**. Figures can appear to hover rather than sit or lean; chairs show no compression under load; clothing may not bunch, crease, or drape believably against surfaces. Hands often “float” near objects instead of gripping with plausible tension. Feet may not plant on ground planes; object–object relationships can feel weightless or disconnected.

### Lighting, global illumination, and environmental cohesion

Beyond simple shading, models struggle with **bounce light and color bleeding** (e.g. a red shirt subtly tinting a nearby white wall). Elements are often treated as isolated assets rather than parts of one lit volume. **Shadows** may disagree with a single dominant light direction, breaking immersion in complex scenes. In 3D-leaning renders, weak or missing **ambient occlusion** (contact shadows in crevices) can make objects feel ungrounded.

### Line work, brushwork, and edges (where visible)

Where line or edge structure matters, outputs can show **uniform line weight** and over-defined contours instead of dynamic thickness, tapering, and “lost and found” edges that dissolve into shadow—contributing to a hyper-processed or “deep-fried” look compared to intentional traditional work.

### Anatomy beyond the “hand problem”

The classic hand issue extends to **deeper structure**: impossible limb bends (“noodle limbs”), dubious elbows and knees, and weak attachment logic (e.g. neck–shoulder–trapezius transitions). Missing skin detail further undermines realism even when pose is roughly correct.

### Composition, visual flow, and design intent

Strong **center bias** and **horror vacui** (filling every corner) are common. Models underuse negative space, rule-of-thirds balance, and background **leading lines** that steer narrative focus. The result can lack deliberate rest and guided flow.

### Materials, texture logic, and aging

Surface labels (metal, fabric, skin) may be recognized while **placement logic** fails: rust or grime in random patches rather than moisture traps and wear patterns; weathering without story. **Subsurface scattering** (light through skin, wax, marble) is often weak or absent, so subjects read as painted plastic or clay. Folds, hair strands, and liquids may ignore gravity, wind, and interaction.

### Narrative consistency and functional logic

Scenes can look polished but **ahistorical**: generic scratches on armor instead of plausible battle damage; hems without mud or wear from terrain. Architecture and machinery may look impressive but **non-functional** (pipes to nowhere, impossible stairs). That weakens the sense of a lived-in world.

### Perspective, foreshortening, and depth

Standard views often work better than **strong foreshortening**: limbs or objects thrust toward the camera may lose scale or warp; deep stacks of overlap can merge instead of keeping readable silhouettes and the “coming at you” clarity artists build with overlap and scale.

### Facial nuance and emotion

Broad categories (happy, sad, angry) are easier than **micro-expressions** (a sarcastic lip quirk, Duchenne eye crinkle, coordinated muscle groups). Eyes may default to a flat stare; faces can be **over-symmetrical**, pushing uncanny valley. Natural facial asymmetry is often missing.

### Color grading, value, and discipline

Defaults often skew **high-contrast, high-saturation** (“rainbow” palettes). **Limited palettes** (e.g. Zorn-style restraint) are harder to hold; stray hues or noise can break mood. In grayscale, **value structure** may be chaotic if color is treated as a substitute for designed light/shadow rather than a separate compositional tool.

### Additional gaps (often overlooked)

- **Legible text and typography** — distorted letters, wrong spelling, unstable layout.
- **Small repeating structure** — buttons, patterns, stitches inconsistent across the image.
- **Fluids and particles** — splashes, smoke, motion fabric that feels stiff.
- **Multi-subject scenes** — background figures with inconsistent faces or proportions.
- **Complex prompts** — secondary elements drifting from intent.

---

## Digital art, screen-native, and hybrid workflows

**Raster painting (Photoshop, Procreate, Clip Studio, etc.)** — Models often default to overly smooth, airbrushed, or “AI-blended” surfaces instead of believable **brush economy** (hard vs soft edges, stroke direction, intentional texture). Midtones turn to mud; edges lose intention.

**Concept art, matte painting, photobash** — Weak **perspective and lighting unity** across collaged or painted regions; cutout look, scale drift, or conflicting color cast. Design reads as generic “grey sculpt” rather than a clear focal idea.

**Pixel art and retro game graphics** — **Subpixel blur**, inappropriate gradients, and inconsistent **pixel scale** break the medium. Good pixel work needs crisp tiling, palette discipline, and deliberate dither or AA.

**Vector, flat design, icons, UI illustration** — Wobbly paths, inconsistent stroke rules, and accidental photoreal or 3D leakage. Icons need clean silhouettes and consistent geometric discipline.

**Hand-painted game assets / stylized 3D** — **Albedo vs lighting** confusion, muddy texture paint, and **texel density** that jumps across UV islands. Reads as noise mud instead of directed hand-painted direction.

These map to mitigation ids `digital_painting`, `concept_matte_digital`, `pixel_digital`, `vector_flat_digital`, `stylized_game_digital` in `config/defaults/ai_image_shortcomings.py` (included in `auto` when keywords match, and in `all` with other non–2D-anime packs).

---

## Stylized 2D (anime, manga, cartoons, comics, cel-shade, watercolor, graphic novel)

Many failures trace back to the same roots—**no true physics**, **pattern completion over intent**, **data bias**—but stylized work breaks in ways tuned to **convention and exaggeration**, not photographic accuracy.

### Style consistency and drift

**Style drift** within one image or across a set: e.g. “anime” mixing cel-shading with painterly patches, or Western proportions with anime eyes. **Character consistency** suffers (hair tips, eye highlights, folds, facial structure varying between generations)—painful for sheets, comics, and animation keys. Popular aesthetics may dominate while rarer strip or indie looks stay hard to hit.

### Line art and edges in 2D

Intentional **variable weight**, tapering, silhouette clarity, and selective lost edges are often replaced by uniform, crisp, or extraneous outlines—mechanical rather than hand-drawn. In anime/manga this shows up as messy hair strands, awkward garment contours, or lines that lack energy.

### Shading, light, and cel logic

Stylized 2D uses **simplified, often hard-edged** shadow design. AI may contradict a single light, mix soft gradients into flat-color expectations, or place “SSS-like” plastic sheen where the style calls for flat reads. Stylized bounce/balance, when present, may be inconsistent; **palette discipline** is easy to break with sneaked extra hues or noise.

### Anatomy, proportion, and exaggeration

Even with non-real proportions, **rules must stay consistent**: eye size/highlight conventions, head–body ratios, limb taper. Errors feel louder because stylization amplifies them. Noodle limbs, floating hands, and muddy overlap in dynamic or foreshortened poses remain common.

### Composition, negative space, and storytelling

Center bias and horror vacui hurt **panel-like** or **illustration-first** layouts that rely on breathing room and leading lines. Faces may fall back to exaggerated emotion masks without subtle asymmetry or coordinated cues that sell “alive” cartoon acting.

### Texture, medium, and material in 2D

Medium simulation (watercolor bleed, paper tooth, ink wash, flat vectors, halftones) often becomes **uniform smoothness** or **random busy texture**. Wear and dirt lack narrative placement; motion effects (hair, cartoon water) can feel stiff.

### Extra challenges specific to 2D

- **Prompt adherence** — requests for children’s book or classic comic looks may veer toward photorealism or glossy CGI.
- **Multi-character / sequential work** — inconsistency compounds.
- **Text** — still weak for bubbles, titles, and signage.
- **Over-polished kitsch** — airbrushed, generically vibrant outputs vs. raw hand-drawn energy.

---

## Technical / 3D-render aesthetics (when the target looks like CG)

When the desired look is **high-end 3D**, models may invent **melting or fused geometry**, bad **UV logic** (stretched or swimming patterns), and inconsistent **focal length / camera grammar** within one frame—fine at thumbnail scale, weak under scrutiny.

---

## Root causes (summary)

Limitations cluster around: **no grounded physical simulation**; **correlational pixel/statistical modeling** rather than explicit artistic decisions; **training distribution and bias**; and **weak long-horizon consistency** for text, counts, and spatial relations. Mitigation in practice combines **better data and captions**, **prompt and negative design**, **inference tooling** (where the codebase provides it), and **post-production**—not a single architectural switch.

---

## In this repo

- **Registry and keyword detection:** `config/defaults/ai_image_shortcomings.py` (`config.ai_image_shortcomings`).
- **Artist-first medium guidance:** `config/defaults/art_mediums.py` (`config.art_mediums`) for traditional, digital, photography, and anatomy/proportion packs.
- **Style-domain + artist/game guidance:** `config/defaults/style_guidance.py` (`config.style_guidance`) for anime/comic/editorial/concept/game/photo language + artist/game-name stabilization cues.
- **Sampling:** `python sample.py ... --shortcomings-mitigation auto` (match prompts to categories) or `all` (full photoreal pack; add `--shortcomings-2d` for stylized 2D packs).
- **Sampling (medium/anatomy):** `python sample.py ... --art-guidance-mode auto|all --anatomy-guidance lite|strong` (optional `--no-art-guidance-photography`).
- **Sampling (style domains):** `python sample.py ... --style-guidance-mode auto|all` (optional `--no-style-guidance-artists`).
- **Inference framing:** `python sample.py ... --resize-mode center_crop|saliency_crop` (optional `--resize-saliency-face-bias`) to reduce stretched/non-semantic framing when target aspect differs.
- **Training:** `python train.py ... --train-shortcomings-mitigation auto|all --train-art-guidance-mode auto|all` with optional `--train-shortcomings-2d`, `--train-anatomy-guidance`, `--no-train-art-guidance-photography`.
- **Training (style domains):** `python train.py ... --train-style-guidance-mode auto|all` (optional `--no-train-style-guidance-artists`).
- **Offline manifests:** `python -m scripts.tools normalize_captions ... --shortcomings-mitigation auto --art-guidance-mode auto --style-guidance-mode auto` (plus optional 2D/anatomy toggles).