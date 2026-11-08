# Quality guide

Merged: sampling fixes, community issues, model weaknesses, and failure-mode reference.

---



---

## Part 1 — Sampling fixes and community issues

Sampling fixes (Civitai-style) and a **community issue matrix** (SDXL, Flux, Z-Image, etc.) with sample.py flags.

---

## Civitai-style quality tips

Based on common problems reported in Civitai checkpoint comments and SD community guides (oversaturation, blur, bad hands, wrong resolution). Use these with `sample.py` and training to get more reliable, high-quality output.

---

### 1. Oversaturation / “burned” colors (high CFG)

**Problem:** Images look overexposed or oversaturated when using strong guidance.

**Fixes:**

- **`--cfg-rescale 0.7`** — CFG rescale (Lin et al. 2024): matches the guided x0's per-sample std to the conditional x0, reducing overexposure at `--cfg-scale` 7–10. At lower CFG prefer `--apg-parallel-eta 0` and `--fdg-low 0.6 --fdg-high 1.3`.
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

- **Stay near the native pixel area** — `--width` / `--height` set the sampling grid (any aspect ratio, snapped to whole patches). Sizes beyond 2× the native area are composed at the native area and enlarged after decode, which is soft; add `--hires-fix` to render that detail, and `--vae-tiling` for large decodes.
- **More steps** — 25–35 steps often improve clarity; we default to 50.
- **Post-process** — `--sharpen 0.3` and `--contrast 1.05` can help (see `utils/quality/finishing.py`).
- **VAE** — Train/sample with a good VAE (e.g. sd-vae-ft-mse or sdxl-vae); avoid heavily quantized VAEs for final decode.

---

### 3. Bad hands / anatomy

**Problem:** Extra fingers, fused fingers, deformed hands or limbs.

**Fixes:**

- **Default negative prompt** — If you leave `--negative-prompt` empty, we use a Civitai-style default: `low quality, worst quality, blurry, bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, deformed, duplicate`. Override with `--negative-prompt "..."` if your model prefers minimal negatives (e.g. some SDXL-style models).
- **Positive cues** — In the prompt, describe hands in context: e.g. “holding a cup”, “hands on keyboard”, “correct hands, five fingers”.
- **Simpler poses** — “Arms crossed”, “hands in pockets” tend to work better than complex hand poses.
- **Inpainting** — For a single image, use `--init-image` + `--mask` to inpaint only the hand/face area at 0.4–0.5 strength.


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

**Training:** Include many examples with correct text in images and captions that describe the exact text (e.g. “sign that says OPEN”). The data pipeline boosts “text_in_image” domain tags (see `data/caption_utils.py`).

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


---

### 8. Challenging content (NSFW, surreal, abstract, weird)
**Problem:** Model under-adheres or produces generic output for mature, surreal, or unusual prompts.

**Fixes:**
- **Do not censor** — Use clear, descriptive tags in training data and at inference; the model learns what you show.
- **Quality tags help** — Add "masterpiece, best quality, detailed, sharp focus" so any challenging prompt gets stronger conditioning.
- **Be concrete** — For surreal/abstract, describe colors, layout, and mood; put subject first, then setting, then style.
- **`--boost-quality`** — Use for maximum adherence on challenging prompts.
- **Training:** Include diverse examples with consistent tags; the data pipeline boosts "complex" and "challenging" domain tags (see `data/caption_utils.py`).

---

### 9. Hard styles (3D, photorealistic, style mixes) and LoRA mixing

**Problem:** Many models blur 3D vs 2D, realistic vs illustrated, and mixed styles (2.5D, semi-realistic, photorealistic anime). Combining multiple LoRAs (e.g. style + character) can give muddy or oversaturated results.

**Fixes:**

- **`--hard-style`** — Use `--hard-style 3d`, `realistic`, `3d_realistic`, or `style_mix` to prepend recommended tags so the model anchors on the right look. Add a matching `--negative-prompt` (e.g. for 3d: "flat, 2d, blurry, bad proportions").
- **Training:** Captions that contain 3D, photorealistic, or style-mix phrases (e.g. "2.5d", "semi-realistic", "photorealistic anime") are boosted **first** in the pipeline (`data/caption_utils.py` → `HARD_STYLE_TAGS`, `boost_hard_style_tags`) so the model learns these hard styles well.
- **Style mixing:** Put the dominant style first in the prompt; use explicit phrases like "2.5d", "semi-realistic", "photorealistic anime".
- **Multiple LoRAs:** Use lower scales (0.5–0.6 each) so they blend; put the dominant LoRA first and its trigger at the start of the prompt. If output is muddy, try `--cfg-scale 5` or `--cfg-rescale 0.7`.

See [docs/PROMPT_STACK.md](PROMPT_STACK.md#generating-3d-realistic-interiorexterior-and-other-hard-domains) for full hard-style prompts and negatives.

---

### 10. Large output sizes

`sample.py` samples on the requested aspect ratio directly. When `--width` × `--height` is more than 2× the native area it composes at the native area, and prints a note if the decode is then enlarged. For best quality:

- Use `--hires-fix`: compose at the native area, then upscale the latent and re-denoise at the full size (the start noise is SD3-shifted for the larger grid), and
- `--vae-tiling` for large decodes.

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
| Challenging (NSFW, surreal) | No censorship; add quality tags; use `--boost-quality` |
| Hard styles (3D, realistic, 2.5D) | `--hard-style 3d | realistic | 3d_realistic | style_mix`; add a matching negative |
| Multiple LoRAs / style mix | Lower LoRA scales (0.5–0.6); dominant LoRA first; `--cfg-rescale 0.7` if muddy |
| Large output OOM   | `--vae-tiling` |
| Softer / more natural | `--cfg-scale 5` |
| Less AI-looking / plastic | `--naturalize` (negative + prompt prefix + medium-routed capture finish); pick the camera with `--camera-isp` |
| Small artifacts polish / lost texture | `--refine restart` (Restart sampling; tune `--refine-t`, `--restart-cycles`) or `--solver dpmpp_2m_sde` |
| Color/concept bleeding (SDXL) | `--anti-bleed`; put spatial relations early; see [Community model issues](#community-model-issues) |
| Repetitive / same face (Flux) | `--diversity`; `DIVERSITY_POSITIVE` |
| Less templated / more novel | `--originality 0.3-0.8` (injects unique composition tokens + tunes creativity/CFG) |
| White dots / speckles / spiky artifacts | `--anti-artifacts`; “particle” in negative |
| Stubborn watermarks (e.g. Illustrious) | `--strong-watermark` |
| V-pred burn (NoobAI) | `--cfg-scale 4.5`, `--cfg-rescale 0.7` |
| FLUX grid artifact | CFG ≤ 3.5, LoRA strength ≤ 1.20 |
| Full body / two-head | Add standing, legs, long dress, shoes; 1:1 for head shot |
| Negative prompt ineffective | Be specific (e.g. extra fingers), not vague |

---

### 12. FLUX grid artifact and full-body / two-head (community fixes)

**FLUX grid:** Visible grid pattern in dark areas (worse with Depth/Canny and upscaling). From [FLUX GitHub #406](https://github.com/black-forest-labs/flux/issues/406): keep **CFG at 3.5 or lower** and **LoRA strength at or below 1.20**; avoid overtrained LoRAs.

**Full body / two-head:** From [Stable Diffusion Art](https://stable-diffusion-art.com/common-problems-in-ai-images-and-how-to-fix-them/): (1) For full body, add **standing, long dress, legs, shoes** (describe what should appear) rather than only “full body portrait”. (2) Use **1:1** for head/shoulder shots to avoid two-head; use portrait aspect for full-body so one body fills the frame. (3) Garbled faces often need more pixels: use close-up, Hi-Res Fix, better VAE, or face restorer/inpainting.

---

### 13. Making images look less AI-generated

**Problem:** Outputs look too smooth, plastic, waxy, or “obviously generated.”

**Fixes:**

- **`--naturalize`** — One flag that: (1) adds an **anti-AI-look negative** (plastic skin, oversmooth, airbrushed, waxy, doll-like, synthetic, CGI, uncanny, etc.) so the model is steered away from those traits; (2) prepends **natural-look positive** hints (natural skin texture, subtle imperfections, raw photo, natural lighting) to the prompt; (3) after best-of-N selection, runs the **capture finish** (`utils/quality/finishing.py`): photos go through a linear-light lens (cos^4 vignette, radial lateral CA, optical blur), Poisson-Gaussian sensor noise on a Bayer mosaic with Malvar-He-Cutler demosaic, camera tone curve, sharpening and JPEG 4:2:0; film prompts get density-space emulsion grain instead; anime, illustration, painting and 3D get no photographic texture.
- **`--naturalize-grain 0`** — prompt/negative changes only, no capture finish (any positive value keeps it on; noise comes from the camera/film model).
- **`--camera-isp auto|none|phone|dslr|compact|film`**, **`--camera-iso`**, **`--camera-jpeg-quality`** — force or tune the capture finish; **`--save-16bit`** writes 16-bit PNG from the float pipeline (8-bit output is TPDF-dithered).
- **`--num 4 --pick-best combo_natural`** (or `combo_art`) — rank candidates *before* the capture finish with the critics in `utils/quality/critics.py` (decoder-grid residual, noise-floor shape, 1/f² spectral slope, oversmoothing, gamut-hugging chroma, HDR halos, mirror symmetry) and `utils/quality/critics.py` (saliency composition, horizon tilt, subject cropping, exposure / key, local contrast, hue-template harmony). Every metric aggregates by weighted Borda count, and the critics read the prompt's intent: a requested teal-orange or other grade, low-/high-key exposure, centred or symmetric layout, or dutch angle is never penalised. `superior_composite` also adds any HF reward model (CAFE, PickScore, HPS, ImageReward) whose weights are on disk.
- **Lower CFG** — Try `--cfg-scale 5` or `6` for a less “pushed” look.
- **Training:** Include diverse, non-perfect references and captions that mention “natural skin texture,” “film grain,” “imperfections,” so the model learns a less polished look. You can add `ANTI_AI_LOOK_NEGATIVE`-style terms to your training negatives (see `config/prompts.py`).

---

### 14. Training-side tips (fewer issues at inference)

- **Negative prompt in data** — Include quality/anatomy negatives in captions or the dataset so the model learns to avoid them (`train.py --train-shortcomings-mitigation auto` or `normalize_captions --shortcomings-mitigation auto`).
- **Min-SNR loss** — `--min-snr-gamma 5` (or similar) stabilizes training and can improve clarity; the weight follows `--prediction-type` (Hang et al. 2023), so eps, v and x0 runs optimise the same objective.
- **Full tonal range** — `--zero-terminal-snr` (with `--prediction-type v`) removes the signal leak at `t = T` so the model can make true blacks, night scenes and bright high-key frames instead of everything drifting to mid-grey; `--noise-offset 0.05` (per channel) is the lighter alternative.
- **Framing** — `--resolution-buckets` assigns each image the bucket nearest its aspect ratio, so portraits keep heads and feet instead of being cropped into squares.
- **Flow runs** — `--flow-matching-training --timestep-sample-mode logit_normal --flow-shift S` (SD3 recipe: about 3 for 1024 px training, 1 to 2 at 256 to 512 px); the shift is saved with the checkpoint and scaled per bucket the same way at sampling time.
- **Native resolution** — Train at the resolution you plan to use for inference to avoid resolution mismatch blur.

These tips align with common Civitai and A1111/ComfyUI recommendations; adjust per checkpoint if a model card suggests different settings.

---

## Community model issues

Community-reported problems (SDXL, Flux/Klein, Illustrious/NoobAI, Z-Image, Civitai) and the mitigations available in SDX: **config/prompts.py** negatives, **sample.py** flags, and **training** practices.

**Sources:** Reddit r/StableDiffusion, Civitai articles & comments, FLUX GitHub (#406 grid artifact), [Stable Diffusion Art](https://stable-diffusion-art.com/common-problems-in-ai-images-and-how-to-fix-them/), ComfyUI prompt-engineering docs, Black Forest Labs FLUX.

---

### Quick reference: issue → mitigation

| Issue | Models | Mitigation |
|-------|--------|------------|
| **Concept/color bleeding** (red shirt + blue pants → purple) | SDXL | `--anti-bleed`; prompt: `CONCEPT_BLEEDING_POSITIVE`; negative: `CONCEPT_BLEEDING_NEGATIVE`; training: boost `DOMAIN_TAGS["concept_bleed"]` |
| **Poor spatial awareness** (behind, next to, under wrong) | SDXL | Put spatial phrases early |
| **Plastic/waxy skin** | SDXL, base Flux | `--naturalize`; `ANTI_AI_LOOK_NEGATIVE`, `NATURAL_LOOK_POSITIVE` |
| **Inconsistent prompt following** (ignores end of long prompt) | SDXL | Put key details first; `--subject-first`; `--boost-quality`; keep &lt;250 tokens |
| **Repetitive / “default” face** (Flux face) | Flux, Klein | `--diversity`; negative: `FLUX_FACE_DIVERSITY_NEGATIVE`; positive: `DIVERSITY_POSITIVE` |
| **Over-polished / “too AI”** | Flux, Klein | `--naturalize`; lower CFG; `ANTI_AI_LOOK_NEGATIVE`, film grain |
| **Rigidity / no “creative accidents”** | Flux | `--naturalize` (grit, grain); vary seeds; training: diverse styles |
| **Poor emotion control** (smug, terrified → neutral) | Flux | Put emotion early; explicit phrases |
| **Catastrophic forgetting** (real-world objects wrong) | Illustrious, NoobAI | Train on mixed data; explicit object tags; domain prompts |
| **Quality tag dependency** (no masterpiece → 2022 look) | Anime models | Always use quality tags or `--boost-quality` |
| **White dots / speckles / artifacts** | Illustrious, SDXL | `--anti-artifacts`; negative: `ARTIFACT_NEGATIVES`; “particle” in negative |
| **Flatness / no depth** | Anime base | Lighting tags; “dynamic lighting”, “depth”; lighting LoRAs |
| **Low seed variance** (same composition per prompt) | Z-Image, some Flux | `--creativity`; multiple seeds; vary prompt slightly |
| **Vocabulary/language gaps** (Qwen, niche tags) | Z-Image | Common words; synonyms |
| **Character bleeding** (multi-char colors blend) | Z-Image, SDXL | Anti-blending in dataset; explicit “distinct”; `--anti-bleed`; see MODEL_WEAKNESSES §6 |
| **Stubborn watermarks** | Illustrious | `--strong-watermark`; negative: `WATERMARK_NEGATIVE_STRONG` |
| **V-pred “burn”** (CFG too high → burnt colors) | NoobAI v-pred | CFG 3–5.5; `--cfg-rescale 0.7` |
| **Background amnesia** (blur, impossible geometry) | NoobAI | Describe background explicitly |
| **Centering bias** (always middle, passport feel) | Klein | “Off-center”, “rule of thirds” |
| **“Flux mouth” / same mouth shape** | Flux + Person LoRAs | Vary prompts; different LoRA scales; diversity in training |
| **Loss of artistic “soul”** (too smooth, no grit) | Flux | `--naturalize`; film grain; rough/sketch in prompt; training: textured data |
| **Distant face meltdown** (smear beyond medium shot) | SDXL | Close-up/medium for faces; face restorer (ADetailer) |
| **Spiky / pixel-stretch artifacts** | SDXL | `--anti-artifacts`; try different steps/seed; see `ARTIFACT_NEGATIVES` |
| **Resolution inflexibility** (double-head off buckets) | SDXL | Native resolutions; 1:1 or full-body cues |
| **FLUX grid artifact** (grid in dark areas, with ControlNet/upscale) | FLUX | CFG ≤ 3.5, LoRA strength ≤ 1.20; avoid overtrained LoRAs |
| **Full body not showing / two-head** | SDXL | Describe lower body: standing, legs, long dress, shoes; 1:1 for head shot |
| **Garbled faces** (not enough pixels) | SDXL | Hi-Res Fix, better VAE, face restorer, inpainting |
| **Negative prompt not working** | General | Be specific, avoid vague terms; test per checkpoint |
| **LoRA too strong / grid / overcooked** | FLUX, Civitai | Strength ~1.0, FLUX ≤ 1.20 |
| **Orange/green tint** (e.g. Civitai online) | Platform | Negative: "orange tint, green tint, color cast, wrong white balance"; export PNG locally to avoid compression |
| **Compression artifacts** | Civitai generator | Generate locally; save PNG; avoid re-encoding |

---

### sample.py flags that map to these issues

| Flag | What it does |
|------|----------------|
| `--naturalize` | Anti-AI look: negative + natural-look prefix + medium-routed capture finish |
| `--naturalize-grain 0` | Keep the prompt changes, skip the capture finish |
| `--camera-isp` | Capture finish: `auto` (by medium), `none`, `phone`, `dslr`, `compact`, `film` |
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

The negatives live in **config/prompts.py**:

- **Concept bleeding:** `CONCEPT_BLEEDING_NEGATIVE`, `CONCEPT_BLEEDING_POSITIVE`
- **Artifacts:** `ARTIFACT_NEGATIVES`
- **Watermark:** `WATERMARK_NEGATIVE_STRONG`
- **Face diversity:** `FLUX_FACE_DIVERSITY_NEGATIVE`, `DIVERSITY_POSITIVE`
- **Anti-AI look:** `ANTI_AI_LOOK_NEGATIVE`, `NATURAL_LOOK_POSITIVE`

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
- Negatives: `--shortcomings-mitigation auto` adds the anatomy pack from `config/prompts.py`.
- **Prompt tips:** object anchoring, simple poses, avoid vague “no deformed hands”.

---

## 2. Faces and eyes

**What goes wrong:** Garbled or distorted faces, asymmetric or wrong eyes.

**How we prepare:**
- Quality and anatomy tags are boosted; include “detailed face”, “clear eyes”, “symmetrical” in training captions where relevant.

**Fixes at inference:**
- Negatives: `--shortcomings-mitigation auto` (face/eye pack).
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
- Train on multiple aspect ratios if possible (see `docs/RESEARCH.md` multi-resolution).

**Fixes at inference:**
- Use **1:1** for head/shoulder shots, or for portrait aspect add **full-body cues** so one body fills the frame: “standing”, “long dress”, “legs”, “shoes”.

---

## 5. Text and spelling

**What goes wrong:** Unreadable or wrong text; models treat text as texture, not symbols.

**How we prepare:**
- Boost “text”, “lettering” when present; include clearly captioned images with readable text if you need text generation.
- **Practical approach:** Prefer generating images without text and adding typography in post (design software). Document this in your pipeline.

**Fixes at inference:**
- Negatives: the default negative avoids garbled text and watermarks unless `--text-in-image` is set.
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
- Negatives: `--shortcomings-mitigation auto` (duplicate / merged subjects / cut-off pack).

**Fixes at inference:**
- Use `--shortcomings-mitigation auto` for the anatomy and composition negative packs.

---

## 8. Resolution and aspect ratio

**What goes wrong:** Quality drop at resolutions or aspect ratios the model wasn’t trained on.

**How we prepare:**
- Train at your target resolution (`--image-size`). For multiple aspect ratios, see `docs/RESEARCH.md` (multi-resolution / bucketing).

**Fixes at inference:**
- `--width` / `--height` choose the aspect ratio of the sampling grid itself (no stretch). Keep the area within about 2× of training, or use `--hires-fix` for larger outputs.

---

## Quick reference: our tools

| Problem           | Training / prep                    | Inference / config reference                    |
|-------------------|------------------------------------|-------------------------------------------------|
| Hands             | Boost anatomy tags; object anchoring, simple poses | `--shortcomings-mitigation auto` |
| Faces/eyes        | Quality + anatomy in captions      | `--shortcomings-mitigation auto`                                 |
| Full body         | “full body”, “standing”, “legs” in captions | Same tags in prompt                             |
| Double head       | “single subject”, “no duplicate”   | 1:1 or full-body cues   |
| Text              | “text”, “lettering” boost; or add text in post | default negative; add text in post recommended  |
| Multiple subjects | Anti-blending in dataset           | Explicit count; negative: blending, merged      |
| Composition       | “clear composition”, “well proportioned” | `--shortcomings-mitigation auto`     |

**Code/config:**
- `config/prompts.py` — taxonomy + `mitigation_fragments` for `sample.py --shortcomings-mitigation` and `train.py --train-shortcomings-mitigation` (photoreal, **digital painting / concept / pixel / vector / game textures**, 3D render, plus optional 2D-anime packs with `--shortcomings-2d` / `--train-shortcomings-2d`; see [QUALITY.md](#quality-guide)).
- `config/styles.py` — artist-first medium + anatomy/proportion guidance for `sample.py --art-guidance-mode --anatomy-guidance`, `train.py --train-art-guidance-mode --train-anatomy-guidance`, and book pipeline forwarding.
- `config/styles.py` — style-domain + artist/game-name guidance for `sample.py --style-guidance-mode`, `train.py --train-style-guidance-mode`, normalize-captions tooling, and book pipeline forwarding.
- `data/caption_utils.py` — `DOMAIN_TAGS`, `boost_domain_tags`, `apply_training_guidance_to_caption_pair` (shortcomings / art / style guidance on caption pairs)

---

## 9. Community-wide issues (SDXL, Flux, Illustrious, NoobAI, Z-Image)

Concept bleeding, plastic skin, repetitive faces, artifacts, watermark stubbornness, CFG burn, centering bias, distant face meltdown, resolution lock, seed variance, and vocabulary gaps are all documented with **mitigations** (prompt tips, negatives, and sample.py flags) in **[docs/QUALITY.md](#quality-guide)** (*Community model issues*).

**sample.py flags that address these:**
- `--naturalize` — plastic/AI look (negative + natural prefix + capture finish)
- `--anti-bleed` — color/object bleeding (distinct colors positive + bleed negative)
- `--diversity` — repetitive/default face (diversity positive + same-face negative)
- `--anti-artifacts` — white dots, speckles, spiky artifacts
- `--strong-watermark` — stubborn logos/watermarks
- `--boost-quality` — prompt adherence; `--subject-first` — tag order
- `--cfg-rescale 0.7`, lower `--cfg-scale` — v-pred burn, oversaturation

**Config:** The related negatives are in `config/prompts.py` (e.g. `CONCEPT_BLEEDING_NEGATIVE`, `ARTIFACT_NEGATIVES`). See [QUALITY.md](#quality-guide) (*Community model issues*) for the full issue → mitigation table.

---

## 10. Gaps: what frontier models still struggle with — and what SDX does **not** fully solve

This section is the honest complement to §1–9 and [QUALITY.md](#quality-guide): **community mitigations** (prompts, negatives, flags) are not the same as a **guaranteed fix**. Below, **“no in-repo fix”** means there is no integrated module that *enforces* correctness; you rely on training data, luck, or external tools.

### A. Still fundamentally hard (diffusion + CLIP/T5 blind spots)

| Problem | Why models fail | What SDX has | What we **don’t** have |
| :--- | :--- | :--- | :--- |
| **Arbitrary spelled text** | Text is learned as texture; long strings are unstable. | OCR-guided **repair** when you know the target string (`sample.py` / `generate_book.py` + `utils/generation/editing.py`); lettering negatives; book pipeline expected text. | General “render this exact paragraph” without **known** expected text; no diffusers-style dedicated text renderer layer in-core. |
| **Exact counts** (“exactly 7 coins”) | No discrete counter in the denoiser. | Dataset-side `add_anti_blending_and_count`, explicit prompt counts, negatives for merging, and inference pick-best `--pick-best combo_count` (+ `--expected-count`, `--expected-count-target`, `--expected-count-object`) as lightweight people/simple-object count verification. | No hard cardinality guarantee for arbitrary object classes; still heuristic without constrained decoding/segmentation. |
| **Fine spatial logic** (“left hand holds blue cup, right waves”) | Conditioning is global; relations are fuzzy. | `--subject-first`, early prompt ordering. | Scene-graph / layout-conditioned attention (ControlNet-class conditioning is only partially exposed via `control_cond_dim` on `DiT_Text` — not a full layout stack). |
| **Physical plausibility** | No simulation; only statistics of pixels. | Tips, pick-best metrics (`utils/quality/test_time_pick.py`), orchestration hooks. | Physics engine, 3D consistency, reflection law enforcement ([`architecture_map`](../utils/modeling/architecture.py): `physical_grounding` = not in repo). |
| **Identity across unrelated images** | No persistent memory per user. | `--character-sheet`, **`--reference-image`** + CLIP → extra cross-attn tokens (`DiT_Text`), book anchors, `consistency_helpers`. | Strong identity still needs **trained** `--reference-adapter-pt` or LoRA; default projector is randomly initialized. |
| **Video / temporal coherence** | Out of scope for single-image DiT. | — | Native video diffusion or frame consistency training. |

### B. Partially mitigated (prompts + data — not architectural guarantees)

| Problem | SDX mitigation | Remaining gap |
| :--- | :--- | :--- |
| **Hands / anatomy** | `DOMAIN_TAGS`, shortcomings anatomy pack | No **auxiliary anatomy loss** or hand keypoint head on the denoiser ([`auxiliary_structure_supervision`](../utils/modeling/architecture.py)). |
| **Distant / small faces** | Close-up or medium shot; more pixels on the face | In `sample.py`: **`--face-enhance`** (Haar + local sharpen/contrast), **`--face-restore-shell`** for external GFPGAN/ADetailer CLIs; not a full in-repo face restoration model. |
| **Composition / cropping** | shortcomings composition pack, quality tags, inference `--resize-mode center_crop|saliency_crop` (+ `--resize-saliency-face-bias`) | Saliency crop is heuristic (not semantic segmentation/subject detector); complex multi-subject scenes may still need manual framing. |
| **Style / concept bleeding** | `--anti-bleed`, dataset boosts | No token-level cross-attention steering (SAG / per-token scale) — see [IMPROVEMENTS.md](RESEARCH.md#improvement-ideas-quality-fixes-and-better-training) §2.2. |

### C. Training / inference tooling gaps (listed elsewhere, summarized)

These are called out in [IMPROVEMENTS.md](RESEARCH.md#improvement-ideas-quality-fixes-and-better-training) and [DIFFUSION_LEVERAGE_ROADMAP.md](RESEARCH.md#diffusion-stack-high-leverage-upgrades-10-class-levers):

- **Few-step distilled student** (consistency / DMD) — no trainer in repo.  
- **Full flow-matching / rectified-flow training** — not drop-in for current `GaussianDiffusion` ([MODERN_DIFFUSION.md](RESEARCH.md#modern-diffusion--flow-ideas-20242026)).  
- **Extra samplers** (DPM++ / UniPC as first-class flags) — DDIM/Euler-style path is primary; more schedulers are “add” items.  
- **Inpaint-aware *training*** (random latent masks) — not wired; inpainting is inference/workflow (`sample.py`, book pipeline).  
- **DDP + resolution buckets** — buckets exist; multi-GPU + buckets is constrained.  
- **WebDataset / giant-scale streaming** — optional future in IMPROVEMENTS.  
- **Live web retrieval** — facts via `utils/prompt/rag.py` only; no built-in crawl.

### D. Where to invest if you want to *close* gaps (not just document them)

1. **Text:** double down on OCR + expected-text workflows **or** train with readable text in-domain; consider external typography for production UI.  
2. **Structure:** auxiliary heads or ControlNet-style conditioning (see [DIFFUSION_LEVERAGE_ROADMAP.md](RESEARCH.md#diffusion-stack-high-leverage-upgrades-10-class-levers) §3–4).  
3. **Identity:** reference-image adapter or stronger character-sheet conditioning in `DiT_Text`.  
4. **Inference control:** SAG / cross-attn hooks in `sample.py` ([IMPROVEMENTS.md](RESEARCH.md#improvement-ideas-quality-fixes-and-better-training) §2.2).  
5. **Faces at distance:** optional post-pass script wrapping GFPGAN/CodeFormer or face-region inpaint — **bridge** the doc↔code gap ADetailer users expect.

When you ship a new fix, update **§1–9** or [QUALITY.md](#quality-guide) for users, and **`architecture_map.py`** for theme status.

---

## Part 3 — General AI image failure modes

A reference guide to typical failure modes in current image diffusion and generative models—useful for training goals, caption design, and evaluation. For SDX-specific mitigations and config hooks, see [QUALITY.md](#quality-guide) and [QUALITY.md](#quality-guide).

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

These map to mitigation ids `digital_painting`, `concept_matte_digital`, `pixel_digital`, `vector_flat_digital`, `stylized_game_digital` in `config/prompts.py` (included in `auto` when keywords match, and in `all` with other non–2D-anime packs).

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

- **Registry and keyword detection:** `config/prompts.py`.
- **Artist-first medium guidance:** `config/styles.py` for traditional, digital, photography, and anatomy/proportion packs.
- **Style-domain + artist/game guidance:** `config/styles.py` for anime/comic/editorial/concept/game/photo language + artist/game-name stabilization cues.
- **Sampling:** `python sample.py ... --shortcomings-mitigation auto` (match prompts to categories) or `all` (full photoreal pack; add `--shortcomings-2d` for stylized 2D packs).
- **Sampling (medium/anatomy):** `python sample.py ... --art-guidance-mode auto|all --anatomy-guidance lite|strong` (optional `--no-art-guidance-photography`). Anatomy packs apply only when the prompt names a person, in both modes.
- **Art direction and prompt budget (default on):** `utils/prompt/art_direction.py` turns the prompt into an explicit plan (shot size, framing, horizon, lens and aperture, key light direction/quality/ratio, time of day and colour temperature, palette harmony and grade), keeps what you wrote, decides the rest from genre defaults, and places a short caption of the decisions right after your words. The prompt-stack budget then keeps your words, the caption, and pack text that fits `--prompt-budget` (96 T5 tokens by default; `--negative-budget` 160 for the negative), dropping pack text the plan contradicts (skin/hands with nobody in frame, camera terms on a painting). `--art-direction off` keeps the budget but skips the plan caption. The quality policy uses `auto` shortcomings packs and adds photo-capture packs (human-made, naturalize, anti-slop, DSLR media, photo realism) only when no non-photo medium is named.
- **Sampling (style domains):** `python sample.py ... --style-guidance-mode auto|all` (optional `--no-style-guidance-artists`).
- **Inference framing:** `python sample.py ... --resize-mode center_crop|saliency_crop` (optional `--resize-saliency-face-bias`) to reduce stretched/non-semantic framing when target aspect differs.
- **Training:** `python train.py ... --train-shortcomings-mitigation auto|all --train-art-guidance-mode auto|all` with optional `--train-shortcomings-2d`, `--train-anatomy-guidance`, `--no-train-art-guidance-photography`.
- **Training (style domains):** `python train.py ... --train-style-guidance-mode auto|all` (optional `--no-train-style-guidance-artists`).
- **Offline manifests:** `python -m scripts.tools normalize_captions ... --shortcomings-mitigation auto --art-guidance-mode auto --style-guidance-mode auto` (plus optional 2D/anatomy toggles).
## Game imagery and pixel-perfect pixel art

Detected from the prompt (`utils/prompt/game.py`), overridable with `--game-asset <type>|off`. Each asset type adds its
conventions to the prompt: sprite sheets (even grid, same framing per frame), seamless tiles, item icons (centred,
readable at 32 px), HUD/UI, character turnarounds, in-engine screenshots, key/card art, maps, environment concepts,
isometric, voxel, low-poly and cel-shaded.

Pixel art is finished as real pixel art (`utils/quality/pixel_art.py`): the art-pixel grid is detected (or set with
`--pixel-size` / "32x32" in the prompt), each cell gets one colour, the palette is k-means-quantised (`--palette-colors`;
1-bit 2, 2-bit 4, 8-bit 16, else 32) with near-duplicate shades merged, anti-aliasing leftovers are removed (glints kept),
and the image is rebuilt on an exact grid as uint8 so saving can't dither it. A native-size `<name>_sprite.png` is saved
beside the output. Sprites, sheets and icons get a transparent PNG background (`--transparent-bg auto|on|off`); tiles are
offset-blended to wrap seamlessly. With best-of-N, a grid / seam / icon-readability critic joins the ranking.

## Photorealism and hard prompts

`--realism off|auto|max` (default auto; `utils/prompt/realism.py`, `utils/quality/realism.py`). For prompts the art
direction plans as photographic, a shot spec picks camera, shutter and ISO that agree with the planned lens and light
(EV at ISO 100 per time of day / light source; handheld 1/focal, 1/125 with people, 1/1000 for action; ISO makes up the
rest, opening the aperture before passing ISO 6400; studio = strobe, ISO 100, 1/160). Values the prompt states are kept;
the rest is captioned (`--shot-spec off` drops the caption, `--shot-spec "85mm f/1.4 ISO 6400"` adds camera text). The
capture finish then always runs for those prompts, at the shot's ISO (luminance-weighted Poisson-Gaussian grain plus
CFA chroma noise) with vignette / lateral CA from its focal length and aperture, after a measured saturation tame-down
(off unless the 99th-percentile CIELAB chroma passes 80) and a filmic highlight shoulder (off without highlights).
`max` corrects at full strength and ranks best-of-4 with `combo_realism`. The AI-look critic gains plastic-skin and
waxy-highlight parts (skin fine-band texture, smooth or clipped sheen), so every metric using `ai_look` sees them.

`--hard-prompts off|auto|max` (default auto): exact counts ("five apples"), spatial relations (left/right of, above,
below, on top of, in front of, behind, next to) and colour binding with several subjects get prompt packs and a box
layout (one region per subject or counted instance, other subjects' colours in each region's negative) unless a layout
was given; the expected count / text feed the critics, text in the image (quotes, curly quotes, "a sign that says OPEN")
turns on the text-friendly negative, and `--pick-best auto` uses combo_count / combo_spatial_bind / combo_hand /
combo_human. `max` also turns on best-of-4. Product, food, interior, machinery, nature macro, technical illustration and
diagram/map prompts get `--scene-domain` packs; holding / waving / group prompts infer the hand and people-layout packs.

---

## Superior Stack

Composable modules that raise **inference and data-prep quality** on top of your trained DiT checkpoint. This is not a separate model — it is orchestration, retrieval, ranking, and correction around `sample.py`.

### Modules (`utils/agentic/quality_loop.py`)

The former `utils/superior/` files below are now sections of `utils.agentic.quality_loop` (old file names kept for orientation).

| Module | Role |
|--------|------|
| `retrieval.py` | TF-IDF over JSONL facts (local RAG, no API) |
| `auto_stack.py` | Merge retrieved facts + light quality hints into the prompt |
| `composite_ranker.py` | The one candidate aggregator: weight presets over `test_time_pick` terms (`superior_composite`, `online` for GRPO, `ensemble`); Borda ranking or absolute `score_one` |
| `self_correct.py` | CLIP alignment gate + short refine pass |
| `inference_pipeline.py` | Build `sample.py` argv for high-quality defaults |
| `utils/generation/guidance.py` | Composable guidance: APG, FDG, CFG rescale, CFG-Zero★, TCFG, PAG, SLG, guidance interval |

### Quick start

**CLI wrapper** (4 candidates + composite ranking + optional RAG):

```bash
python -m scripts.tools superior_generate \
  --ckpt results/your_run/best.pt \
  --prompt "neon alley at night, cinematic" \
  --local-rag-jsonl datasets/style_notes.jsonl \
  --num 4 --out out.png
```

**Direct `sample.py` flags:**

```bash
python sample.py --ckpt results/.../best.pt \
  --prompt "..." --num 4 --pick-best superior_composite \
  --local-rag-jsonl datasets/captions.jsonl --local-rag-top-k 8 \
  --superior-self-correct --compile-inference --out out.png
```

### Training-side

- **`python -m scripts.tools superior_dpo_loop`** — mine benchmark pairs → train Diffusion-DPO
- **`python -m scripts.tools superior_curate`** — dedup/filter JSONL via `utils/quality/critics.py`
- **`python -m scripts.tools model_soup`** — average multiple checkpoints
- Use **`utils/training/preference.py`** for teacher→student latent distillation (custom loop).
- See **`python -m scripts.tools train_kd_distill`** for KD entry points.

### Full alignment loop (wave 3)

One command runs benchmark → ViT-weighted preference mining → DPO → re-benchmark → optional model soup:

```bash
python -m scripts.tools superior_auto_loop \
  --base-ckpt results/run/best.pt \
  --vit-ckpt vit_quality/results/best.pt \
  --local-rag-jsonl datasets/style_notes.jsonl \
  --model-soup --iterations 2
```

Lower-level control: `python -m scripts.tools auto_improve_loop` with `--vit-ckpt`, `--local-rag-jsonl`, `--model-soup`.

### Full flywheel (wave 4)

Curate data → align (benchmark/DPO/soup) → promote best checkpoint:

```bash
python -m scripts.tools run_flywheel \
  --base-ckpt results/run/best.pt \
  --manifest-in datasets/train.jsonl \
  --manifest-out datasets/train_clean.jsonl \
  --local-rag-jsonl datasets/style_facts.jsonl \
  --vit-ckpt vit_quality/results/best.pt \
  --work-dir flywheel_run
```

**Ensemble inference** (multiple checkpoints, one winner):

```bash
python -m scripts.tools superior_ensemble \
  --checkpoints results/a/best.pt results/b/dpo_policy.pt soup_policy.pt \
  --prompt "portrait of an engineer" --num-per-ckpt 2 --out best.png
```

**Eval report** from benchmark output:

```bash
python -m scripts.tools superior_eval_report bench_after/
```

Use **`--preset superior`** in ``sample.py`` for Superior Stack sampler defaults.

### Libraries added (wave 4)

| Module | Role |
|--------|------|
| `config/presets.py` | Central defaults + `FlywheelPlan` |
| `utils/agentic/quality_loop.py` | Curate + align orchestration |
| `utils/agentic/quality_loop.py` | Multi-checkpoint generation + global rank |
| `utils/agentic/quality_loop.py` | Mine negatives from failed cases |
| `utils/agentic/quality_loop.py` | Markdown benchmark reports |

### Libraries added (wave 3)

| Module | Role |
|--------|------|
| `utils/agentic/quality_loop.py` | ViT-weighted DPO pair mining |
| `utils/agentic/quality_loop.py` | LLM or heuristic prompt expansion |
| `utils/agentic/quality_loop.py` | Python API for auto-improve loop |

| Module | Role |
|--------|------|
| `utils/quality/critics.py` | Programmatic manifest curation API |
| `utils/agentic/quality_loop.py` | Benchmark → DPO training orchestration |
| `utils/agentic/quality_loop.py` | Checkpoint averaging |
| `utils/agentic/quality_loop.py` | Fast pre/post generation gates |
| `models/attention.py` | Fused SDPA cross-attention path |

### Wave 5 (research-backed alignment + inference)

| Module | Role |
|--------|------|
| `utils/training/preference.py` | Timestep DPO weights, SDPO safeguard, EMA ref |
| `utils/modeling/architecture.py` | SpeCa-lite step reuse (`--feature-cache-delta`) |

**DPO trainer flags:** `--timestep-dpo-weight high_noise`, `--safeguarded-dpo 0.85`, `--ref-ema-alpha 0.01`

**Flywheel (any OS):** `python -m scripts.tools superior flywheel --base-ckpt results/best.pt --work-dir flywheel_run`

See [research/SUPERIOR_RESEARCH_2026.md](RESEARCH.md#superior-stack--research-map-20252026) for paper mapping.

### Wave 6 (speed + online RL + distill flywheel)

| Module | Role |
|--------|------|
| `utils/modeling/architecture.py` | BWCache/TeaCache-lite per-block DiT reuse |
| `utils/training/preference.py` | Flow-GRPO scaffold (sample → reward → weighted update) |
| `train_consistency_distill.py` | LCM-style consistency distillation (few-step student) |
| `train_flow_grpo.py` | Online GRPO alignment trainer |
| `train_ladd_distill.py` | LADD teacher MSE + latent discriminator |
| `hard_negative.py` | Flywheel closure via `--use-hard-negatives` in auto_improve_loop |

**Block cache inference:** `--block-cache-thresh 0.18 --block-cache-recompute-every 4`

**Few-step student:** `python -m scripts.tools train_consistency_distill --teacher-ckpt ... --data ...`

**Online RL:** `python -m scripts.tools train_flow_grpo --ckpt ... --prompts prompts.txt --vit-ckpt ...`

### Wave 7 (TaylorSeer + DenseGRPO + LCM fast path)

| Module | Role |
|--------|------|
| `utils/modeling/architecture.py` | TaylorSeer block forecast (`--taylor-cache`) |
| `utils/training/preference.py` | Step-wise dense GRPO rewards |
| `sample.py --lcm-ckpt` | Few-step consistency student (4-step flow) |

**Taylor + block cache:** `--block-cache-thresh 0.18 --taylor-cache --taylor-cache-order 1`

**DenseGRPO training:** `train_flow_grpo --dense-grpo --dense-ode-steps 4`

**LCM inference:** `sample.py --ckpt base.pt --lcm-ckpt lcm_student.pt --lcm-steps 4`

### Honest scope

Frontier labs have proprietary data scale, eval harnesses, and foundation pretraining. SDX Superior Stack gives you **testable, local** levers: grounded prompts, better candidate selection, and alignment-based refine — wired to your checkpoint and prompt stack. Quality still depends on **training data and your DiT weights**.

### See also

- [IMPROVEMENTS.md](RESEARCH.md#improvement-ideas-quality-fixes-and-better-training) — training/inference roadmap
- [LANDSCAPE_2026.md](RESEARCH.md#image-generation-landscape-architecture-and-workflows-2026) — industry patterns (Designer / Verifier / Reasoner)
- [recipes/fast_training.md](GETTING_STARTED.md#fast-training-and-sampling-no-quality-trade-offs) — throughput defaults

---

## RSI: your model teaches itself to make better art

`rsi_loop` improves an SDX checkpoint with no human in the loop. Your thumbs-up/down are optional extra signal.
It only practises art: every prompt it trains on is a medium (oil painting, pixel art, linocut,
portrait photo, …, the 47 mediums in `config/styles.py`) plus a subject, light and framing.

### One round

1. **Curriculum**: 16 art prompts, weighted toward the medium family (traditional / 2d / photo / generic)
   the champion currently scores worst on.
2. **Paint**: the champion makes 4 candidates per prompt through the normal `sample.py` pipeline.
   `--jobs-jsonl` loads the checkpoint and text encoders once per round.
3. **Judge**: one art score per image. It blends the HF reward panel (PickScore, CLIP-H, CAFE aesthetic,
   whichever are on disk), `art_critic` (composition, exposure, colour harmony) and `ai_look`
   (decoder grid, halos, oversmoothing and other generated-look tells).
4. **Learn**: best-vs-worst pairs per prompt (gap ≥ `--min-margin`) plus your feedback pairs train a
   candidate with `train_diffusion_dpo`.
5. **Gate**: candidate and champion paint the same held-out prompts with the same seeds. Held-out
   subjects never appear in training prompts. The candidate is promoted only if the mean score rises
   by `--min-gain` **and** it wins at least half the head-to-heads. Otherwise it is deleted and the
   learning rate halves.

### Run it

```bash
python -m scripts.tools rsi_loop --base-ckpt results/run/best.pt --rounds 5   # start
python -m scripts.tools rsi_loop --rounds 5                                   # resume, 5 more rounds
python -m scripts.tools rsi_loop --status                                     # history
python -m scripts.tools rsi_loop --base-ckpt results/run/best.pt --dry-run    # preview prompts + commands
```

Output goes to `outputs/rsi/`: `champion.pt` (current best), `champion.prev.pt` (the one before it),
`state.json`, and `round_NNN/` with images, `prefs.jsonl` and `report.json`.

Before the first round:

- **Train first.** RSI sharpens a model that already draws. It cannot rescue a checkpoint that still outputs noise.
- **The checkpoint must sample.** `python sample.py --ckpt <ckpt> --prompt test` has to work with its text
  encoder (T5-XXL etc.) and VAE on disk.
- **Offline by default.** The judge sets `HF_HUB_OFFLINE=1` unless you set it, so a long unattended run
  never starts a surprise multi-GB download.

### Add your own taste

```bash
python sample.py --ckpt ... --prompt "..." --out out.png --log-feedback
python -m scripts.tools feedback like out.png --prompt "..."
python -m scripts.tools feedback pair --win a.png --lose b.png --prompt "..."
python -m scripts.tools feedback sync-taste    # immediate inference bias: sample.py --apply-user-taste
```

Every round mixes the pairs from `outputs/feedback/feedback.jsonl` into training (`--from-feedback ''` turns that off).
Video frames can be rated with `--media-type video`, but RSI trains on images only.

---

## Baseline prompt pack for quick eval

The repo ships [`examples/eval_prompts_baseline.json`](../examples/eval_prompts_baseline.json): short prompts that stress **object**, **hands**, **text**, **counting**, **style**, etc.

### Driver script (recommended)

[`examples/run_baseline_eval.py`](../examples/run_baseline_eval.py) prints `sample.py` commands by default; add `--execute` to run them.

```bash
python examples/run_baseline_eval.py --ckpt results/your_run/best.pt
python examples/run_baseline_eval.py --ckpt results/your_run/best.pt --execute -- --steps 30 --preset auto
```

Arguments after `--` are forwarded to `sample.py`.

### Load in Python

```python
from examples.run_baseline_eval import load_eval_prompts

for eid, prompt in load_eval_prompts("examples/eval_prompts_baseline.json"):
    print(eid, "->", prompt[:60], "...")
```

### Docs

- Experiment grids: [SAMPLING_EXPERIMENTS_BACKLOG.md](RESEARCH.md#sampling-experiments-backlog)
- Quality research map: [IMAGE_QUALITY_LEVERS_2026.md](RESEARCH.md#image-quality-research-levers-20252026--sdx-map)

---

## Quick evaluation recipe (Holy Grail + manifests)

Copy-paste oriented. Paths assume repo root.

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2. One-command sanity (no custom checkpoint)

Uses DiT-XL/2 ImageNet weights from Hugging Face (see repo `demo.py`).

```bash
python demo.py
```

### 3. Sample with a text checkpoint + Holy Grail preset

Replace the checkpoint path with your `results/.../best.pt` (or equivalent).

```bash
python sample.py --ckpt results/your_run/best.pt \
  --prompt "your prompt here" \
  --preset auto \
  --out out.png
```

Adjust steps, resolution, and CFG flags per `README.md` and `docs/QUALITY.md`.

### 4. Training run artifacts

When training with `train.py`, the experiment directory receives reproducibility files:

- `config.train.json` — frozen training config snapshot
- `run_manifest.json` — command, git info, torch/CUDA versions, seed, etc.

Disable with `--no-save-run-manifest` if you truly do not want these files.

### 5. Deeper reading

- [HOLY_GRAIL_OVERVIEW.md](HOW_GENERATION_WORKS.md#holy-grail-adaptive-sampling-how-it-fits-together)
- [TCIS_OVERVIEW.md](HOW_GENERATION_WORKS.md#tcis--tri-consensus-iterative-synthesis)
- [QUALITY.md](#quality-guide)

### See also

- [eval_baseline_prompts.md](#baseline-prompt-pack-for-quick-eval) — structured prompt pack + 
un_baseline_eval.py.
### See also

- [local_ci_mirror.md](GETTING_STARTED.md#mirror-github-ci-locally) — run the same checks as CI before you push.
