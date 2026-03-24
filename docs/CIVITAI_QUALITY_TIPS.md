# Civitai-style quality tips: fix common SD model issues

Based on common problems reported in Civitai checkpoint comments and SD community guides (oversaturation, blur, bad hands, wrong resolution). Use these with `sample.py` and training to get more reliable, high-quality output.

---

## 1. Oversaturation / “burned” colors (high CFG)

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

## 2. Blurry or soft output

**Causes:** Non-native resolution, too few steps, VAE/quantization, or scaling artifacts.

**Fixes:**

- **Use native resolution** — Prefer the model’s training size (e.g. 256 or 512). If you set `--width` / `--height` far from that, you’ll see a note; use `--vae-tiling` for large decodes to reduce VRAM and artifacts.
- **More steps** — 25–35 steps often improve clarity; we default to 50.
- **Post-process** — `--sharpen 0.3` and `--contrast 1.05` can help (see `utils/quality/quality.py`).
- **VAE** — Train/sample with a good VAE (e.g. sd-vae-ft-mse or sdxl-vae); avoid heavily quantized VAEs for final decode.

---

## 3. Bad hands / anatomy

**Problem:** Extra fingers, fused fingers, deformed hands or limbs.

**Fixes:**

- **Default negative prompt** — If you leave `--negative-prompt` empty, we use a Civitai-style default: `low quality, worst quality, blurry, bad anatomy, bad hands, missing fingers, extra fingers, fused fingers, deformed, duplicate`. Override with `--negative-prompt "..."` if your model prefers minimal negatives (e.g. some SDXL-style models).
- **Positive cues** — In the prompt, describe hands in context: e.g. “holding a cup”, “hands on keyboard”, “correct hands, five fingers”.
- **Simpler poses** — “Arms crossed”, “hands in pockets” tend to work better than complex hand poses.
- **Inpainting** — For a single image, use `--init-image` + `--mask` to inpaint only the hand/face area at 0.4–0.5 strength.

See also `config/prompt_domains.py`: `ANATOMY_NEGATIVES`, `HAND_FIX_PROMPT_TIPS`.

---

## 4. Positive vs negative prompt conflict

**Problem:** You want “cat, dog, portrait” but your negative prompt says “dog, blurry”. CFG would then push away from *dog* as well, contradicting the positive prompt.

**Fix (default):** `sample.py` **filters the negative prompt** by removing any token that also appears in the positive (split on comma/space, case-insensitive). So with pos “cat, dog, portrait” and neg “dog, blurry”, the effective negative becomes “blurry” only. The model is guided away from blur without fighting your request for a dog.

- To **disable** this and use the raw negative prompt, pass **`--no-neg-filter`**.

---

## 5. Text in image (signs, lettering, legible text)

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

## 6. Tags and LoRAs

**Tag-based prompts:** Use `--tags "1girl, long hair, outdoors, sunset"` or `--tags-file path/to/tags.txt` (one tag per line or comma-separated). Tags are normalized and reordered for better adherence: **subject** (1girl, 1boy, etc.) → **age** (teen, young, adult, elderly) → **height** (tall, short, petite) → **build/size** (slim, muscular, curvy) → **anatomy/framing** (full body, portrait, bust) → **body parts** (hands, long hair, etc.) → rest. Use `--subject-first` with a plain `--prompt` to get the same ordering. You can omit `--prompt` when using `--tags` or `--tags-file`.

**LoRAs:** `sample.py` loads `.pt` and `.safetensors` LoRAs (`--lora path.safetensors path2.pt:0.6`). Many style/character LoRAs expect a **trigger word** in the prompt. Use `--lora-trigger "style name"` (or the trigger from the LoRA's model card); it is prepended to the prompt automatically when any LoRA is applied. Example:

```bash
python sample.py --ckpt .../best.pt --tags "1girl, smile" --lora my_style.safetensors --lora-trigger "MyStyle" --out out.png
```

This produces a prompt like `MyStyle, 1girl, smile` so the LoRA activates correctly while keeping tag order.

**People: height, age, size, anatomy:** The tag order puts **subject → age → height → build/size → anatomy/body parts** before other tags. So prompts like `long hair, 1girl, tall, teen, slim, full body` are reordered to `1girl, teen, tall, slim, full body, long hair`. Training data uses the same ordering and boosts these person-descriptor tags so the model learns height, age, body size, and anatomy (e.g. full body, portrait, hands, feet) reliably.

---

## 7. Complex and long prompts
**Problem:** Long or multi-part prompts get truncated (T5 max 300 tokens); the model may ignore later parts or lose adherence.

**Fixes:**
- **Put the main subject first** — e.g. "1girl, red dress, in a forest, sunset, ..." so the most important part is not truncated.
- **Use comma-separated tags** — clearer than long sentences for the model.
- **`--boost-quality`** — Prepends "masterpiece, best quality" to the prompt for stronger adherence (helpful for complex or challenging prompts).
- If the prompt has **>250 tokens**, `sample.py` prints a note; shorten or reorder so key elements are early.
- Use **(word)** to emphasize critical terms and **[word]** to de-emphasize.

See `config/prompt_domains.py`: `COMPLEX_PROMPT_TIPS`.

---

## 8. Challenging content (NSFW, surreal, abstract, weird)
**Problem:** Model under-adheres or produces generic output for mature, surreal, or unusual prompts.

**Fixes:**
- **Do not censor** — Use clear, descriptive tags in training data and at inference; the model learns what you show.
- **Quality tags help** — Add "masterpiece, best quality, detailed, sharp focus" so any challenging prompt gets stronger conditioning.
- **Be concrete** — For surreal/abstract, describe colors, layout, and mood; put subject first, then setting, then style.
- **`--boost-quality`** — Use for maximum adherence on challenging prompts.
- **Training:** Include diverse examples with consistent tags; the data pipeline boosts "complex" and "challenging" domain tags (see `data/caption_utils.py`, `config/prompt_domains.py`: `CHALLENGING_PROMPT_TIPS`).

---

## 9. Hard styles (3D, photorealistic, style mixes) and LoRA mixing

**Problem:** Many models blur 3D vs 2D, realistic vs illustrated, and mixed styles (2.5D, semi-realistic, photorealistic anime). Combining multiple LoRAs (e.g. style + character) can give muddy or oversaturated results.

**Fixes:**

- **`--hard-style`** — Use `--hard-style 3d`, `realistic`, `3d_realistic`, or `style_mix` to prepend recommended tags so the model anchors on the right look. Set `--negative-prompt` from `config/prompt_domains.py` → `HARD_STYLE_NEGATIVES` for that style (e.g. for 3d: "flat, 2d, blurry, bad proportions").
- **Training:** Captions that contain 3D, photorealistic, or style-mix phrases (e.g. "2.5d", "semi-realistic", "photorealistic anime") are boosted **first** in the pipeline (`data/caption_utils.py` → `HARD_STYLE_TAGS`, `boost_hard_style_tags`) so the model learns these hard styles well.
- **Style mixing:** Put the dominant style first in the prompt; use explicit phrases like "2.5d", "semi-realistic", "photorealistic anime". See `config/prompt_domains.py` → `STYLE_MIX_TIPS`.
- **Multiple LoRAs:** Use lower scales (0.5–0.6 each) so they blend; put the dominant LoRA first and its trigger at the start of the prompt. See `LORA_MIX_TIPS` in the same file. If output is muddy, try `--cfg-scale 5` or `--cfg-rescale 0.7`.

See [docs/DOMAINS.md](DOMAINS.md) for full hard-style prompts and negatives.

---

## 10. Wrong resolution warning

If you pass `--width` / `--height` much larger or smaller than the model’s native size (e.g. 1024 when trained at 256), `sample.py` prints a note. For best quality:

- Prefer the model’s native resolution, or
- Use a small multiple (e.g. 1.5×) and optionally `--vae-tiling` for large decode.

---

## 11. Quick reference: sample.py flags that address these issues

| Issue              | Flags to try |
|--------------------|--------------|
| Oversaturation     | `--cfg-rescale 0.7`, `--cfg-scale 5` or `7` |
| Blur               | Native resolution, `--steps 35`, `--sharpen 0.3`, `--vae-tiling` for large size |
| Bad hands/anatomy  | Default negative (or add `bad hands, extra fingers`), describe hands in prompt |
| Pos/neg conflict   | Default: neg is filtered so tokens in pos are removed from neg; use `--no-neg-filter` to disable |
| Text in image      | Describe exact text in prompt (“sign that says X”); use `--text-in-image` or leave neg empty so text-friendly negative is used |
| Tags / LoRAs        | `--tags "tag1, tag2"` or `--tags-file path`; `--lora path.safetensors --lora-trigger "word"` for style/character LoRAs |
| Reproducibility     | `--save-prompt` writes prompt, seed, steps to a `.txt` next to the image; `--subject-first` reorders comma-separated prompt (subject first) |
| Complex/long prompts | Put subject first; use `--boost-quality`; keep key elements in first ~250 tokens |
| Challenging (NSFW, surreal) | No censorship; add quality tags; use `--boost-quality`; see CHALLENGING_PROMPT_TIPS |
| Hard styles (3D, realistic, 2.5D) | `--hard-style 3d | realistic | 3d_realistic | style_mix`; set negative from `HARD_STYLE_NEGATIVES`; see STYLE_MIX_TIPS, LORA_MIX_TIPS |
| Multiple LoRAs / style mix | Lower LoRA scales (0.5–0.6); dominant LoRA first; `--cfg-rescale 0.7` if muddy |
| Large output OOM   | `--vae-tiling` |
| Softer / more natural | `--cfg-scale 5` |
| Less AI-looking / plastic | `--naturalize` (negative + prompt prefix + film grain); tune `--naturalize-grain 0.02` |
| Small artifacts polish | Refinement is on by default; disable with `--no-refine` (or tune `--refine-t`) |
| Color/concept bleeding (SDXL) | `--anti-bleed`; put spatial relations early; see [COMMON_ISSUES.md](COMMON_ISSUES.md) |
| Repetitive / same face (Flux) | `--diversity`; see `EMOTION_PROMPT_TIPS`, `DIVERSITY_POSITIVE` |
| Less templated / more novel | `--originality 0.3-0.8` (injects unique composition tokens + tunes creativity/CFG) |
| White dots / speckles / spiky artifacts | `--anti-artifacts`; “particle” in negative |
| Stubborn watermarks (e.g. Illustrious) | `--strong-watermark` |
| V-pred burn (NoobAI) | `--cfg-scale 4.5`, `--cfg-rescale 0.7`; see `CFG_BURN_TIPS` |
| FLUX grid artifact | CFG ≤ 3.5, LoRA strength ≤ 1.20; see `FLUX_GRID_ARTIFACT_TIPS` |
| Full body / two-head | Add standing, legs, long dress, shoes; 1:1 for head shot; see `FULL_BODY_AND_TWO_HEAD_TIPS` |
| Negative prompt ineffective | Be specific (e.g. extra fingers), not vague; see `NEGATIVE_PROMPT_BEST_PRACTICES` |

---

## 12. FLUX grid artifact and full-body / two-head (community fixes)

**FLUX grid:** Visible grid pattern in dark areas (worse with Depth/Canny and upscaling). From [FLUX GitHub #406](https://github.com/black-forest-labs/flux/issues/406): keep **CFG at 3.5 or lower** and **LoRA strength at or below 1.20**; avoid overtrained LoRAs. See `config/prompt_domains.py` → `FLUX_GRID_ARTIFACT_TIPS`.

**Full body / two-head:** From [Stable Diffusion Art](https://stable-diffusion-art.com/common-problems-in-ai-images-and-how-to-fix-them/): (1) For full body, add **standing, long dress, legs, shoes** (describe what should appear) rather than only “full body portrait”. (2) Use **1:1** for head/shoulder shots to avoid two-head; use portrait aspect for full-body so one body fills the frame. (3) Garbled faces often need more pixels: use close-up, Hi-Res Fix, better VAE, or face restorer/inpainting. See `FULL_BODY_AND_TWO_HEAD_TIPS`, `GARBLED_FACE_TIPS`.

---

## 13. Making images look less AI-generated

**Problem:** Outputs look too smooth, plastic, waxy, or “obviously generated.”

**Fixes:**

- **`--naturalize`** — One flag that: (1) adds an **anti-AI-look negative** (plastic skin, oversmooth, airbrushed, waxy, doll-like, synthetic, CGI, uncanny, etc.) so the model is steered away from those traits; (2) prepends **natural-look positive** hints (film grain, natural skin texture, subtle imperfections, raw photo, natural lighting) to the prompt; (3) runs a **post-process**: subtle film grain + slight micro-contrast on the decoded image to break up the plastic smoothness.
- **`--naturalize-grain 0.02`** — Adjust grain strength (0 = no grain, 0.01–0.03 typical). Use 0 if you only want the prompt/negative changes.
- **Lower CFG** — Try `--cfg-scale 5` or `6` for a less “pushed” look.
- **Training:** Include diverse, non-perfect references and captions that mention “natural skin texture,” “film grain,” “imperfections,” so the model learns a less polished look. You can add `ANTI_AI_LOOK_NEGATIVE`-style terms to your training negatives (see `config/prompt_domains.py`).

---

## 14. Training-side tips (fewer issues at inference)

- **Negative prompt in data** — Include quality/anatomy negatives in captions or the dataset so the model learns to avoid them (see `RECOMMENDED_NEGATIVE_BY_DOMAIN`, `ANATOMY_NEGATIVES` in `config/prompt_domains.py`).
- **Min-SNR loss** — `--min-snr-gamma 5` (or similar) stabilizes training and can improve clarity.
- **Native resolution** — Train at the resolution you plan to use for inference to avoid resolution mismatch blur.

These tips align with common Civitai and A1111/ComfyUI recommendations; adjust per checkpoint if a model card suggests different settings.
