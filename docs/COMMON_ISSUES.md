# Common model issues and how we mitigate them

Community-reported problems (SDXL, Flux/Klein, Illustrious/NoobAI, Z-Image, Civitai) and the mitigations available in SDX: **config/prompt_domains.py** tips/negatives, **sample.py** flags, and **training** practices.

**Sources:** Reddit r/StableDiffusion, Civitai articles & comments, FLUX GitHub (#406 grid artifact), [Stable Diffusion Art](https://stable-diffusion-art.com/common-problems-in-ai-images-and-how-to-fix-them/), ComfyUI prompt-engineering docs, Black Forest Labs FLUX.

---

## Quick reference: issue → mitigation

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

## sample.py flags that map to these issues

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

## Config reference

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

Training: **data/caption_utils.py** boosts `DOMAIN_TAGS["concept_bleed"]` (distinct colors, no color bleed) when present in captions. Use anti-blending for multiple characters (§6 in MODEL_WEAKNESSES.md).
