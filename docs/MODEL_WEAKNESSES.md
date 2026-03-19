# What other models suck at — and how we prepare and fix it

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
- `data/caption_utils.py` — `DOMAIN_TAGS`, `boost_domain_tags`, `NEGATIVE_ANATOMY`, `NEGATIVE_FACE`, `NEGATIVE_COMPOSITION`, `NEGATIVE_QUALITY`, `NEGATIVE_TEXT`, `NEGATIVE_ANATOMY_FULL`
- `config/prompt_domains.py` — `ANATOMY_NEGATIVES`, `HAND_FIX_PROMPT_TIPS`, `PORTRAIT_ASPECT_TIPS`, `RECOMMENDED_NEGATIVE_BY_DOMAIN`

---

## 9. Community-wide issues (SDXL, Flux, Illustrious, NoobAI, Z-Image)

Concept bleeding, plastic skin, repetitive faces, artifacts, watermark stubbornness, CFG burn, centering bias, distant face meltdown, resolution lock, seed variance, and vocabulary gaps are all documented with **mitigations** (prompt tips, negatives, and sample.py flags) in **[docs/COMMON_ISSUES.md](COMMON_ISSUES.md)**.

**sample.py flags that address these:**
- `--naturalize` — plastic/AI look (negative + natural prefix + film grain)
- `--anti-bleed` — color/object bleeding (distinct colors positive + bleed negative)
- `--diversity` — repetitive/default face (diversity positive + same-face negative)
- `--anti-artifacts` — white dots, speckles, spiky artifacts
- `--strong-watermark` — stubborn logos/watermarks
- `--boost-quality` — prompt adherence; `--subject-first` — tag order
- `--cfg-rescale 0.7`, lower `--cfg-scale` — v-pred burn, oversaturation

**Config:** All related negatives and tip lists are in `config/prompt_domains.py` (e.g. `CONCEPT_BLEEDING_NEGATIVE`, `ARTIFACT_NEGATIVES`, `EMOTION_PROMPT_TIPS`, `SPATIAL_AWARENESS_TIPS`). See COMMON_ISSUES.md for the full issue → mitigation table.
