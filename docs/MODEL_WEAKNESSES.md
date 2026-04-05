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
- `config/defaults/ai_image_shortcomings.py` — taxonomy + `mitigation_fragments` for `sample.py --shortcomings-mitigation` and `train.py --train-shortcomings-mitigation` (photoreal, **digital painting / concept / pixel / vector / game textures**, 3D render, plus optional 2D-anime packs with `--shortcomings-2d` / `--train-shortcomings-2d`; see [COMMON_SHORTCOMINGS_AI_IMAGES.md](COMMON_SHORTCOMINGS_AI_IMAGES.md)).
- `config/defaults/art_mediums.py` — artist-first medium + anatomy/proportion guidance for `sample.py --art-guidance-mode --anatomy-guidance`, `train.py --train-art-guidance-mode --train-anatomy-guidance`, and book pipeline forwarding.
- `config/defaults/style_guidance.py` — style-domain + artist/game-name guidance for `sample.py --style-guidance-mode`, `train.py --train-style-guidance-mode`, normalize-captions tooling, and book pipeline forwarding.
- `data/caption_utils.py` — `DOMAIN_TAGS`, `boost_domain_tags`, `apply_shortcomings_to_caption_pair`, `NEGATIVE_ANATOMY`, `NEGATIVE_FACE`, `NEGATIVE_COMPOSITION`, `NEGATIVE_QUALITY`, `NEGATIVE_TEXT`, `NEGATIVE_ANATOMY_FULL`
- `config/prompt_domains.py` — `ANATOMY_NEGATIVES`, `HAND_FIX_PROMPT_TIPS`, `PORTRAIT_ASPECT_TIPS`, `RECOMMENDED_NEGATIVE_BY_DOMAIN`

---

## 9. Community-wide issues (SDXL, Flux, Illustrious, NoobAI, Z-Image)

Concept bleeding, plastic skin, repetitive faces, artifacts, watermark stubbornness, CFG burn, centering bias, distant face meltdown, resolution lock, seed variance, and vocabulary gaps are all documented with **mitigations** (prompt tips, negatives, and sample.py flags) in **[docs/QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md)** (*Community model issues*).

**sample.py flags that address these:**
- `--naturalize` — plastic/AI look (negative + natural prefix + film grain)
- `--anti-bleed` — color/object bleeding (distinct colors positive + bleed negative)
- `--diversity` — repetitive/default face (diversity positive + same-face negative)
- `--anti-artifacts` — white dots, speckles, spiky artifacts
- `--strong-watermark` — stubborn logos/watermarks
- `--boost-quality` — prompt adherence; `--subject-first` — tag order
- `--cfg-rescale 0.7`, lower `--cfg-scale` — v-pred burn, oversaturation

**Config:** All related negatives and tip lists are in `config/prompt_domains.py` (e.g. `CONCEPT_BLEEDING_NEGATIVE`, `ARTIFACT_NEGATIVES`, `EMOTION_PROMPT_TIPS`, `SPATIAL_AWARENESS_TIPS`). See [QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md) (*Community model issues*) for the full issue → mitigation table.

---

## 10. Gaps: what frontier models still struggle with — and what SDX does **not** fully solve

This section is the honest complement to §1–9 and [QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md): **community mitigations** (prompts, negatives, flags) are not the same as a **guaranteed fix**. Below, **“no in-repo fix”** means there is no integrated module that *enforces* correctness; you rely on training data, luck, or external tools.

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

When you ship a new fix, update **§1–9** or [QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md) for users, and **`architecture_map.py`** for theme status.
