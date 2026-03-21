# Image generation landscape (~March 2026)

**Context:** Text-to-image is increasingly **production infrastructure**, not only a “cool art” demo. Teams care about **functional precision**, **grounding**, and **outputs that read as real**—not just high scores on aesthetics.

This doc **does not** claim SDX ships commercial parity with any external product. It **does** map where the field is moving and how SDX can evolve (see also [IMPROVEMENTS.md](IMPROVEMENTS.md) §12, [utils/orchestration.py](../utils/orchestration.py)).

---

## 1. Authenticity over “the AI look”

**Trend:** Audiences are good at spotting **plastic** skin, **over-perfect** symmetry, and **HDR mush**. Newer systems often train or sample toward **lens-like** imperfections: micro-texture, plausible noise, asymmetric framing, natural light falloff.

**SDX today**

- Training/inference knobs that reduce **CFG blowout** and oversaturation: CFG rescale, dynamic threshold, Min-SNR, noise offset ([CIVITAI_QUALITY_TIPS.md](CIVITAI_QUALITY_TIPS.md)).
- **Refinement** and post-process (`utils/quality.py` naturalize/sharpen).
- **Domain / style** tooling ([STYLE_ARTIST_TAGS.md](STYLE_ARTIST_TAGS.md), [DOMAINS.md](DOMAINS.md)).

**Good next steps in-repo**

- Optional **texture / frequency** regularization or auxiliary losses (document loss schedule; see [IMPROVEMENTS.md](IMPROVEMENTS.md) §11).
- **Multi-resolution / aspect buckets** so the model sees non-square and varied crops ([IMPROVEMENTS.md](IMPROVEMENTS.md) §1.1).
- Dataset-side **“lived-in”** captions (wear, dust, film grain) in JSONL—not only tag lists.

---

## 2. Multi-component “system of experts”

**Trend:** One monolith is replaced by a **pipeline**: e.g. a **layout/composition** stage, a **verification** stage (anatomy, physics, consistency), and a **reasoning** stage (complex instructions, ordering, negation).

**SDX today**

- **Designer-like core:** `sample.py` + DiT + VAE (and optional Control / regional captions).
- **Verifier-like hooks:** `--pick-best` with CLIP / edge / OCR ([utils/test_time_pick.py](../utils/test_time_pick.py)); ViT quality tools under `ViT/`.
- **Reasoner-like conditioning:** T5 (+ optional triple fusion); optional LLM prompt expansion ([utils/llm_client.py](../utils/llm_client.py)).

**Good next steps**

- Explicit **orchestration** API or script that chains: generate K → score → optional refine → optional second-pass prompt (see [utils/orchestration.py](../utils/orchestration.py)).
- Optional **lightweight anatomy/physics** checks (hand bbox + finger count heuristics, shadow/light consistency) as *scores* feeding pick-best—not a full closed verifier.

---

## 3. Native high resolution & aspect freedom

**Trend:** Fewer **upscaler-only** workflows; more **native** high-res or wide aspect without stretchy subjects or tiled repetition.

**SDX today**

- `sample.py` **width/height**; **VAE tiling** for large decodes.
- Training still often **single `--image-size`** unless you script buckets externally.

**Good next steps**

- **Resolution / aspect buckets** in training ([IMPROVEMENTS.md](IMPROVEMENTS.md) §1.1).
- Document a **safe** recipe: train at moderate res, sample higher with tiling + light post-sharpen.

### Should we add an upscaler “to the model”?

**Recommendation:** treat upscaling as an **optional post-stage**, not as extra weights fused into the core DiT.

| Approach | Pros | Cons |
|----------|------|------|
| **Separate SR model** (e.g. Real-ESRGAN, SwinIR, commercial APIs) after `sample.py` | Different VRAM profile; can swap vendors; matches Comfy/A1111 workflows | Another dependency; can hallucinate texture if pushed too hard |
| **Train the DiT natively at target res** (buckets, longer training) | Coherent structure at full resolution | More GPU memory and data |
| **Latent upscale + second diffusion pass** (Hi-Res Fix style) | Strong for SD-family pipelines | Extra engineering in *this* repo; not the default SDX path today |

SDX already has **VAE tiling** for large decodes and **sharpen/naturalize** in `utils/quality.py`. A dedicated **upscaler** is still useful for production (4K deliverables from 512²–1024² generators), but it belongs in **docs + optional script** calling an external SR tool, or a small **wrapper**—not necessarily a new trainable head inside `models/dit_text.py`.

---

## 4. Text, UI, and graphic design in the image

**Trend:** Strong models target **readable paragraphs**, **logos**, and **UI mockups** with fewer spelling/layout failures.

**SDX today**

- OCR-aware **pick-best** helps select sharper text-like outputs.
- [MODEL_WEAKNESSES.md](MODEL_WEAKNESSES.md) discusses text-in-image limits.

**Good next steps**

- Training data with **rendered text** and **negative_caption** for common text failures.
- Optional **text-specific** reward during preference or distillation stages (long-term; see §11 in IMPROVEMENTS).

---

## 5. Grounding and “live” knowledge

**Trend:** Products advertise **retrieval** or **live** context (e.g. “what’s on a billboard today”). That requires **external** data and policy—not just a bigger U-Net.

**SDX today**

- Prompt goes through **frozen** encoders; no built-in web access.

**Good next steps**

- **RAG-style** optional module: retrieve reference images/text → inject into caption or cross-attn (design doc + stub).
- Clear **separation** in docs: “model knowledge” vs “user-supplied facts” for safety and reproducibility.

---

## External products (illustrative only)

Commercial stacks and names change quickly. Treat the following as **examples** of the *categories* above, not as SDX integrations:

| Category | Example themes (names may be product-specific) |
|----------|-----------------------------------------------|
| Authentic / lens-like | High-res, low-“plastic” photography style |
| Fast iteration | Low-latency preview models |
| Customization | Strong LoRA / local fine-tuning ecosystems |
| Text + layout | Marketing and UI-heavy generations |

---

## See also

- [IMPROVEMENTS.md](IMPROVEMENTS.md) §11–§12 — research hooks and **2026 alignment** tickets  
- [MODERN_DIFFUSION.md](MODERN_DIFFUSION.md) — timestep sampling and flow-era ideas  
- [utils/orchestration.py](../utils/orchestration.py) — named pipeline roles (Designer / Verifier / Reasoner)
