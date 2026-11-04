# Research notes

Roadmaps, industry snapshots and design notes — **not required** for day-to-day train/sample work. Some notes describe planned or since-merged modules; the current tree is in [FILES.md](FILES.md).

---

## Competitive Analysis & Solution Roadmap

> What the leading text-to-image / video systems do well and badly, how **SDX**
> compares as a **trainable framework**, and concrete architecture-specific
> levers in this repo (`models/`, `diffusion/`, `vit_quality/`, `utils/`,
> `pipelines/video/`, `native/`).
>
> **Snapshot:** mid‑2026 public products and open weights. Names, Elo boards,
> and licenses churn; treat version labels as approximate. Prefer *capability
> classes* over marketing demos. README summary: [../README.md](../README.md#how-sdx-compares).

---

### 0. Landscape research (mid‑2026)

#### 0.1 Closed image leaders — what they can do

| System | What it can do well | What it cannot / will not |
|--------|---------------------|---------------------------|
| **GPT Image 2** (OpenAI) | Strong prompt following, multi-turn **editing**, text-in-image, API automation, multi-element composition | Train/export your weights; air-gap; fork the sampler; preference RL on your pairs; open video director |
| **Midjourney V8.x** | Aesthetic taste, lighting, cinematic / fashion / concept exploration; character refs in-product | Reliable public automation API; literal layout contracts; self-host; fine-tune on proprietary datasets |
| **Gemini 3 · Nano Banana** (Google) | Fast multimodal generate+edit; Pro path for complex refs / 4K-class deliverables | Leave Google stack; local train loop; inspectable critic; open scene-graph video |
| **Ideogram 4** | Typography, posters, layout; **open weights** on the latest line enable LoRA / self-host experiments | Full from-scratch research train OS; in-tree DPO/GRPO; open film studio |
| **Recraft V4** | Design-led brand graphics, icons, vector-ish commercial assets | General photoreal frontier; open training science; video direction |
| **Adobe Firefly 5** | Creative Cloud fit, enterprise licensing narrative | Independent air-gap research stack; preference training lab |
| **Seedream 5** (ByteDance) | Cinematic stills / semantic prompts; useful as video source frames | Ownership of pipeline; reproducible offline science |

**Pattern:** closed APIs optimize for *approved pixels with zero ML ops*. They almost never expose train loops, critic source, or forkable directors.

#### 0.2 Open-weight image leaders — what they can do

| System | What it can do well | Gaps vs a full studio |
|--------|---------------------|------------------------|
| **FLUX.2** (Black Forest Labs) | Open-weight quality bar; photoreal; klein LoRA fine-tunes; partner APIs; multi-ref variants | Mostly **adapter** training, not one opinionated `train.py` science stack; licenses split (dev vs klein); no TCIS/video director in-tree |
| **Stable Diffusion 3.5** | Deepest **LoRA / ControlNet / Comfy** ecosystem; community fine-tunes | Preference RL and closed-loop critics are DIY; video is a different product line |
| **Qwen-Image** | Multilingual / CJK **text-in-image**; Apache-friendly commercial story | Thinner Western tooling vs SD; not a director/critic framework |
| **Z-Image Turbo** | Speed / throughput on mid GPUs | Quality vs frontier tradeoff; small ecosystem |
| **HunyuanImage / others** | Large open MoE-style experiments | Hardware hungry; not a small-lab default |
| **SDXL** (legacy) | Runs everywhere; endless LoRAs | Older U-Net ceiling vs DiT / flow leaders |
| **ComfyUI / A1111 / Forge** | Ultimate **graph UI** over any weights | You assemble science by hand; no shared train→critique→scene-graph product |

**Pattern:** open weights give *inference ownership*. Fine-tune ecosystems (especially SD + Flux LoRA) are strong. What stays rare: **in-tree preference RL + self-critique + open video scene graph** in one readable repo.

#### 0.3 Video leaders — what they can do

| System | What it can do well | Gaps |
|--------|---------------------|------|
| **Veo 3.1** (Google) | Hosted fidelity + **native audio**; strong all-round clips | Closed; cloud; watermarking / ToS; no open director OS |
| **Kling 3** | Motion, multi-shot value, competitive 4K-class paths | Closed API; residency / ToS; not forkable |
| **Runway Gen-4.5** | Motion brush, editor UX, character/control workflows | Hosted product economics; not a training framework |
| **Luma Ray / Seedance** | Atmospheric I2V; multi-shot narrative experiments | Closed; limited scientific reproducibility |
| **Wan / LTX** (open-ish) | Emerging **self-host** video weights | Model-centric; little open continuity / scene-graph film studio |

**Pattern:** video quality crowns sit with hosted labs. Open video is catching up on *weights*, not on *open directing*.

#### 0.4 Cross-product capability matrix

Legend: **✓** strong · **◐** partial · **ext** via community · **✗** no · **n/a** wrong shape

##### Image

| Capability | GPT Image | MJ | Ideogram | FLUX.2 | SD3.5 | Qwen | Comfy | **SDX** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hosted aesthetic defaults | ✓ | **✓** | ◐ | ✓ | ◐ | ◐ | n/a | n/a |
| Text-in-image | **✓** | ◐ | **✓** | ✓ | ◐ | **✓** | ext | ◐ |
| Download weights | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| LoRA fine-tune | ✗ | ✗ | ✓ | ✓ | **✓** | ◐ | **✓** | **✓** |
| Full folder train loop | ✗ | ✗ | ✗ | ◐ | ◐ | ◐ | ext | **✓** |
| DPO / GRPO in-tree | ✗ | ✗ | ✗ | ✗ | ext | ✗ | ext | **✓** |
| Native regional boxes | ◐ | ◐ | ✓ | ext | ext | ext | ext | **✓** |
| Self-critique retry loop | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | **✓** |
| Style Genome–class invention | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Open scene-graph video | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ext | **✓** |
| Air-gap + test provenance | ✗ | ✗ | ◐ | ◐ | ◐ | ◐ | ◐ | **✓** |

##### Video

| Capability | Veo | Kling | Runway | Luma | Wan/LTX | **SDX** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Hosted clip + audio quality | **✓** | **✓** | ✓ | ✓ | ◐ | n/a |
| Editor / motion-brush UX | ◐ | ◐ | **✓** | ◐ | ✗ | ◐ |
| Open JSON scene director | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Continuity validators + retry | ◐ | ◐ | ✓ | ◐ | ✗ | **✓** |
| Forkable train+compose stack | ✗ | ✗ | ✗ | ✗ | ◐ | **✓** |

#### 0.5 What SDX uniquely (or rarely) offers

Verified against this repo’s runnable surfaces (see also code inventory):

1. **`train.py` research loop** — VP + flow matching + LoRA/DoRA on *your* folder, not only hosted LoRA dashboards.
2. **Preference stack** — Diffusion-DPO trainer + Flow-GRPO family scaffolds in-tree.
3. **TCIS / pick-best** — multi-metric critique and retry, not single-shot SaaS generate.
4. **Holy Grail** — adaptive CFG / sampling presets as a first-class tested path.
5. **Style Genome** — invent style axes without artist-name cloning.
6. **Layout stack** — regional boxes, prompt→scene layout, glyph canvas, ControlNet hooks.
7. **Video scene-graph studio** — one JSON → plan → retrieve/keyframe/motion/polish/stitch + continuity / critic modules.
8. **`frontier/` registry** — experimental cinema/narrative/realism modules beside production CLIs.
9. **Compat bridges** — sniff foreign LoRA/LyCORIS toward DiT roles.
10. **Self-host discipline** — air-gap possible; CI + large pytest surface + run metadata.

#### 0.6 What SDX should *not* claim

- Winning Midjourney / Veo aesthetics Elo without a published bake-off.
- Being a drop-in replacement for FLUX.2 or Ideogram OCR demos out of the box.
- Shipping a single mega-checkpoint that “is” GPT Image / Kling.

SDX’s bet: **own the loop** (data → train → sample → critique → direct). Theirs: **own the default pixels**.

#### 0.7 Sources (orientation, not endorsement)

Public mid‑2026 surveys and vendor docs informing §0 (verify before purchasing decisions):

- Teamday — *Best AI Image Models 2026* (job-based ranking; GPT Image 2, MJ V8, Ideogram 4, FLUX.2, Seedream, Firefly, …)
- Thunder Compute / BentoML / Second Talent — open-weight guides (FLUX.2, SD3.5, Qwen-Image, Z-Image, Hunyuan, …)
- BFL docs — FLUX.2 klein LoRA training
- AI video roundups — Veo 3.1, Kling 3, Runway Gen-4.5, Luma, Seedance, Wan/LTX; Sora API sunset messaging in 2026 press
- In-repo: [research/LANDSCAPE_2026.md](#image-generation-landscape-architecture-and-workflows-2026)

---

### 1. The landscape at a glance (legacy early‑2026 notes)

| System | Backbone (public) | Strong at | Weak at |
|---|---|---|---|
| **Midjourney v6/v7** | proprietary | aesthetics, coherence, "it just looks good" | prompt literalness, text rendering, control/repro, licensing |
| **DALL·E 3** | diffusion + heavy LLM prompt rewrite | prompt understanding, text | plastic/uniform look, censorship over-blocks, no fine control |
| **FLUX.1 (dev/pro)** | rectified-flow DiT, dual text enc (T5+CLIP) | prompt adherence, hands, text, open weights | compute cost, aesthetic sameness, weak controllability out of box |
| **SD3.5 Large** | MM-DiT, 3 text encoders | open, controllable, ecosystem | anatomy at distance, prompt drift on long prompts, needs tuning |
| **Ideogram 2/3** | proprietary | **text rendering**, typography, layout | photoreal texture, niche styles |
| **Imagen 3/4** | proprietary | photoreal, lighting, color | access, control, style range |
| **Recraft v3** | proprietary | vector/brand/design ops, text | illustration diversity, community |

The **durable** pattern: models cluster into "looks great but won't do what you
said" (MJ, Imagen) vs "does what you said but looks generic" (DALL·E 3, FLUX).
The open middle — *adherent AND aesthetic AND controllable* — is under-served,
and it is exactly where sdx's DiT-Text + ViT-critic + TCIS design is aimed.

---

### 2. Cross-cutting weaknesses → sdx opportunities

Each item = a weakness shared by most competitors, then the concrete sdx lever.

#### 2.1 Prompt adherence collapses on long / compositional prompts
Everyone degrades as prompts get longer, more relational ("A left of B, holding
C"), or contain counts ("exactly three"). Attention dilutes; late tokens get
ignored.

- **What sdx already has:** `prompt_reinject_every_n`, timestep-aware prompt
  scaling (`prompt_early_scale`/`late_scale`) in `dit_text.py`, TCIS count/OCR
  scoring in `hybrid_dit_vit_generate.py`, and now a **CLIP-conditioned ViT
  adherence head** (see `vit_quality/model.py`).
- **Solution A (evaluator → generator feedback):** the ViT adherence head now
  sees real prompt semantics. Wire its *per-region* disagreement back into TCIS
  as a spatial reweighting signal, not just a scalar accept/reject. Land in
  `utils/agentic/quality_loop.py` + `hybrid_dit_vit_generate.py`.
- **Solution B (structured prompts):** add a lightweight prompt→scene-graph
  parser (subject / relation / attribute / count) feeding regional conditioning
  (`utils/generation/regional.py` already exists). This is a
  *small tool*, not a model — see §4.
- **Solution C (count adherence):** competitors have no count guarantee. sdx can
  close-loop it: generate → native object-count metric → resample the failing
  region. The counter belongs in `native/` (fast, deterministic) — see §5.

#### 2.2 Text-in-image rendering (only Ideogram/Recraft nail it)
Most diffusion models still smear multi-word text.

- **Solution:** sdx already has an OCR term in TCIS consensus. Promote it to a
  *first-class glyph-conditioning* path: render the requested string to a mask
  and inject via the existing ControlNet stack (`control_image`/`control_type`
  in `dit_text.py`), then verify with the OCR score and iterate. No new backbone
  needed — it reuses control + critic infra.

#### 2.3 "House style" sameness (MJ, DALL·E, Imagen all have a detectable look)
Their strength is also a ceiling: outputs are recognizable and hard to push off.

- **Solution:** sdx's `creativity_embed_dim` + `style_embed_dim` + MoE experts
  are a genuine differentiator. Formalize a **style-dispersion objective**:
  during preference mining (`utils/agentic/quality_loop.py`), reward *intra-prompt
  diversity* that stays on-prompt (ViT adherence high, pairwise embedding
  distance high). This directly attacks the sameness competitors can't escape.

#### 2.4 Controllability is bolted on, not native
FLUX/SD control needs external ControlNets/LoRAs; MJ/DALL·E barely expose it.

- **What sdx has:** native ControlNet injection, REPA, AR blocks, RAE latents.
- **Solution:** lean into this as the *product wedge*. Ship a single
  "control bundle" API (pose + depth + glyph + region + count) that competitors
  can't match without a plugin zoo. The pieces exist; the gap is a unified,
  documented surface.

#### 2.5 No self-critique loop (all competitors are single-shot)
None of the majors verify their own output against the prompt and retry. This is
sdx's biggest structural edge.

- **What sdx has:** TCIS (`docs/HOW_GENERATION_WORKS.md`) + ViT critic + auto/flywheel loops.
- **Solution:** make the critic *cheaper and better* so the loop can run more
  iterations per second: (a) the CLIP-semantic adherence head (done), (b) native
  scoring kernels for the non-neural TCIS terms (count/saturation/diversity) so
  the consensus math isn't a Python bottleneck (§5), (c) uncertainty-triggered
  early-exit (already partially present) tuned against the new head.

#### 2.6 Anatomy / small-face / far-figure degradation
Shared weakness (SD3.5 especially) at distance and in crowds.

- **Solution:** multi-crop high-res ViT scoring. Today `vit_quality` resizes
  everything to 224 and `book_model_readiness.py` already warns about it. Add a
  tiled/multi-crop scoring path so the critic actually *sees* small faces, then
  let TCIS resample crops that fail. Pick-time scoring already tiles aspect-preserving
  crops (`test_time_pick._vit_views`); manifest scoring (`vit_quality/infer.py`) still squashes.

---

### 3. Where sdx is already ahead — protect and sharpen

1. **Closed-loop critic (TCIS).** No major competitor self-verifies. Keep
   investing here; it compounds.
2. **Separable adherence vs quality heads.** Competitors conflate "good" and
   "on-prompt"; sdx scores them independently — enables targeted fixes.
3. **Native-accel discipline.** `native/rust/` + ctypes/NumPy-fallback is a real
   moat for loop throughput. Extend it (§5).
4. **AR-regime awareness.** ViT↔DiT AR calibration (`ar_block_conditioning.py`)
   is unusually principled; keep the evaluator and generator regimes in sync.

---

### 4. Small tools to build (Python, "not huge") — model-improving utilities

These improve the *model via better data/eval/loops*, not by shipping big weights.

| Tool | Purpose | Home |
|---|---|---|
| **Prompt scene-graph parser** | subject/relation/attribute/count → regional cond | `utils/prompt/` |
| **Caption↔image alignment auditor** | flags mislabeled training pairs using the CLIP featurizer already added | `scripts/tools/manifest.py` |
| **Adherence-failure miner** | runs the CLIP-ViT head over a manifest, buckets the worst prompts by failure type (count/relation/text) → targeted training set | `vit_quality/` |
| **Diversity/duplication filter** | near-dup detection on training data via embedding + pHash to fight style collapse | `scripts/tools/manifest.py` |
| **Eval harness vs competitors** | fixed prompt suite scoring adherence/text/count/anatomy so we track progress on *their* weak points | `scripts/tools/` |

Every one of these is bounded, testable, and reuses infra that now exists.

---

### 5. Native (Rust) speedups — real hotspots, small crates

The `native/rust/` pattern (C-ABI cdylib + ctypes wrapper + NumPy fallback,
see `docs/MODEL_STACK.md`) is the right home. Candidates, in rough
value order — **each should be profiled first**:

1. **TCIS consensus kernels** — the per-candidate count/saturation/diversity/
   Pareto math runs every loop iteration in Python. A batched Rust kernel would
   cut critic latency, letting the loop do more refinement per wall-second. This
   is the highest-leverage native win because it multiplies the closed-loop edge.
2. **Object/blob counter** — deterministic connected-components + size filter for
   count adherence (§2.1-C). Pure integer image work → ideal for Rust, useless as
   a big model.
3. **pHash / near-dup** for the diversity filter (§4) — hash + Hamming over large
   manifests; trivially parallel, GIL-free.
4. **Fast manifest ops** — extend `sdx-jsonl-tools` for the alignment auditor's
   streaming passes over million-line manifests.

Non-goals: do **not** rewrite the neural forward path in Rust — PyTorch/CUDA
already owns that. Native wins live in the *glue and metrics* around the model,
which is where the Python loop actually spends its non-GPU time.

---

### 6. Suggested sequencing

1. **Measure**: stand up the eval harness (§4) on competitors' weak points so
   every later change is scored, not vibed.
2. **Cheapen the loop**: native TCIS kernels + object counter (§5.1–5.2) so the
   critic edge can run harder.
3. **Feed critic → generator**: per-region adherence reweighting (§2.1-A) and
   glyph conditioning (§2.2).
4. **Attack sameness**: diversity objective in preference mining (§2.3).
5. **Data hygiene**: alignment auditor + dedup (§4) to stop training in the
   failure modes we're trying to beat.

The through-line: sdx's structural advantage is the **self-critique loop**.
Every recommendation either makes that loop *cheaper* (native), *smarter*
(semantic critic, per-region feedback), or *better-fed* (data tools). That is
the axis on which none of the single-shot incumbents can easily follow.

---

## Image generation landscape, architecture, and workflows (2026)

**Merged doc:** industry trends, post-diffusion research themes, and workflow/efficiency commentary (previously three separate files). Jump: [Industry snapshot](#industry-snapshot-march-2026) · [Post-diffusion shifts](#post-diffusion-and-structural-shifts-late-20252026) · [Workflow integration](#workflow-integration-march-2026).

**Context:** Text-to-image is increasingly **production infrastructure**, not only a “cool art” demo. Teams care about **functional precision**, **grounding**, and **outputs that read as real**—not just high scores on aesthetics.

This doc **does not** claim SDX ships commercial parity with any external product. It **does** map where the field is moving and how SDX can evolve (see also [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) §12, [utils/agentic/agent.py](../utils/agentic/agent.py)).

**Note:** Product names, release dates, and benchmarks change quickly. Treat the **industry snapshot** below as an orientation to common themes in early-2026 discussions—not a live leaderboard.

---

### Industry snapshot (March 2026)

The field has shifted from “single text-to-image box” toward **integrated creative systems**: **4K-class** outputs as a default expectation, **reliable text-in-image** (glyphs, posters, UI), **Diffusion Transformers (DiT)** over older U-Net stacks for scalability, and **editing/consistency** as first-class (not one-off img2img hacks).

#### Frontier models (closed / API-class)

| Theme | Often-cited examples (illustrative) |
| :--- | :--- |
| **Speed + production polish** | **FLUX.1.1 Pro** (fast, high-fidelity previews), **FLUX.2** family (strong “production-grade” look; reduced “plastic” skin vs earlier gens) |
| **Visual storytelling / lighting** | **MAI-Image-2** (Microsoft) — emphasis on skin tone, natural light, cinematic composition |
| **Spatial reasoning & typography** | **GPT Image 1.5** (OpenAI ecosystem) — faster than prior DALL·E-era stacks; better object contact / non-floating props; strong multi-line text |
| **Text-in-image (glyphs)** | **Imagen 3** (Google) — dedicated glyph / text paths for readable signs, posters, labels |
| **Google / Gemini image stack** | **Nano Banana** line (e.g. Pro / flash variants tied to Gemini) — often highlighted for consistency, text, and editing speed |
| **Artistic breadth** | **Midjourney v7** (2025+) — still a reference for stylized / aesthetic-led work |

#### Open-weight & community

| Theme | Often-cited examples (illustrative) |
| :--- | :--- |
| **Large open weights, complex prompts** | **Qwen-Image-2512** (Alibaba) — very large open-weight stack; strong on long, precise prompts (colors, layout) |
| **Real-time / distilled** | **Z-Image-Turbo** — sub-second class on mid-range GPUs; popular for live UI and streaming |
| **Unified image + video** | **LTX** family — shared “unified diffusion” story across stills and short clips; style parity stills ↔ motion |
| **Flexible OSS base** | **Stable Diffusion 3.5+** / **SD3 lineage** — MMDiT-style text and prompt following; large fine-tune ecosystem |
| **FLUX ecosystem** | Open-weight **FLUX** variants + **Kontext**-style **contextual editing** (iterate clothes/background while holding identity) |
| **Versatile commercial-style OSS** | **Seedream** (various versions) — multi-scenario, multi-style handling |

#### Architecture: U-Net → DiT, Flow Matching, editing

- **DiT over U-Net:** Industry direction favors **transformer backbones** for diffusion at scale—better scaling to **high resolution** (4K/8K) without classic CNN upsampling artifacts (doubled limbs, tiled seams).
- **Flow matching / context:** Some stacks (e.g. FLUX **Kontext**) lean on **flow-matching**-style training and **context-aware** conditioning so **iterative edits** preserve identity and scene structure.
- **SDX alignment:** This repo is **DiT-centric** (`GaussianDiffusion` + DiT; see `train.py` / `sample.py`)—aligned with the architectural trend, independent of any vendor API.

#### Cross-cutting trends (early 2026)

| Trend | What it means |
| :--- | :--- |
| **Authenticity / “lo-fi”** | Deliberate **film grain**, **lens flare**, **motion blur** to avoid the “too perfect” look — see [§1](#1-authenticity-over-the-ai-look) below. |
| **Text & graphic design** | Legible **paragraphs**, **logos**, **UI mockups** — see [§4](#4-text-ui-and-graphic-design-in-the-image). |
| **Subject / character consistency** | **Character locking** and multi-scene identity without hand-tuned LoRA for every project. |
| **Grounding & freshness** | **Retrieval / live context** (e.g. current products, outfits) — still mostly **product-layer**, not “baked into” weights alone — see [§5](#5-grounding-and-live-knowledge). |
| **Speed vs quality** | High-res in **seconds** is normal; distilled and “turbo” paths target **interactive** loops. |

---

### 1. Authenticity over “the AI look”

**Trend:** Audiences are good at spotting **plastic** skin, **over-perfect** symmetry, and **HDR mush**. Newer systems often train or sample toward **lens-like** imperfections: micro-texture, plausible noise, asymmetric framing, natural light falloff.

**SDX today**

- Training/inference knobs that reduce **CFG blowout** and oversaturation: CFG rescale, dynamic threshold, Min-SNR, noise offset ([QUALITY.md](QUALITY.md)).
- **Refinement** and post-process (`utils/quality/finishing.py` naturalize/sharpen).
- **Domain / style** tooling ([STYLE_ARTIST_TAGS.md](PROMPT_STACK.md#style-and-artist-tags-pixai-danbooru-gelbooru), [DOMAINS.md](PROMPT_STACK.md#generating-3d-realistic-interiorexterior-and-other-hard-domains)).

**Good next steps in-repo**

- Optional **texture / frequency** regularization or auxiliary losses (document loss schedule; see [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) §11).
- **Multi-resolution / aspect buckets** — **implemented:** `train.py --resolution-buckets` (e.g. `256,512` or `512x768`), [`data/t2i_dataset.py`](../data/t2i_dataset.py), [`data/t2i_dataset.py`](../data/t2i_dataset.py) (single-GPU only; no `--val-split`; latent cache disabled when buckets are on). Prefer `--size-embed-dim` > 0 for best extrapolation ([IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) §1.1).
- **Capture authenticity** — **implemented:** physically based lens/sensor/ISP and film-grain finish in [`camera_isp.py`](../utils/quality/finishing.py), routed by medium.
- Dataset-side **“lived-in”** captions (wear, dust, film grain) in JSONL—not only tag lists.

---

### 2. Multi-component “system of experts”

**Trend:** One monolith is replaced by a **pipeline**: e.g. a **layout/composition** stage, a **verification** stage (anatomy, physics, consistency), and a **reasoning** stage (complex instructions, ordering, negation).

**SDX today**

- **Designer-like core:** `sample.py` + DiT + VAE (and optional Control / regional captions).
- **Verifier-like hooks:** `--pick-best` with CLIP / edge / OCR ([utils/quality/test_time_pick.py](../utils/quality/test_time_pick.py)); ViT quality tools under `ViT/`.
- **Reasoner-like conditioning:** T5 (+ optional triple fusion); optional LLM prompt expansion ([utils/agentic/agent.py](../utils/agentic/agent.py)).

**Good next steps**

- Explicit **orchestration** — **implemented:** [`scripts/tools/generate.py`](../scripts/tools/generate.py) (wraps `sample.py` with `--num` + `--pick-best`); optional **`--pick-best combo_exposure`** ([`score_exposure_balance`](../utils/quality/test_time_pick.py)) for lightweight highlight/shadow clipping avoidance.
- Optional **lightweight anatomy/physics** checks (hand bbox + finger count heuristics, shadow/light consistency) as *scores* feeding pick-best—not a full closed verifier.

---

### 3. Native high resolution & aspect freedom

**Trend:** Fewer **upscaler-only** workflows; more **native** high-res or wide aspect without stretchy subjects or tiled repetition.

**SDX today**

- `sample.py` **width/height**; **VAE tiling** for large decodes.
- Training still often **single `--image-size`** unless you script buckets externally.

**Good next steps**

- **Resolution / aspect buckets** in training — see §1 above (`--resolution-buckets`).
- Document a **safe** recipe: train at moderate res, sample higher with tiling + light post-sharpen.

#### Should we add an upscaler “to the model”?

**Recommendation:** treat upscaling as an **optional post-stage**, not as extra weights fused into the core DiT.

| Approach | Pros | Cons |
|----------|------|------|
| **Separate SR model** (e.g. Real-ESRGAN, SwinIR, commercial APIs) after `sample.py` | Different VRAM profile; can swap vendors; matches Comfy/A1111 workflows | Another dependency; can hallucinate texture if pushed too hard |
| **Train the DiT natively at target res** (buckets, longer training) | Coherent structure at full resolution | More GPU memory and data |
| **Latent upscale + second diffusion pass** (Hi-Res Fix style) | Strong for SD-family pipelines | Extra engineering in *this* repo; not the default SDX path today |

SDX already has **VAE tiling** for large decodes and **sharpen/naturalize** in `utils/quality/finishing.py`. A dedicated **upscaler** is still useful for production (4K deliverables from 512²–1024² generators), but it belongs in **docs + optional script** calling an external SR tool, or a small **wrapper**—not necessarily a new trainable head inside `models/dit_text.py`.

---

### 4. Text, UI, and graphic design in the image

**Trend:** Strong models target **readable paragraphs**, **logos**, and **UI mockups** with fewer spelling/layout failures.

**SDX today**

- OCR-aware **pick-best** helps select sharper text-like outputs.
- [MODEL_WEAKNESSES.md](QUALITY.md) discusses text-in-image limits.

**Good next steps**

- Training data with **rendered text** and **negative_caption** for common text failures.
- Optional **text-specific** reward during preference or distillation stages (long-term; see §11 in IMPROVEMENTS).

---

### 5. Grounding and “live” knowledge

**Trend:** Products advertise **retrieval** or **live** context (e.g. “what’s on a billboard today”). That requires **external** data and policy—not just a bigger U-Net.

**SDX today**

- Prompt goes through **frozen** encoders; no built-in web access.

**Good next steps**

- **RAG-style** optional module — **stub:** [`utils/prompt/rag.py`](../utils/prompt/rag.py) (`merge_facts_into_prompt`, `load_facts_from_jsonl`). Wire your own retrieval; then **encode** the merged prompt as usual.
- **Character / subject consistency** — **helpers:** [`utils/consistency.py`](../utils/consistency.py) (`build_character_prompt_additions`) turns a character sheet into stable prompt/negative additions (`sample.py --character-sheet`).
- Clear **separation** in docs: “model knowledge” vs “user-supplied facts” for safety and reproducibility.

---

### External products (illustrative only)

Commercial stacks and names change quickly. Treat the following as **examples** of the *categories* above, not as SDX integrations:

| Category | Example themes (names may be product-specific) |
|----------|-----------------------------------------------|
| Authentic / lens-like | High-res, low-“plastic” photography style |
| Fast iteration | Low-latency preview models |
| Customization | Strong LoRA / local fine-tuning ecosystems |
| Text + layout | Marketing and UI-heavy generations |

---

---

### Post-diffusion and structural shifts (late 2025–2026)

**Context:** The field has moved past the first wave of “DiT replaces U-Net” hype into **deeper mathematical and systems refinement**. For a machine-readable map see [`utils/modeling/architecture.py`](../utils/modeling/architecture.py). For a “four-pillar” framing see [NEXTGEN_SUPERMODEL_ARCHITECTURE.md](#next-gen-image-generation-four-design-pillars-and-sdx-mapping).

#### 1. Beyond pure diffusion: Flow Matching & diffusion bridges

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **Flow Matching (FM)** | Learn **straight-line** (or near-straight) transport from noise to data instead of highly **curved stochastic** VP paths. | Often **more stable training** and **fewer sampling steps** than classic DDPM schedules. |
| **Rectified flow / OT** | Couples noise and data with **optimal-transport**-style couplings; velocity field along geodesic-like paths. | Common in **latent** generators (e.g. FLUX-class stacks in industry talks). |
| **Diffusion bridges** | Transition between **two** distributions (not only **noise → image**): sketch→render, day→night, edit paths that **preserve structure**. | Powers **unified edit + gen** without throwing away the source layout. |

**SDX today:** VP DDPM + `GaussianDiffusion` + DiT; **v-prediction** (`--prediction-type v`), **non-uniform timestep sampling** (`diffusion/schedules.py`), and [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) discuss flow-adjacent ideas — **full rectified-flow training is not a drop-in** (different objective/sampler). **Status:** partial / research alignment.

---

#### 2. Hybrid AR + Diffusion (LLM-style planning + denoiser)

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **AR “planner” + diffusion decoder** | **Autoregressive** module outputs **coarse semantic tokens** (layout, relations); **DiT / diffusion** refines **texture and detail**. | Addresses **spatial logic** (“X under Y”), **long text**, and **instruction-heavy** prompts. |
| **Block-causal / raster AR in DiT** | Causal attention over **patches or blocks** so generation has a **sequence** bias without a separate tokenizer stack. | Lighter-weight hybrid than a full 9B AR image tokenizer — **controllable** in open-source. |

**SDX today:** **`num_ar_blocks`** (0 / 2 / 4 / …), block masks in `models/attention.py`, [docs/MODEL_STACK.md](MODEL_STACK.md#block-wise-autoregressive-ar-generation), **`utils/modeling/architecture.py`** for ViT alignment. This is **one** credible open-source angle on “AR + diffusion” — not a full GLM-Image-style discrete token LM. **Status:** partial (DiT-native AR, not separate AR tokenizer).

---

#### 3. State space models (Mamba) vs Vision Transformers

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **SSM / Mamba** | **Subquadratic** sequence models — **linear** scaling in length vs **quadratic** self-attention. | Tempting for **8K+** or **long video** with **VRAM** limits. |
| **Vision Mamba, U-Mamba** | Replace or augment **ViT** backbones in vision encoders or decoders. | Still **research-heavy**; fewer mature T2I checkpoints than DiT. |

**SDX today:** DiT uses **transformer** attention (optional **xformers**, **RoPE**, **token routing**, etc.). **No Mamba backbone** in the repo. **Status:** not implemented; listed in [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) as long-term architecture exploration.

---

#### 4. One-step & few-step distillation (DMD, consistency, turbo)

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **Distribution Matching Distillation (DMD)** | Distill a **teacher** diffusion into a **fast** student (sometimes **1 step**), combining ideas from **GANs** + **diffusion**. | **Real-time** UI, “generate as you type,” consumer GPUs. |
| **Consistency / CM** | Enforce **trajectory consistency** for few-step sampling. | Alternative path to turbo inference. |

**SDX today:** **DDIM-style** multi-step sampling; **no** built-in DMD/consistency trainer. **Test-time:** `--num K --pick-best` ([`utils/quality/test_time_pick.py`](../utils/quality/test_time_pick.py)) as a **quality** gate, not a distilled student. **Status:** not in-repo training; roadmap in [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) §2.4 / §11.

---

#### 5. Semantic & scientific grounding

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **Physics-informed / physical consistency** | Bias generations toward **plausible** lighting, gravity, contact — **not only** “looks plausible.” | Strong in **video / 3D**; emerging in **still** image stacks via **VLM** or **auxiliary losses**. |
| **Scientific imaging** | Diffusion in **non-photographic** spaces (e.g. molecular, material) — **noise** = **uncertainty**, not pixels. | Different product line than **SDX**; included for **ecosystem** awareness. |
| **RAG / facts in prompt** | **User-supplied** facts merged into the prompt before encoding — [`utils/prompt/rag.py`](../utils/prompt/rag.py). | **Grounding** without retraining. |

**SDX today:** **T5 (+ optional CLIP fusion)**; **REPA** aligns features to **frozen vision** encoders; **RAE** path + `RAELatentBridge` for non–4ch latents; **no** built-in physics simulator. **Status:** partial (semantic alignment / RAE / REPA), not physics engine.

---

#### 6. Representation autoencoders + DiT (NYU-style line)

| Concept | Idea | Why it matters |
| :--- | :--- | :--- |
| **RAE** | **Semantic** latents from **frozen** encoders + learned decoder; **DiT** denoises in that space. | **Richer** than “compress pixels only” VAEs. |

**SDX today:** **`--autoencoder-type rae`**, `RAELatentBridge`, tr/inference paths — see [MODEL_STACK.md](MODEL_STACK.md), README. **Status:** implemented (RAE path). **REPA** ([`--repa-weight`](../train.py)) aligns with **external** vision reps — **related** to “semantic-first latents.”

---

#### Summary: 2023–2024 vs 2026 (illustrative)

| Feature | Old way (2023–2024) | New way (2026 discussions) |
| :--- | :--- | :--- |
| **Backbone** | U-Net | **DiT** / **Mamba** (experimental) |
| **Logic** | Pure diffusion | **Hybrid** (AR planning + diffusion decoder), **VLM** conditioning |
| **Inference** | 20–50 steps | **1–8** steps (distilled / flow-matched) |
| **Pathing** | Stochastic / curved VP | **Flow matching**, **bridges**, bridges for **edit** |
| **Resolution** | 1024² + upscaler | **Native** multi-megapixel / **4K-class** in product stacks |
| **Latent** | “Pixel compression” VAE | **Semantic** / **RAE** / **alignment** (REPA) |

---

#### Product-scale stacks (examples only)

| Line | Themes (illustrative) | SDX analogy |
| :--- | :--- | :--- |
| **Hybrid AR + DiT** (e.g. industry “AR planner + DiT decoder”) | Discrete tokens → refine | **Block AR** + DiT; **no** separate 9B AR image LM in-repo |
| **Latent flow + VLM** (e.g. FLUX-class) | Flow + VLM + new VAE | **Flow** not native; **triple** text encoders + **T5**; **VAE** from diffusers |
| **RAE + DiT** (research) | Semantic latents | **`autoencoder_type=rae`** + bridge |

---

---

### Workflow integration (March 2026)

### Disclaimers

| | |
| :--- | :--- |
| **Medical** | This page is **informational only**. It is **not** medical advice, diagnosis, or treatment. For health decisions, **consult a licensed professional**. |
| **Technical / legal** | **Product and assistant names** in industry commentary are **illustrative**; capabilities and benchmarks **change frequently**. SDX does **not** integrate third-party APIs or weights unless explicitly documented. |

---

#### Context

As of **early 2026**, much of the field’s energy has shifted from “make prettier pixels” to **workflow integration** (fits real pipelines) and **mathematical efficiency** (fewer steps, better scaling, less waste). Below are **themes** often discussed in that context — with a **straight mapping to SDX** where applicable ([`utils/modeling/architecture.py`](../utils/modeling/architecture.py), extended rows `workflow_*` / `llada_*` / `test_time_*` / `live_grounding_*`).

---

#### 1. High coherency & “structural” understanding (illustrative product stacks)

**Talking points in industry commentary**

- **Reverse-engineering / 3D-style logic:** Some stacks claim stronger **consistency** under **viewpoint or pose** changes (e.g. “side view” vs “above”) without identity or clothing **melting** — often via **better conditioning**, **multi-view data**, or **internal 3D/pose** priors (details vary by vendor).
- **4K native:** **Single-pass** high resolution vs **upscale-only** workflows — shifts professional photo and arch-viz pipelines.

**SDX**

- **Native resolution:** `sample.py` **width/height**, **VAE tiling** for large decodes; training **`--resolution-buckets`** ([`data/t2i_dataset.py`](../data/t2i_dataset.py)) for multi-aspect / multi-res **single-GPU** training.
- **Multi-angle consistency:** **Not** a built-in 3D/pose engine; use **data + captions**, **Control**-style paths where supported, **character sheet** helpers ([`utils/consistency.py`](../utils/consistency.py)), and **[`docs/RESEARCH.md#industry-snapshot-march-2026`](#industry-snapshot-march-2026)** for consistency themes.

---

#### 2. Discrete diffusion & text (e.g. “LLaDA”-class ideas)

**Concept**

- **Classic LLMs:** often **autoregressive** (one token after another).
- **Classic image models:** often **diffusion** in latent or pixel space.
- **Unified / “LLaDA”-style narratives:** **diffusion-style** processes applied to **text** or **shared** weights across modalities — **whole-sequence** reasoning vs strict left-to-right only.

**Why it matters**

- Drives interest in **one stack** for **prompt reasoning** + **image generation** (tighter **adherence** in complex instructions).

**SDX**

- **Text:** **T5** (+ optional **CLIP fusion**) — **autoregressive** text encoding, **not** discrete diffusion over text tokens in this repo.
- **Image:** **DiT** + **VP diffusion** — see [`docs/RESEARCH.md`](#modern-diffusion--flow-ideas-20242026), [`docs/RESEARCH.md#post-diffusion-and-structural-shifts-late-20252026`](#post-diffusion-and-structural-shifts-late-20252026).

---

#### 3. Test-time compute (“think before paint”)

**Concept**

- **Inference scaling:** extra **compute at sample time** (e.g. internal **critique** / **refine** loops), inspired by **reasoning**-style LLM narratives — **layout vs anatomy** checks before finalizing.

**SDX (partial)**

- **Refinement** pass in **`sample.py`** (small **re-noise** + **denoise**).
- **Multi-candidate + selection:** `--num K` + **`--pick-best`** ([`utils/quality/test_time_pick.py`](../utils/quality/test_time_pick.py)) — e.g. `clip`, `edge`, `ocr`, `combo`, **`combo_exposure`**.
- **Pipeline wrapper:** [`scripts/tools/generate.py`](../scripts/tools/generate.py), [`utils/agentic/agent.py`](../utils/agentic/agent.py).

This is **not** a full internal **latent self-critique** loop like a closed proprietary stack — it’s **explicit** test-time **scoring** and **optional** refinement.

---

#### 4. Real-time grounding & RAG-for-images

**Concept**

- **Stale training data:** mitigated by **retrieval** — **facts** or **references** injected at **generation time**.
- **“Live web”** narratives: **product-layer** integrations (policy, safety, licensing) — **not** “just bigger weights.”

**SDX (partial)**

- **Merge user facts into the prompt:** [`utils/prompt/rag.py`](../utils/prompt/rag.py) (`merge_facts_into_prompt`, `load_facts_from_jsonl`). **You** supply retrieval; the model sees **frozen** encoders after that.
- **No** built-in web browser or API in core `train.py` / `sample.py`.

---

#### 5. Mamba / SSM backbones (linear scaling)

**Concept**

- **Transformers:** **quadratic** cost in sequence length — painful at **8K+** or long **video**.
- **State-space models (e.g. Mamba):** **subquadratic** / **linear** scaling narratives — **candidate** backbones for **vision** encoders or decoders.

**SDX**

- **DiT** uses **transformer** attention (see [`docs/RESEARCH.md#post-diffusion-and-structural-shifts-late-20252026`](#post-diffusion-and-structural-shifts-late-20252026)). **No Mamba** backbone in-tree **today**.

---

---

### See also

- [BLUEPRINTS.md](#research-blueprints-fast-generation--prompt-accuracy) — flow/solvers/distillation + prompt-accuracy ideas vs SDX
- [NEXTGEN_SUPERMODEL_ARCHITECTURE.md](#next-gen-image-generation-four-design-pillars-and-sdx-mapping) — planner / critic / Fourier / DPO pillars
- [`utils/modeling/architecture.py`](../utils/modeling/architecture.py) — theme → repo status
- [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) §11–§12 — research hooks and alignment
- [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) — timestep sampling and flow-era ideas
- [utils/agentic/agent.py](../utils/agentic/agent.py) — Reasoner → Designer → Verifier role pipeline

---

## Modern diffusion & flow ideas (2024–2026)

Short survey of **recent directions** and how they relate to **SDX** (VP DDPM + DiT + `GaussianDiffusion`).

For a **prioritized, compounding** list of upgrades (latent stack, conditioning, objectives, distillation, alignment) mapped to repo paths, see **[DIFFUSION_LEVERAGE_ROADMAP.md](#diffusion-stack-high-leverage-upgrades-10-class-levers)**.

For **2026 industry + architecture + workflow** context (merged), see **[LANDSCAPE_2026.md](#image-generation-landscape-architecture-and-workflows-2026)** and **[`utils/modeling/architecture.py`](../utils/modeling/architecture.py)**. For **few-step math** and **prompt-accuracy** blueprints in one place, see **[BLUEPRINTS.md](#research-blueprints-fast-generation--prompt-accuracy)** (Part 1: flow/solvers/distillation; Part 2: GLS, critics, frequency).

---

### ε-prediction, v-prediction, x₀, and flow paradigms

Many **new flagship** text-to-image stacks (e.g. SD3-class and FLUX-class lines, naming varies by release) advertise **flow matching** or **rectified flow**: learn a **vector field** that transports noise toward data along a path that is often **straighter** than classic VP-DDPM, then integrate with an ODE-style solver. That is **not the same** as switching `--prediction-type v` on a **VP** schedule—SD2-style **v** is still a reparameterization of the **same VP forward process** and the same DDIM-style family of solvers.

Rough map (examples in the wild are illustrative; product details differ by checkpoint):

| Paradigm | Typical idea | Step count (rule of thumb) | Often associated with | **SDX today** |
|:---------|:-------------|:----------------------------|:----------------------|:----------------|
| **ε-prediction** | Predict Gaussian noise in the VP forward | Often 20–50+ with strided samplers | SD 1.x, many SDXL ckpts | `--prediction-type epsilon` (default) |
| **v-prediction** | Velocity-style target tied to VP | Same solver class as ε | SD 2.x, many community finetunes | `--prediction-type v` |
| **x₀ / “data” prediction** | Predict clean latent \(x_0\) | Same VP solvers after conversion | Less common as sole T2I objective; restoration / strong img2img priors | `--prediction-type x0` (**VP only**, not flow) |
| **Flow matching** | Learn field along a **chosen** noise→data path; continuous-time view | Often **fewer** steps when **training + solver** match | SD3-class, FLUX-class (high level) | **Not** implemented as FM training; VP `GaussianDiffusion` + DDIM-style loop |
| **Rectified flow / reflow** | Straighten paths; often + **distillation** for few-step | Turbo / “1–4 step” students | InstaFlow, many Lightning/Turbo variants | **Not** implemented; needs new objective and usually a student stage |

**Caveats**

- **Fewer steps** for FM/RF in papers and products usually requires training **and** a **matched** integrator—not merely lowering `--steps` on a VP-ε or VP-v checkpoint.
- **x₀ prediction in VP** can be **ill-conditioned at very high noise** (the clean signal is weak); **min-SNR** and **non-uniform `t`** help but do not turn VP into flow matching.
- Treat forum “industry standard” claims as **directional**: VP + DiT + good data is still a dominant open-source path; FM/RF is a **parallel** design space worth understanding when reading SD3/FLUX-era docs.

#### Spectral Flow Prediction (SFP) — prototype in this repo

**SFP** here means a **frequency-weighted training loss** on the VP-DDPM target (ε, v, or x₀): take `FFT(pred − target)` on the latent grid, weight radial frequency bins by timestep, and minimize weighted energy. At **high noise** (large diffusion index `t`), weights emphasize **low** spatial frequencies (global layout); near **clean** (`t` small), weights shift toward **high** frequencies (texture).

This is **not** continuous **Flow Matching** (no new noise→data ODE, no FLUX-style integrator). The forward process and **sampling** code are unchanged; only the **loss** may differ from spatial MSE.

| Item | Detail |
|:-----|:-------|
| **Code** | `diffusion/flow_matching.py`, `GaussianDiffusion.training_losses(..., use_spectral_sfp_loss=True)` |
| **Train CLI** | `--spectral-sfp-loss` · `--spectral-sfp-low-sigma` · `--spectral-sfp-high-sigma` · `--spectral-sfp-tau-power` |
| **MDM** | Masked diffusion training still uses **spatial** MSE (spectral loss is not mixed with patch masks in this prototype). |
| **Variable step speed per band** | **Not implemented** — inference step count is unchanged; the doc narrative about “2–3 steps for low freq” would need a **multi-rate / multi-branch** sampler, not loss weighting alone. |

Treat SFP as an **experimental** knob; ablate against plain MSE on your data before trusting it for quality claims.

**Combining SFP with x₀ prediction:** use `--prediction-type x0` and `--spectral-sfp-loss` together. The network still predicts **clean latent** \(x_0\); the loss is no longer uniform spatial MSE on \((\hat x_0 - x_0)\) but **frequency-weighted** energy of \(\mathcal{F}(\hat x_0 - x_0)\). At high noise timesteps, low spatial frequencies of that error dominate the objective; near clean timesteps, high frequencies dominate—on the **x₀ error**, not on ε or v.

---

### Implemented in SDX (this repo)

| Idea | Where | Notes |
|:-----|:------|:------|
| **Logit-normal training timesteps** | `diffusion/schedules.py`, `train.py --timestep-sample-mode logit_normal` | Discrete-time analogue of SD3-style sampling on a normalized axis; emphasizes some noise levels vs uniform `randint`. |
| **High-noise emphasis** | `--timestep-sample-mode high_noise` | `Beta(2,1)` on normalized time → more samples at **large** `t`. |
| Min-SNR loss weighting | Training | Stabilizes across timesteps. |
| V-prediction | `--prediction-type v` | Velocity parameterization (SD2-style). |
| **x0 prediction** | `--prediction-type x0` | Predicts clean latent `x_0` under **VP-DDPM** (not flow matching). DDIM path converts output to implied noise. **Not** compatible with ε/v checkpoints. At **very high noise** the target is hard to infer—use min-SNR + sensible `timestep_sample_mode`; consider **v** or **ε** if training is unstable. |
| CFG rescale / dynamic threshold | `sample.py` | High-CFG friendly. |
| Cosine **beta** schedule | `--beta-schedule cosine` | Alternative noise schedule (not the same as “cosmap” loss weighting in SD3). |
| **Sigmoid** / **squaredcos_cap_v2** schedules | `--beta-schedule sigmoid` or `squaredcos_cap_v2` | See `diffusion/schedules.py` (diffusers-style squared cosine + smooth sigmoid ramp). |
| **Soft min-SNR** loss weight | `--loss-weighting min_snr_soft` | Smooth `gamma/(snr+gamma)` weight (see `diffusion/schedules.py`); alternative to hard `min_snr`. All weightings are defined on the x0 error and converted per `--prediction-type`. |
| **Zero terminal SNR** (Lin et al. 2024) | `--zero-terminal-snr` (schedule suffix `_ztsnr`) | Rescales any beta schedule so `abar_T = 0`; needs v or x0 prediction. Sampling starts at `t = T-1` (`euler` spacing). |
| **SD3 flow recipe** | `--flow-matching-training --timestep-sample-mode logit_normal --flow-shift S` | Logit-normal flow time plus the resolution shift `S * sqrt(tokens / native tokens)` per bucket, mirrored by `sample.py`. |
| **Noise offset** (Guttenberg 2023) | `--noise-offset 0.05` | Per-channel constant added to the training noise and regressed as part of the target. |
| **Spectral SFP loss (prototype)** | `--spectral-sfp-loss` | FFT-weighted loss on pred−target; timestep-dependent radial weights (`diffusion/flow_matching.py`). See **SFP** subsection above. |

---

### Research directions (not fully wired — why)

| Topic | What it is | Feasibility here |
|:------|:-----------|:-----------------|
| **Rectified flow / flow matching** | Train **velocity** along straight paths in data–noise space; often continuous-time. | Needs a **different training objective** and usually a new sampler. Possible as a **separate** experimental trainer, not a drop-in for current `training_losses`. |
| **Rectified-CFG++** | Predictor–corrector on **conditional vs unconditional velocity** to reduce off-manifold drift under CFG. | Targets **flow** checkpoints (e.g. SD3, Flux). Our loop is **DDIM-style VP**; porting needs careful equivalence or a flow checkpoint. |
| **RF-Sampling / test-time scaling** | Gradient steps on alignment scores during sampling (e.g. FLUX). | Inference-time only; depends on score / CLIP setup and is heavy. |
| **TPC (temporal pair consistency)** | Couple velocity at paired times to reduce gradient variance in flow training. | Flow-training specific. |
| **RecTok** | Distill semantics into tokenizer + flow. | Architecture + tokenizer change. |
| **Consistency / distillation** | 1–4 step student. | Separate training stage; listed in [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training). |

---

### Scaling, efficiency & architecture (2024–2025)

| Direction | What it is | SDX today / next step |
|:----------|:-----------|:----------------------|
| **FasterDiT** (NeurIPS 2024, arXiv **2410.10356**) | Analyzes **SNR distributions** across timesteps and proposes **training acceleration** + supervision tweaks **without changing DiT architecture** (reported ~7× faster to comparable FID on ImageNet-class setups). | You already use **Min-SNR weighting** and **non-uniform `t`** (`timestep_sampling`). Quick **distribution preview:** `python -m scripts.tools training_timestep_preview`. A full “FasterDiT-style” pass would **measure SNR PDFs on your data** and tune **loss weights + timestep schedule** jointly (experiment script, not default). |
| **Efficient scaling of DiTs** (arXiv **2412.12391**) | Large-scale study of **data + model scaling** for text-to-image DiTs (incl. U-ViT-style scaling claims in that line of work). | Use for **ablation priorities**: data quality > blind width; align **bucket / data engine** work ([IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) §1.1, §11.5). |
| **DiT-Air** (arXiv **2503.10618**) | Parameter-efficient DiT variants + conditioning studies; strong **GenEval / T2I-CompBench** with less bulk. | If you need **smaller deployable checkpoints**: try **shared blocks**, **narrower FFN**, or **depth vs width** trades before adding parameters. |
| **SANA 1.5** (arXiv **2501.18427**) | **Linear / efficient DiT** ideas, **depth-growth** training, pruning, and **inference-time compute scaling** (e.g. repeated sampling + judge). | **Inference:** stack with `--num K --pick-best …`. **Train:** depth-growth = staged unfreeze / progressive depth (heavy — new trainer mode). |
| **PixelDiT** (arXiv **2511.20645**) | **Pixel-space** dual-level DiT (patch semantics + pixel detail), reducing reliance on a single VAE latent. | Major pipeline change; treat as **long-term** unless you add a **separate pixel experiment** branch. |
| **Flow matching / RF** (e.g. arXiv **2403.03206** scaling rectified flow) | Continuous-time **velocity** paths; often pairs with **fewer steps** at inference. | Still **not drop-in** for current `GaussianDiffusion` training; see table above and [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) §11.1. |

---

### Human preference & alignment (post-train)

| Direction | What it is | SDX angle |
|:----------|:-----------|:----------|
| **Diffusion-DPO** (CVPR 2024; Wallace et al.) | **Direct preference optimization** on diffusion models from pairwise (win/lose) image data. | **Stage-2** after base T2I: small LR, frozen VAE/text encoder, preference pairs from **Pick-a-Pic-style** or **internal A/B** exports. |
| **RankDPO / scalable preferences** (e.g. ICCV 2025 line) | **Ranked** or **synthetic** preferences from reward models to scale beyond hand-labeled pairs. | Use **existing pick scores** (`utils/quality/test_time_pick.py`) as a **weak teacher** to **mine pairs** from generations (automated preference dataset). |
| **Curriculum-DPO++** (arXiv **2602.13055**) | **Curriculum** over pair difficulty + progressive capacity (unfreeze / LoRA rank). | Compose with your **caption / difficulty curriculum** knobs—same philosophy: **easy alignment first**, then hard cases. |

These do **not** replace a solid base model; they **steer** aesthetics and prompt-following once the base is stable.

---

### Product / engineering playbooks (non-paper)

- **Training ablations first** — e.g. Photoroom-style posts on **isolating one change at a time** before stacking objectives (avoids “mystery soup” configs).  
- **Noise / time sampling** — industry models increasingly treat **which noise levels you train on** as a first-class hyperparameter (you’ve started this with `timestep_sampling`).  
- **facebookresearch/flow_matching** — reference code for **continuous** flow training if you ever add a parallel trainer.

---

### Suggested reading

- arXiv **2410.10356** — FasterDiT (SNR PDF, faster DiT training).  
- arXiv **2412.12391** — Efficient scaling of diffusion transformers (T2I).  
- arXiv **2503.10618** — DiT-Air (efficient DiT + benchmarks).  
- arXiv **2501.18427** — SANA 1.5 (efficient DiT, depth growth, inference scaling).  
- arXiv **2511.20645** — PixelDiT (pixel-space DiT).  
- arXiv **2512.13421** — RecTok (flow + tokenizer).  
- arXiv **2603.06165** — Reflective flow sampling (inference).  
- Hugging Face **diffusers** SD3 discussions: logit-normal timestep sampling and loss weighting (issues #9056, #8591).  

When in doubt, prefer **documented flags** in `train.py` / `sample.py` and keep experimental objectives in a **new module** rather than overloading `GaussianDiffusion` without tests.

---

## Improvement ideas: quality, fixes, and better training

Ideas to make SDX better—quality gains, modern replacements for old techniques, and features borrowed from how other SD/DiT/FLUX models train and infer. Things we **already have** are noted so you know the baseline.

---

### Already in place (baseline)

- **CFG rescale** (`--cfg-rescale`) and **dynamic threshold** (`--dynamic-threshold`) in sampling — reduce oversaturation at high guidance (ComfyUI/SD3-style). **Configurable CFG** (`--cfg-scale`, default 7.5) so realistic/SDXL-style models can use 5–7. **Batch generation** (`--num N`), **VAE tiling** (`--vae-tiling` or auto when output > 512²) for large decodes, and **safetensors export** (`python -m scripts.tools export_safetensors`) for ComfyUI/A1111. **Civitai-style default negative prompt** (quality + anatomy/hands when negative is empty). **Resolution note** when output size is far from model native (blur warning). See [QUALITY.md](QUALITY.md) (*Civitai-style* section).
- **Min-SNR weighting** in training — stabilizes learning; avoids overfitting to easy timesteps (SD/SDXL).
- **Non-uniform training timestep sampling** — `--timestep-sample-mode logit_normal|high_noise` (SD3-style discrete logit-normal or high-noise bias); see [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) and `diffusion/schedules.py`.
- **Noise offset** — better light/dark balance in latents (SD/SDXL).
- **EMA** — inference from EMA weights; smoother, better generalization.
- **Passes-based training** — “N passes over data” instead of raw steps/epochs.
- **Validation + early stopping** — best checkpoint by val loss; stop when quality plateaus.
- **Negative prompt** — model learns to avoid unwanted content.
- **Refinement** (small-t training + optional refinement pass) — fix imperfections.
- **Post-process** — optional sharpen / contrast (`utils/quality/finishing.py`).
- **Aesthetic / sample weighting** — `weight` or `aesthetic_score` in JSONL.
- **Precomputed latents** — `--latent-cache-dir` for faster training.

---

### 1. Training: quality and stability

#### 1.1 Multi-resolution / aspect-ratio bucketing (SDXL / FLUX / SD3)

- **Idea:** Train on multiple resolutions or aspect ratios (e.g. 256², 512×256, 256×512) instead of a single square. Improves composition and “native” non-square generation.
- **How others do it:** SDXL uses resolution buckets; FLUX/Kontext use fixed preferred resolutions per aspect. We currently have a single `--image-size`.
- **Coded (partial):** `train.py --resolution-buckets` — comma-separated squares (`256,384,512`) or explicit `HxW` (`512x768,256x512`). Each image goes to the bucket nearest its aspect ratio (largest it fills without upscaling), so portraits are not cover-cropped into landscape frames; [`ResolutionBucketBatchSampler`](../data/t2i_dataset.py) keeps each batch one shape. **Constraints:** single GPU only (no DDP); `--val-split` must be 0; latent cache disabled for bucket runs. Prefer `--size-embed-dim` > 0 when training multiple sizes.
- **Future:** DDP-aware bucketing; optional multi-scale without buckets (padding) — not implemented.

#### 1.2 Smarter crop strategy — **coded**

- **Idea:** Center crop can cut off important content; random crop can miss the subject. “Largest center crop” that fits the short side, or “face/subject-aware” crop (if metadata exists), often gives better training signal.
- **Coded:** `--crop-mode center|random|largest_center` in `train.py`; `data/t2i_dataset.py` crops with `_cover_crop` (resize to cover, then a centre crop; `random` picks one offset shared by image, mask, control and init).

#### 1.3 Caption dropout schedule (curriculum) — **coded**

- **Idea:** Start with higher caption dropout so the model learns structure first; reduce dropout over time so it learns prompt adherence later. Reduces overfitting to captions early.
- **We have:** Fixed `caption_dropout_prob`. Curriculum for **caption length** exists (`curriculum_caption_steps`).
- **Coded:** `--caption-dropout-schedule 0,0.2,10000,0.05` (comma-sep pairs step,prob); linear interpolation. Dropout is always per sample in the training loop (caption -> empty prompt before encoding, matching the CFG uncond at inference); `--caption-dropout-prob` is the constant used when no schedule is set.

#### 1.4 V-prediction as default or option

- **Idea:** Some papers and SD2 use v-prediction (predict velocity); can behave better at high CFG and with certain schedulers. We support `--prediction-type v` but default is epsilon.
- **Add:** Document when to prefer v; optionally A/B test or make v the default for new configs.

#### 1.5 Checkpoint averaging (Polyak) — **coded**

- **Idea:** Average the last K checkpoints (or exponential moving average of weights over last N steps) and save as the “final” model. Often improves robustness and reduces variance.
- **Coded:** `--save-polyak N` keeps a running average of model weights (buffer = (1-1/N)*buffer + (1/N)*model each step); saves `polyak.pt` in the checkpoint dir every `ckpt_every`.

#### 1.6 Data quality pipeline (dedup, aesthetic, caption filter)

- **Idea:** Deduplicate by embedding or perceptual hash; filter or downweight low-aesthetic / low-caption-quality samples. Used by SD, SDXL, and many fine-tuners.
- **Add:** Script or dataset option: (a) dedup (e.g. by CLIP or image hash), (b) aesthetic score from a small rater (or use existing `weight` in JSONL), (c) caption length/quality filter (drop or weight by length, bad words, etc.). Document in data prep section.

#### 1.7 Training timestep distribution (logit-normal / high-noise) — **coded**

- **Idea:** Uniform random `t` is not the only choice; SD3-style pipelines often use a logit-normal on normalized time so the model sees some noise regimes more often. You can also bias toward **high noise** (large `t`) to stress hard denoising.
- **Coded:** `--timestep-sample-mode uniform|logit_normal|high_noise`, plus `--timestep-logit-mean` / `--timestep-logit-std` for logit-normal. VP-DDPM forward process is unchanged; only the batch distribution over discrete indices changes. Details: [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026).

---

### 2. Inference: quality and control

#### 2.1 More samplers / schedules

- **Idea:** DDIM is standard; DPM++ 2M, Euler, or flow-matching-style schedules can give better quality or fewer steps (e.g. 20–30 steps with a good schedule).
- **We have:** DDIM-style step in `gaussian_diffusion.py` with CFG rescale and dynamic threshold.
- **Add:** Optional second scheduler (e.g. DPM++ 2M or simple Euler) and a `--scheduler` flag so users can compare.

#### 2.2 Self-attention guidance (SAG) or cross-attention control

- **Idea:** Reduce attention to background (SAG) or scale cross-attention per token (prompt emphasis/de-emphasis at inference). Gives more control without retraining.
- **Coded (heuristic SAG):** `sample.py --sag-blur-sigma` + `--sag-scale` — in `GaussianDiffusion.sample_loop`, an extra forward on **Gaussian-blurred** latent blends `out += scale * (out - out_blur)` (~2× forwards). Not full in-model attention editing.
- **Coded (reference / IP-Adapter-style):** `--reference-image` encodes with CLIP vision, `ReferenceTokenProjector` → `reference_tokens` concatenated to cross-attn context in `DiT_Text`; CFG uncond omits reference. Optional `--reference-adapter-pt` for a trained projector.
- **Add:** True in-block SAG (modify self-attn maps) or cross-attn scale hooks beyond prompt `()` / `[]` emphasis.

#### 2.3 Default CFG rescale / dynamic threshold when CFG is high — **coded**

- **Idea:** When user sets high `--cfg-scale` (e.g. > 10), auto-enable a mild rescale or dynamic threshold so outputs don’t blow out. ComfyUI often does this.
- **Coded:** In `sample.py`, if `--cfg-scale` > 10 and user did not set `--cfg-rescale` or `--dynamic-threshold-percentile`, we auto-set `cfg_rescale=0.7` and `dynamic_threshold_percentile=99.5` and log.

#### 2.4 Fewer-step inference (distillation or consistency)

- **Idea:** Train a distilled model (e.g. consistency model or progressive distillation) for 4–8 step inference. Bigger change; often a second training stage.
- **Add:** Long-term: optional consistency/distillation training script that consumes our DiT and produces a few-step model; document as “advanced.”

#### 2.5 Style and artist tags (PixAI, Danbooru, Gelbooru) — **coded**

- **Idea:** Make the model really good at styles by using artist/style tags from tag boards; extract style from captions and prompts for the style conditioning head.
- **Coded:** (a) **config/styles.py**: `ARTIST_STYLE_PATTERNS` (e.g. "by X", "style of X", `artist:name`), `ARTIST_STYLE_TAGS` (known tags), `extract_style_from_text()`; (b) **data/caption_utils.py**: `DOMAIN_TAGS["style_artist"]` so style/artist tags are boosted; (c) **data/t2i_dataset.py**: when `style` is empty, auto-fill from caption via `extract_style_from_text` (`extract_style_from_caption=True`); (d) **sample.py**: `--auto-style-from-prompt` to extract style from prompt at inference; (e) **docs/PROMPT_STACK.md** and README.

#### 2.6 Prompt adherence for complex, NSFW, weird prompts — **coded**

- **Idea:** Improve quality and adherence for long/complex prompts, mature content, and surreal/abstract/weird prompts without censoring.
- **Coded:** (a) (b) **data/caption_utils.py**: domain tags "complex" and "challenging" (surreal, abstract, detailed, etc.) so `boost_domain_tags` reinforces them; `QUALITY_PREFIX` and `prepend_quality_if_short()` for short captions; (c) **sample.py**: `--boost-quality` (prepend "masterpiece, best quality"), long-prompt token count warning (>250 tokens); (d) **docs/QUALITY.md** (*Civitai-style* section): sections “Complex and long prompts” and “Challenging content (NSFW, surreal, abstract, weird)” with tips and no-censorship guidance.

---

### 3. Architecture and conditioning

#### 3.0 Block-wise AR (expand)

- **We have:** `num_ar_blocks` 0 / 2 / 4 with raster block order; mask in `models/attention.py` → `create_block_causal_mask_2d`; see [docs/MODEL_STACK.md](MODEL_STACK.md#block-wise-autoregressive-ar-generation).
- **Idea:** **Alternative block orders** — spiral (center-out or corner-in), row-by-row, or “content-aware” order (e.g. from a saliency map). Could improve composition for certain layouts.
- **Idea:** **AR only in early layers** — use block-causal mask in the first K transformer blocks and full bidirectional in the rest; early layers fix structure, later layers refine globally. Requires config (e.g. `ar_layers: Optional[int]`) and per-block mask in the forward.
- **Idea:** **Non-square AR** — support `h != w` for the patch grid (e.g. 32×16) so portrait/landscape latents get a sensible block order; currently mask is built for square `p×p`.
- **Add:** Document 3×3 blocks (`num_ar_blocks=3`) as valid; optionally add `--ar-block-order raster|spiral` and implement spiral in attention.py.

#### 3.1 Dual text encoder (T5 + CLIP)

- **Idea:** SDXL uses two text encoders (CLIP + T5). CLIP gives a strong global embedding; T5 gives long context. Combining both can improve both prompt adherence and “style” alignment.
- **Add:** Optional second encoder (CLIP) whose embedding is concatenated or projected into the conditioning path; config flag and extra projection layer. Requires more VRAM and data.

#### 3.2 Cached T5 / text encoder caching

- **Idea:** For large batches or many steps, encoding text once and reusing (caching) saves compute. Common in production and in FLUX-style pipelines.
- **We have:** Encode every batch in training.
- **Add:** In inference, cache encoder output per (prompt, negative_prompt) and reuse across steps; optional training path that caches per epoch (more complex).

#### 3.3 Inpainting / latent masking in training

- **Idea:** Train with randomly masked latent regions (model predicts only in the mask or the full image with mask conditioning) so the model can do inpainting/outpainting at inference.
- **Add:** Optional `--inpaint-prob` and mask generation (random rectangle or from dataset mask); pass mask to model (e.g. as extra channel or conditioning). Requires model change to accept mask.

#### 3.4 RoPE or other positional encodings for latent space

- **Idea:** DiT uses 2D sinusoidal or learned pos for patches. RoPE (rotary) is used in FLUX/LLMs and can improve length/resolution generalization.
- **Add:** Experimental: replace or augment patch pos with RoPE in the DiT block; compare at 256 vs 512.

#### 3.5 Latent size conditioning + patch channel gate — **coded (DiT-Text)**

- **Idea (PixArt-style):** Encode latent grid height/width into the timestep conditioning vector so the denoiser knows absolute scale (helps multi-res finetunes and resolution extrapolation).
- **Coded:** `TrainConfig.size_embed_dim` + `--size-embed-dim`; `SizeEmbedder` in `models/dit_text.py`. Training and `sample.py` pass `size_embed` `(B,2)` = latent `H,W` when `size_embed_dim > 0`. If `size_embed_dim > 0` but `size_embed` is omitted, the model infers `(H,W)` from the input latent tensor.
- **Idea (lightweight quality):** A channel-wise gate on patch tokens (SE-style) with **zero-init last layer** so training starts from identity and can learn calibration.
- **Coded:** `patch_se` / `--patch-se` and `patch_se_reduction` / `--patch-se-reduction` on `TrainConfig`; `ZeroInitPatchChannelGate` after patch/control fusion in `DiT_Text`.

---

### 4. Data and scaling

#### 4.1 WebDataset / streaming for huge datasets

- **Idea:** For 10M+ images, loading a single big JSONL or folder can be slow. WebDataset or tar-based streaming loads shards on the fly and scales better.
- **Add:** Optional `DataLoader` backed by WebDataset (or similar) when `--data-format webdataset` and shard path are set; keep current folder/JSONL as default.

#### 4.2 Booru tag normalization and quality tags

- **Idea:** For booru-style data: normalize tag order (e.g. subject, character, style, quality), merge synonyms, and enforce quality/rating tags so the model sees consistent conditioning.
- **We have:** PixAI-style emphasis and tag order in captions; quality boost.
- **Add:** Script or dataset option: tag vocabulary, synonym map, and “canonical” ordering for known booru tags; optional rating/quality prefix (e.g. `score_9, safe, ...`).

#### 4.3 Video / temporal (future)

- **Idea:** Train on video frames with temporal consistency (e.g. same noise across frames, or temporal attention). Makes the model suitable for video or animated sequences.
- **Add:** Long-term: video dataset loader (frame sampling + optional temporal loss or conditioning); document in HARDWARE.md with the rest of video notes.

---

### 5. Ops and reproducibility

#### 5.1 Optional WandB / TensorBoard — **coded**

- **Idea:** Log loss, LR, val loss, and optional samples to WandB or TensorBoard for easier monitoring and comparison.
- **Coded:** `--wandb-project NAME` and/or `--tensorboard-dir DIR`; loss and lr logged every `log_every`. Optional sample images can be added later.

#### 5.2 Export to ONNX / TensorRT / Safetensors

- **Idea:** Export DiT (and optionally VAE + text encoder) to ONNX or TensorRT for fast deployment; safetensors for smaller, portable checkpoints.
- **Add:** Script `scripts/export_onnx.py` or `export_safetensors.py` that loads a checkpoint and exports; document in README.

#### 5.3 Reproducible runs (full seed control)

- **Idea:** We have `--deterministic` and worker seeds. Extend to dataloader generator, CUDA seeds, and env flags (CUBLAS_WORKSPACE_CONFIG) so the same command gives bit-identical results.
- **Add:** Document and, if needed, set `torch.use_deterministic_algorithms(True)` (where supported) and seed everything in one place; note any non-deterministic ops.

---

### 6. Fixes to “old” techniques

| Old / naive approach | Problem | Better approach (we have or could add) |
|---------------------|--------|----------------------------------------|
| Train for fixed epochs | Same “epoch” means different amounts of learning for different dataset sizes; easy to overtrain. | **Passes** + **validation + early stopping** (we have). |
| Save only last checkpoint | Last step is often overtrained. | **Save best** by train or val loss; use **EMA** at inference (we have). |
| Constant LR to the end | Late training can overshoot or overfit. | **Cosine decay** to min_lr (we have). |
| Single resolution, center crop | Poor composition; crops can cut subjects. | **Multi-res / aspect bucketing**; **smarter crop** (ideas above). |
| High CFG without rescale | Oversaturation, blown colors. | **CFG rescale** and **dynamic threshold** (we have). |
| No negative prompt | Model can’t avoid unwanted content. | **Negative prompt** conditioning (we have). |
| Loss same weight for all t | Easy timesteps dominate. | **Min-SNR weighting** (we have). |
| No validation | Can’t detect overtraining. | **Val split + early stopping** (we have). |
| Raw steps / epochs only | User has to guess step count. | **Passes** (we have). |

---

### 8. Novel ideas (rare or never in open image gen)

Ideas that are uncommon or absent in mainstream DiT/SD/FLUX pipelines—worth experimenting with to differentiate.

#### 8.1 RLHF / preference learning for images

- **Idea:** Train a **reward model** on human (or model) preferences: “A is better than B” for prompt adherence, aesthetics, or safety. Then optimize the diffusion model (e.g. DPO-style, or REINFORCE with the reward) so generations score higher. Common in LLMs; almost never applied end-to-end in open image models.
- **Add:** Optional stage-2 training: freeze or low-lr base, add reward head or use reward to weight/ reject samples and fine-tune toward preferred distribution. Requires preference data or a proxy (e.g. CLIP score, aesthetic scorer).

#### 8.2 Constitutional / rule-based auxiliary loss

- **Idea:** Define **rules** (e.g. “no text in image”, “correct hand count”, “no duplicate face”) and add an auxiliary loss or reward that penalizes violations. Can use a small classifier or VLM to score generated samples and backprop or use as a rejection signal. Rarely baked into open image gen training.
- **Add:** Optional `--rule-loss` with a list of rule modules (e.g. “no text”: OCR on decode, penalize if text detected; “anatomy”: hand/face detector, penalize bad count). Blend with denoising loss or use in a second phase.

#### 8.3 Mixture of experts (MoE) in the DiT

- **Idea:** Replace some or all transformer blocks with **MoE**: multiple “expert” sub-networks and a router (e.g. from prompt or latent) that selects which experts run. Standard in LLMs; very rare in image DiT. Could specialize experts by “style”, “object”, “background”, etc.
- **Add:** Experimental MoE block: 2–4 experts, router conditioned on text embedding or pooled latent; same FLOPs budget as one block but more capacity. Requires careful load balancing (aux loss or capacity constraint).

#### 8.4 Retrieval-augmented generation (RAG) for style or layout

- **Idea:** At inference (or training), **retrieve** a small set of relevant vectors (e.g. from a database of style codes, or “similar prompt” embeddings) and condition the model on them via cross-attention or concatenation. “What would this look like in the style of these N reference embeddings?” Rare in open text-to-image.
- **Add:** Optional “style memory”: user or system provides a few reference image embeddings; model gets a small KV set to attend to in addition to T5. Or retrieve similar captions from training and inject their conditioning.

#### 8.5 Spatial “avoid” conditioning (negative regions)

- **Idea:** User specifies **regions** where certain things must not appear (e.g. “no text here”, “no face here”). Negative prompt is global; **spatial negative** conditioning is rare. Could be a latent mask or per-patch “avoid” embedding that the model is trained to respect.
- **Add:** Optional `avoid_mask` or `avoid_regions` in training (and inference): binary or soft mask + optional class (e.g. “text”). Model gets an extra conditioning channel or loss term that discourages those classes in those regions. Requires data with region annotations or user-drawn masks.

#### 8.6 Self-improvement loop (synthetic caption bootstrap)

- **Idea:** Generate images with the current model, caption them with a **vision–language model** (e.g. LLaVA, BLIP-2), then train on (generated_image, synthetic_caption) to improve coherence and “understand” its own outputs. Used in some closed systems; rarely in open DiT.
- **Add:** Optional pipeline: sample from current checkpoint, caption with VLM, add to a synthetic dataset; mix with real data (e.g. 10% synthetic) and continue training. Risk: drift if VLM is biased; use with validation and cap on synthetic ratio.

#### 8.7 “Creativity” or diversity knob (temperature by region or step)

- **Idea:** A single scalar (or per-step schedule) that controls how much the model **deviates** from the mode—e.g. higher noise in the sampler, or a learned “diversity” embedding that shifts the distribution. Lets users ask for “safe, on-prompt” vs “weirder, more varied” without changing the prompt. Partially overlaps CFG; explicit “creativity” parameter is rare.
- **Coded:** `creativity_embed_dim` + per-batch random `creativity` in training; **`--creativity-jitter-std`** widens the training scalar; **`--creativity-jitter`** at sample (esp. `--num` > 1). **`--originality`** / **`--train-originality-prob`** inject `ORIGINALITY_POSITIVE_TOKENS` from [`utils/prompt/art_direction.py`](../utils/prompt/art_direction.py).
- **Add (future):** Per-step creativity schedule; regional diversity without prompt edits.

#### 8.8 Learning from rejection (negative signal from user feedback)

- **Idea:** When a user **rejects** an image (thumbs down, or “regenerate”), use it as **negative** signal: contrastive loss (push embedding away from this) or fine-tune to reduce likelihood of this output given the prompt. Requires logging and a feedback loop; almost never in open image gen.
- **Add:** Optional “rejection set”: store (prompt, negative_prompt, latent or image) for rejected samples; in a separate training phase, add a small loss term that discourages the model from producing similar outputs for that prompt. Or use rejection samples to train a small discriminator and use it as in 8.1.

#### 8.9 Dual output: image + explanation (attention or keyword heatmap)

- **Idea:** Model outputs not only the image but a lightweight **explanation**: e.g. which prompt tokens most influenced which spatial regions (attention rollout or gradient-based), or a short “why” caption. Interpretability for text-to-image is rare in open models.
- **Add:** At inference, optionally compute and return cross-attention weights (token → patch) or integrated gradients; visualize as heatmap or “key words per region”. No training change; just hooks and a small export format.

#### 8.10 Unlearning / concept removal

- **Idea:** Fine-tune the model to **reduce** the probability of generating a specific concept (e.g. a person’s face, a logo, watermarks) without hurting general quality. Machine unlearning for diffusion is emerging; not standard in open pipelines.
- **Add:** Optional script: given a “concept” (e.g. set of images or a text description), run a constrained fine-tune that minimizes likelihood of that concept while regularizing toward the base model (e.g. gradient ascent on concept, plus KL or L2 toward base). Requires careful tuning to avoid catastrophic forgetting.

#### 8.11 Adversarial refinement head (discriminator on top of diffusion)

- **Idea:** Keep diffusion as the main generator but add a **discriminator** (e.g. PatchGAN or small CNN) on decoded images and train with a small GAN loss (real vs generated) to sharpen details and fix blur. Used in some super-resolution and old GAN–diffusion hybrids; rare in modern open DiT.
- **Add:** Optional `--adversarial-refine`: freeze or low-lr DiT, add discriminator, train with combined denoising + adversarial loss (e.g. 0.9 * L_denoise + 0.1 * L_adv). Risk: mode collapse; use small weight and short phase.

#### 8.12 Difficulty-weighted curriculum — **coded**

- **Idea:** Score each training sample by **difficulty** (e.g. caption length, rarity of tokens, number of objects, or failure rate of a baseline model). Train first on easy samples, then gradually mix in harder ones. Some curriculum exists elsewhere; explicit difficulty scoring and scheduling is uncommon in image gen.
- **Coded:** JSONL may include `"difficulty": 0.0–1.0`. Config `curriculum_difficulty_steps` (e.g. [0, 5000, 10000]), `curriculum_difficulty_easy_first`; train.py multiplies sample_weights by difficulty schedule (early=easy, late=hard). `--curriculum-difficulty-steps 0,5000,10000`.

---

### 7. Suggested priorities

- **Quick wins:** Default CFG rescale when CFG is high (2.3); caption dropout schedule (1.3); checkpoint averaging / Polyak (1.5).
- **Medium effort:** Multi-resolution or aspect bucketing (1.1); data quality script (1.6); WandB/TensorBoard (5.1); export safetensors/ONNX (5.2).
- **Larger projects:** Dual encoder (3.1); inpainting training (3.3); WebDataset (4.1); second scheduler (2.1); SAG / cross-attn control (2.2).
- **Novel / research:** RLHF or rule-based loss (8.1, 8.2); RAG for style (8.4); creativity knob (8.7); self-improvement loop (8.6).
- **Push quality toward “insane”:** Read **§11** — flow/EDM objectives, few-step distillation, best-of-N + quality gates, layout/box conditioning, multi-anchor REPA, RAE adapter + decode refine, VLM data flywheel.

---

### 9. Ideas you might add next

- **Sample grid:** **Done** — `sample.py --num 4 --grid` saves individual files and a single `stem_grid.png` 2×2 (or N-up) grid.
- **Export safetensors:** **Done** — `python -m scripts.tools export_safetensors` saves DiT `state_dict` to `.safetensors` for ComfyUI / other tools.
- **WandB or TensorBoard:** **Done** — `--wandb-project`, `--tensorboard-dir` in train.py (see 5.1).
- **Default negative prompt:** Done — when `--negative-prompt` is empty, sample.py uses `config.prompts.DEFAULT_NEGATIVE_PROMPT`.
- **Checkpoint inspector:** Done — `python -m scripts.tools ckpt_info path/to/best.pt` prints config and steps; `--keys` lists checkpoint keys.
- **Smoke test:** Done — `python -m scripts.tools quick_test` runs one forward pass to verify imports and model.
- **Dry-run training:** **Done** — `--dry-run` runs one step and exits (sets max_steps=1).
- **.env.example:** **Done** — `.env.example` documents `HF_TOKEN` and `CUDA_VISIBLE_DEVICES`; copy to `.env` (in .gitignore).
- **Reproducible sampling:** **Done** — `sample.py --deterministic` sets cudnn deterministic + benchmark off and (when supported) deterministic algorithms so same seed → same image.

---

### 10. What the model is still missing (suggested next)

Things commonly expected in production or in ComfyUI/A1111 that we don’t have yet. Pick by priority.

| Gap | What it is | Where (IMPROVEMENTS) | Effort |
|-----|------------|----------------------|--------|
| **Alternative samplers** | **Done** — `--scheduler ddim|euler` in sample.py; Euler uses linear timestep spacing. | §2.1 | — |
| **Prompt emphasis (infer + train)** | **Done** — `(word)` / `[word]` → per-token scale 1.2 / 0.8 (`token_weights`). Training: **`train.py --train-prompt-emphasis`** + [`utils/prompt/text.py`](../utils/prompt/text.py). | §2.2 | — |
| **Multi-res / aspect bucketing** | **Partial** — `train.py --resolution-buckets` + `ResolutionBucketBatchSampler`; single-GPU, no val-split with buckets, latent cache off. DDP + buckets not done. | §1.1 | Medium (finish DDP / val) |
| **Data quality pipeline** | **Done** — `scripts/tools/manifest.py`: dedup (phash/md5), min/max caption length, bad-words, min-weight. MD5 uses Rust **`sdx-jsonl-tools file-md5`** when built (streaming; matches `hashlib`); **`--no-native-md5`** for Python-only. | §1.6 | — |
| **T5 caching (server)** | **Done** — `sample.py` caches T5 output per (prompt, negative, style); `--no-cache` to disable. | §3.2 | — |
| **Log sample images (train)** | **Done** — `--log-images-every N` and `--log-images-prompt`; logs to WandB/TensorBoard. | §5.1 | — |
| **Export ONNX / TensorRT** | **Done** — `python -m scripts.tools export_onnx` exports DiT to ONNX; `--dynamic-batch` supported. | §5.2 | — |
| **Full reproducibility** | **Done** — `--deterministic` in sample.py and train.py; [REPRODUCIBILITY.md](GETTING_STARTED.md#reproducibility-improvements-53). | §5.3 | — |
| **Dual text encoder (T5+CLIP)** | SDXL-style; second encoder for style/alignment | §3.1 | Large |
| **Inpainting in training** | Train with random masks so model learns inpainting natively (MDM-style patch masking). Inference already supports MDM-style known-region freezing via `sample.py --mask --inpaint-mode mdm`. | §3.3 | Large |
| **WebDataset** | Tar-based streaming for 10M+ images | §4.1 | Medium |

Use this doc as a roadmap: pick items that match your goals (quality vs speed vs scale) and implement in small steps, then validate with the same dataset and metrics (e.g. val loss, visual samples).

---

### 11. Next-tier: “insane” image quality (research → practical SDX hooks)

High-impact ideas from recent text-to-image / DiT / flow work. Many compose with what SDX already has (**REPA, MoE, MDM, AdaGen/PBFM, register tokens, RoPE, KV-merge, token routing, OCR/book pipelines**).

#### 11.1 Training objective upgrades (often > raw architecture for “wow”)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **Rectified flow / flow matching** | Straighter noise→data paths; often **fewer steps** and **calmer** optimization than vanilla epsilon prediction on some setups. | Add optional **velocity / flow** training target (you already have `prediction-type` and loss weighting knobs—extend toward RF-style schedules and sampling). |
| **EDM / preconditioning end-to-end** | Cleaner scaling of \( \sigma \), better high-frequency detail when tuned. | You have `loss_weighting` hints; document an **EDM-first** preset and match **sampler** to the same preconditioning. |
| **Multi-scale x0 consistency** | Forces the net to be self-consistent across noise levels; reduces “mushy” midsteps. | Auxiliary loss on decoded previews: **LPIPS / SSIM** vs EMA teacher \(\hat x_0\) (expensive—cap batch or low-res decode). |
| **Beyond REPA: multi-anchor alignment** | One frozen encoder misses texture or typography; multi-encoders catch different failure modes. | **REPA stack**: e.g. **DINOv2 (structure)** + **SigLIP/CLIP (semantic)** with **small fused projector** and schedule (warm-start REPA weight). |

#### 11.2 Few-step generation (where “insane” meets “instant”)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **Consistency / distillation (LCM, SDXL-Turbo, Hyper-SD spirit)** | **4–8 step** quality usable in prod. | Stage-2 script: student DiT with **consistency loss** against a frozen teacher + your latents; export a separate `*-fast.pt`. |
| **Guidance distillation** | Reduces **CFG blowout** and can remove **two-forward** cost. | Train a student that absorbs **_CFG + negative_** into weights (pairwise synthetic data or distillation loss). |
| **Step-wise adversarial / discriminator refine** | Sharper micro-detail (texture, eyes, lettering edges) without wrecking diversity if regularized. | Light **PatchGAN** on **short decode crops** + tiny weight; or **stage-only** after main training (see §8.11). |

#### 11.3 Inference-time scaling (cheap code, big perceived quality)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **Best-of-N with a cheap judge** | Pick the winner from **K** samples; huge win for **hands / faces / text**. | `sample.py --num K` + **pick**: **CLIPScore**, **aesthetic head**, **OCR match** (you already have OCR repair—use score before repair). |
| **Test-time x0 refinement loop** | One or two **extra denoise** passes at low noise fixes artifacts. | You have **refinement**; add a **quality gate**: only refine if **blur/edge** heuristic or **small CLIP margin** fails. |
| **Dynamic schedule** | Allocate steps where the latent **changes most** (related to AdaGen). | Generalize AdaGen to **per-step budget**: more steps when \(\Delta z\) large; fewer when flat (full “adaptive schedule” vs early-exit only). |

#### 11.4 Conditioning & layout (prompt adherence without bigger DiT)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **Box / sketch / layout tokens (GLIGEN-style)** | **Composition** beats raw scale for multi-object scenes. | Extra cross-attn tokens from **bounding boxes** + **class** embeddings in JSONL (`layout` field). |
| **Segmentation or depth as first-class control** | Reduces **wrong object boundaries** and **merged bodies**. | Beyond edges: **sky/ground/character** maps as **control_cond** channels (you have control conditioning—extend channel semantics). |
| **Token-level routing from saliency** | Spend compute on **subject** not **sky**. | **Token routing** exists; feed a **cheap saliency prior** (blur map, face box, text mask) to **bias** `token_router` inputs at inference. |

#### 11.5 Data Engine (often beats a 10% wider model)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **VLM re-caption + filter** | Fixes **wrong associations** and improves **rare concepts**. | Batch: **describe → filter by confidence → dedupe**; mix at **5–15%** with real captions. |
| **Synthetic hard negatives** | Teaches **what not to merge** (two faces, six fingers). | Generate **failure cases** on purpose; label with **negative_caption**; small weight in JSONL. |
| **Difficulty-aware mixing** | You have **curriculum difficulty**—pair with **MoE load balancing** so hard samples route to **“detail experts.”** | Joint schedule: **curriculum ↑** + **moe balance loss** tuned together. |

#### 11.6 Autoencoder / latent space (ceiling on “insane”)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **RAE + learned adapter to DiT** | RAE latents are **high-dim**; a **linear or tiny conv adapter** aligns channels to your **4×h×w** DiT without rewriting the world. | Document + implement **explicit adapter** when `autoencoder_type=rae` (currently guarded). |
| **Decoder refinement net** | Fixes **VAE mush** on text and fine lines. | Tiny U-Net on **RGB residual** after VAE (train frozen DiT, only refine decode—fast iteration). |

#### 11.7 Suggested “insane mode” stacks (preset combos)

1. **Quality ceiling (slow, best pixels):** REPA (multi-anchor) + MDM + high-step sampler + best-of-4 + optional PBFM + refinement gate.  
2. **Speed ceiling (fast, still strong):** Distilled student + RF-style sampler + AdaGen + KV-merge + token routing + **no CFG** (if guidance-distilled).  
3. **Book / comic ceiling:** Face + bubble anchoring + OCR loop + layout tokens (when added) + edge control.

When you implement anything from §11, add a **one-line “wired in” note** here (like §1.2) so the roadmap stays honest.

#### 11.8 Wired in (this repo)

| Idea | Where |
|------|--------|
| **Non-uniform training timesteps** | `diffusion/schedules.py`; `train.py` `--timestep-sample-mode uniform\|logit_normal\|high_noise`, `--timestep-logit-mean`, `--timestep-logit-std`. |
| **RAE ↔ DiT latent bridge** | `models/conditioning.py`; `train.py` creates the bridge when RAE `encoder_hidden_size != 4`, adds optional `--rae-bridge-cycle-weight`, saves `rae_latent_bridge` in checkpoints; `sample.py` / `inference.py` load it and run `dit_to_rae` before decode. |
| **Test-time best-of-N** | `sample.py --num N --pick-best clip\|edge\|ocr\|combo` (+ `--pick-save-all`, `--pick-clip-model`); scores in `utils/quality/test_time_pick.py`. |

#### 11.9 Paper-backed horizon (2024–2026) — make the model “OP”

High-signal directions that are **not all implemented** but are **actionable** as ablations, stage-2 training, or inference stacks. Full citations and SDX mapping: **[MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026)**.

| Cluster | Move | Why it can dominate |
|--------|------|---------------------|
| **Train faster / same budget** | **FasterDiT-style** SNR PDF analysis + tuned supervision and timestep policy (NeurIPS 2024 / arXiv 2410.10356). | Same architecture, better **step efficiency**; pairs with your **Min-SNR** + **`timestep_sample_mode`**. |
| **Scale smarter** | Follow **scaling-law** style studies (e.g. arXiv 2412.12391): **data + curriculum** before raw parameter inflation. | Avoids training a bigger model on **dirty or narrow** data. |
| **Smaller strong checkpoint** | **DiT-Air**-style efficiency (arXiv 2503.10618): shared layers, width/depth trade, conditioning ablations. | Better **GenEval / CompBench per parameter** for deployment. |
| **Inference compute** | **SANA-style** “spend more at test time”: multi-sample + **judge** (you have pick-best); optional **depth/width** student for edge devices. | Perceived quality ↑ without retraining the whole stack. |
| **Alignment pass** | **Diffusion-DPO** / **RankDPO** / curriculum preferences (CVPR 2024 line; ICCV 2025; arXiv 2602.13055). | Fixes **aesthetics, prompt following, safety** after base convergence. |
| **Long-term architecture** | **PixelDiT** (arXiv 2511.20645), full **rectified flow** trainer, **RecTok**-class tokenizers. | Ceiling raise; **large** engineering cost—treat as parallel experiments. |

**Practical “OP stack” order (recommended):** (1) **Data engine + buckets** → (2) **timestep + loss tuning** (SNR-aware) → (3) **best-of-N + refinement gate** at sample → (4) **preference fine-tune** on curated pairs → (5) **distillation** for speed.

---

### 12. Industry alignment (2026): authenticity, systems, resolution, text, grounding

**Context:** Image generation is increasingly **production infrastructure**—precision, grounding, and “real” outputs matter as much as aesthetics. See **[LANDSCAPE_2026.md](#image-generation-landscape-architecture-and-workflows-2026)** for the full narrative and **named pipeline roles** in **[utils/agentic/agent.py](../utils/agentic/agent.py)**.

| Trend | What leading stacks emphasize | SDX hooks today | Good contributions |
|-------|------------------------------|-----------------|-------------------|
| **Authenticity (“anti-plastic”)** | Natural texture, asymmetry, lens-like noise | CFG rescale / dynamic threshold, refinement, domain tags | Bucketed multi-res training (§1.1), optional frequency losses, richer “lived-in” captions |
| **System of experts** | Designer / Verifier / Reasoner pipelines | `sample.py` + `--pick-best` + `ViT/` + T5/triple/LLM | Orchestration script wrapping K samples → score → refine; document in LANDSCAPE_2026 |
| **Native high-res & AR** | Fewer pure upscaler workflows | `--width`/`--height`, VAE tiling | §1.1 buckets, docs for train-low / sample-high |
| **Text & design in-image** | Paragraphs, logos, UI | OCR pick-best, MODEL_WEAKNESSES | Text-heavy JSONL, preference/DPO later (§11) |
| **Live / retrieval grounding** | External facts in the loop | None built-in (frozen encoders) | Optional RAG module + clear safety/docs split |

**Tickets (actionable, small PRs welcome):**

1. Add **one** end-to-end doc example: `sample.py --num 4 --pick-best combo` + when to enable refinement (link from LANDSCAPE_2026).
2. Optional **thin orchestration CLI** under `scripts/tools/` that chains generate → `test_time_pick` → optional refine—keep minimal and tested if added.
3. Extend **spatial / relation** coverage tools (see [repo README](../README.md#extra-features) “Extra features”) for “blue book under red book” style prompts—aligns with “reasoner” stress tests.

Cross-reference: §11 (insane quality), [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026), [QUALITY.md](QUALITY.md).

---

## Research blueprints: fast generation & prompt accuracy

Design notes that map **literature-style ideas** to **SDX today** (not benchmark numbers). For ecosystem context see [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026), [LANDSCAPE_2026.md](#image-generation-landscape-architecture-and-workflows-2026), and [utils/modeling/architecture.py](../utils/modeling/architecture.py).

---

### Part 1: Flow, solvers, distillation, and rectified paths

**Purpose:** Organize **mathematics- and speed-first** directions (few-step / one-step quality) and map them to **SDX today**. This is a **design and literature** guide, not a benchmark report: **step counts, FID, and millisecond figures** in popular write-ups are **checkpoint- and hardware-dependent**; treat them as **directional**, not guarantees for this repo.

**Related:** [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) (VP vs flow, SFP, timestep sampling), [LANDSCAPE_2026.md — architecture section](#post-diffusion-and-structural-shifts-late-20252026), [DIFFUSION_LEVERAGE_ROADMAP.md](#diffusion-stack-high-leverage-upgrades-10-class-levers).

---

#### 1. Consistency-style objectives on flow fields (Consistency-FM narrative)

**Idea (high level):** Standard **flow matching** learns a time-dependent velocity field \(v_\theta(x_t, t)\) that transports noise toward data. **Consistency** variants add constraints so that **endpoints agree** when integrating from **different** starting times along the *same* underlying field—reducing path dependency and enabling **larger integration steps** without drifting off the intended trajectory.

**Sketch:** One imagines enforcing that flows from \(t_1\) and \(t_2\) to a terminal time land on the **same** predicted clean state (exact forms differ by paper: distillation, consistency models, trajectory matching, etc.).

**Why people care:** Fewer solver steps if the field is **self-consistent** and the integrator can take **large jumps** without error accumulation from a highly curved effective trajectory.

**SDX today:** Default training is **VP DDPM** + `GaussianDiffusion.training_losses` (ε, v, or x₀ targets). **Optional** rectified-flow-style velocity training exists (`--flow-matching-training`, `diffusion/flow_matching.py`) with matching sampling (`sample_loop(..., flow_matching_sample=True)` / `sample.py` flags)—this is **not** a multi-time **consistency-FM** objective. Default inference is **DDIM-style** strided stepping on a **discrete** VP schedule. **Status:** dedicated **consistency** constraints on the flow field are **not implemented**; see [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) §2.4 (distillation / consistency roadmap).

---

#### 2. “Dual-solver” / mixed prediction domains (ε, v, x₀) and time warping

**Idea:** Diffusion “work” is often **non-uniform** in a linear time index: early steps establish coarse structure; late steps refine texture. A **dual** or **multi-regime** solver might:

- Use different **parameterizations** (noise ε vs velocity-like v vs data x₀) in different bands of \(t\), or  
- Warp the integration grid (e.g. **log-like** emphasis on low vs high noise) to reduce **NFE** for a fixed perceptual quality.

Some industry stacks also speak of **learned** or **adaptive** schedules (meta-parameters akin to \(\tau\))—that is **not** the same as a fixed strided DDIM schedule.

**SDX today:**

| Knob | Role |
| :--- | :--- |
| `--prediction-type` `epsilon` \| `v` \| `x0` | **Single** parameterization per checkpoint (train and sample must match). |
| `--timestep-sample-mode` (e.g. `logit_normal`, `high_noise`) | **Training-time** emphasis on which noise levels are sampled—related to “where the work is,” but **not** a learned dual integrator. |
| Strided `sample_loop` | Fixed schedule over discrete VP steps. |

**Status:** **Partial** analogy only (parameter choice + non-uniform training times). **No** per-step dynamic switching between ε/v/x₀ at inference and **no** learned log/linear domain mixer \(\tau\).

---

#### 3. Adversarial Diffusion Distillation (ADD-class)

**Idea:** Distill a large **teacher** generator into a smaller **student** with **adversarial** critics so the student matches both **pixel-level** sharpness and **semantic / feature-level** agreement with the teacher—often targeting **1–2 step** or very few-step generation.

**Why people care:** Pure regression-to-teacher can blur high frequencies; a **discriminator** (possibly multi-head: RGB vs frozen encoder features) pressures the student to stay sharp and on-manifold.

**SDX today:** No ADD trainer, no dual-head discriminator distillation path in-tree. **DMD / consistency** themes are likewise **not in repo** (see `dmd_distillation` in [`utils/modeling/architecture.py`](../utils/modeling/architecture.py)). **Status:** **not implemented**.

---

#### 4. “Straight” paths: rectified flow + optimal transport pairing

**Idea:** If the learned transport is **approximately linear** in a suitable coordinate system, an ODE solver can approach **one-step** integration:  
\(x_{t+\Delta t} \approx x_t + v_\theta(x_t, t)\,\Delta t\).

**OT coupling (Hungarian / Sinkhorn):** During training, **pair** noise samples to data samples so that **transport cost** (e.g. squared distance) is minimized batch-wise. Intuition: **shorter average paths** than i.i.d. pairing can yield **easier** fields to integrate and better few-step behavior—when combined with the right architecture and objective.

**SDX today:** Default VP training still uses **independent** Gaussian noise per sample. Optional **mini-batch Sinkhorn / Hungarian** coupling is available via **`train.py --ot-noise-pair-reg`** (see `utils/training/objectives.py`) — an **experimental** analogue, **not** a full rectified-flow ODE trainer.

---

#### Qualitative comparison (illustrative only)

| Approach | Paradigm (rough) | Typical step regime (literature / products) | SDX |
| :--- | :--- | :--- | :--- |
| Classic VP DiT | SDE / discrete VP + DDIM family | 25–50+ | **Default path** |
| Flow matching / RF | Continuous-time velocity ODE | 10–20 (varies) | **Not** drop-in trained |
| Consistency / trajectory matching | Self-consistent field or distilled student | 2–8 | **Not** in-repo trainer |
| ADD / GAN–diffusion hybrid | Adversarial distillation | 1–2 | **Not** in-repo |
| OT-coupled RF | Straight paths + optimal pairing | 1–4 (aspirational) | **Partial** (train OT noise coupling only) |

---

#### Practical build order (if you implement)

1. **Baseline** — stable VP training + v or ε + min-SNR + `timestep_sample_mode` ablations ([MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026)).  
2. **Separate branch** — experimental **flow** trainer + matched sampler (do not silently mix with VP checkpoints).  
3. **Distillation** — teacher → student (consistency or ADD-class) as a **second stage** with its own loss and eval.  
4. **OT / pairing** — only after (2) is stable; measure wall-clock and FID/GenEval vs cost.

---

#### Machine-readable map

Theme IDs in [`utils/modeling/architecture.py`](../utils/modeling/architecture.py):  
`consistency_flow_matching_velocity`, `dual_solver_time_warping`, `add_adversarial_distillation`, `rectified_flow_ot_coupling`.

---

### Part 2: Prompt-accurate generation (GLS, critics, frequency)

**Purpose:** Capture **technical** directions for beating typical DiT-style “layout and texture in one forward” behavior—without treating any row in the comparison table as a **measured** SDX result. DiTs **jointly** refine semantics and appearance; **decoupling**, **in-loop verification**, and **frequency-aware** objectives are levers to reduce **semantic bleeding**, **wrong placement**, and **global inconsistency**.

**Related:** [NEXTGEN_SUPERMODEL_ARCHITECTURE.md](#next-gen-image-generation-four-design-pillars-and-sdx-mapping) (overlapping pillars), [Part 1: Flow, solvers, distillation](#part-1-flow-solvers-distillation-and-rectified-paths), [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) (spectral SFP).

---

#### 1. Structural blueprint — geometric–latent split (GLS)

**Problem:** When **composition** (what is where, which color belongs to which object) and **appearance** (texture, lighting) are entangled in the **same** deep blocks, errors propagate: e.g. “red cube on blue table” can **mix** hues because there is no hard boundary between “geometry solved” and “materials applied.”

**Idea — Geometric–Latent Split (GLS):** A **small, fast, spatially aware** module (often a shallow transformer or conv stack) produces a **low-resolution geometric prior** early in the process: e.g. **depth**, **surface normals**, **edges**, or a learned **layout tensor**. Later denoising is **conditioned** on this prior as a **constraint** (frozen or slowly updated) so that **object boundaries and relations** stabilize before **high-frequency** detail is filled in.

**Engineering variants:** Auxiliary heads on the DiT; a **two-phase** latent pass; explicit **edge/depth** supervision from synthetic or labeled data; or a **separate** network whose output is **concatenated** or **cross-attended** by the main denoiser.

**SDX today (partial):**

| Mechanism | Role |
| :--- | :--- |
| **`--dual-stage-layout`** | Coarse latent pass → upscale → detail pass (**workflow** layout-first, same DiT). |
| **`highfreq_layout_prior` / `--domain-prior-latent`** | Injects a **high-frequency / layout** prior in research hooks — not a full depth–normal stack. |
| **Control / scene blueprint text** | User- or data-side **structure**; not a learned immutable geometric tensor. |

**Gap:** No dedicated **Spatially-Aware Transformer** that outputs **depth + normals + edges** in the **first ~10% of steps** and **locks** the rest of sampling to that tensor as a **hard** constraint.

---

#### 2. Recursive self-correction — discriminative denoising loops

**Problem:** Sampling is **causal in time**: a mistake at mid-noise is **expensive** to undo; the model often **refines** the wrong structure into a sharp but **wrong** image.

**Idea:** Periodically run a **frozen** strong **vision–language** model (or a specialized verifier) on a **decode of the current state** (or on auxiliary features). Compare **visual content** to the **prompt**. If there is a **detectable mismatch** (missing object, wrong attribute), apply a **targeted correction**: e.g. **re-noise** a **spatial mask**, **raise CFG** or **prompt strength** locally, or **rewind** a few steps and branch—**“local gradient re-roll”** in the narrative means **steered** updates in **latent space**, which in practice may be **heuristic** (mask + extra steps) unless the VLM is fully differentiable through the decoder.

**SDX today (partial):**

| Mechanism | Role |
| :--- | :--- |
| **`--clip-guard-threshold`** + [`utils/generation/guidance.py`](../utils/generation/guidance.py) | **CLIP** image–text similarity; optional **short** extra `sample_loop` — coarse **global** alignment, not structured VLM QA. |
| **`--clip-monitor-every`** + same CLIP model | **Mid-loop**: decode `x_0` pred every *N* steps; if cosine **<** `--clip-monitor-threshold`, multiply CFG by `(1 + --clip-monitor-cfg-boost)`. **Expensive**; not a full VLM or localized rewind. |
| **`--volatile-cfg-*`** | **Heuristic** instability signal on latent deltas — not semantic verification. |
| **`--pick-best`**, orchestration | **Post-hoc** multi-sample selection, not in-loop per-step VLM. |
| **`ViT/`** scorer | **Offline** or **post-generation** ranking — not inside the denoising trajectory. |

**Gap:** No **every-k-steps** frozen **VLM** with **spatially localized** rewind + re-denoise tied to **named** discrepancies; **“nearly 100% prompt adherence”** is an **aspirational** claim, not something to expect without heavy infrastructure and eval.

---

#### 3. Frequency-domain diffusion — neural operators (FNO narrative)

**Problem:** Fixed **grid** latents bias models toward **local** mixing; **global** lighting and long-range consistency across a very large canvas are harder when each step only “sees” neighbors through a finite receptive field (even with attention, at extreme resolutions cost bites).

**Idea:** Treat the field as a **function** and operate partly in the **frequency domain** (e.g. **Fourier Neural Operators** or spectral blocks): **low** frequencies encode **global** layout and illumination; **high** frequencies encode **detail**. Denoising or velocity prediction in **spectral** space can emphasize **whole-image** coherence before filling **fine** structure.

**Caveats:** “**Infinite resolution**” and “**perfect** prompt accuracy at any size” are **not** guaranteed by FNOs alone—you still need **training scale**, **conditioning**, and **sampling** matched to the objective.

**SDX today (partial):**

| Mechanism | Role |
| :--- | :--- |
| **`--spectral-sfp-loss`** + [`diffusion/flow_matching.py`](../diffusion/flow_matching.py) | **Training** loss: FFT-weighted error on pred−target in **latent** space — **not** full FNO **forward** denoising. |
| **`--spectral-coherence-latent`** + [`spectral_latent_lowfreq_blend`](../utils/generation/guidance.py) | **Inference** FFT low-frequency blend on the final latent (global coherence heuristic). |
| **Multi-scale inference** (hires, dual-stage) | **Process** scaling, not continuous neural-operator sampling. |

**Gap:** No **sampler** that denoises **primarily** in a learned **spectral / operator** basis with **resolution-agnostic** forward pass as in research stacks.

---

#### Comparison (qualitative — not benchmarked in this repo)

| Axis | Typical joint DiT | GLS blueprint | In-loop discriminative loop | Frequency / operator view |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt / attribute binding** | Good; color–object leaks possible | **Target:** stronger separation of layout vs texture | **Target:** catch errors before final polish | **Target:** global coherence |
| **Composition** | Variable | **Target:** locked low-res structure | **Target:** localized fixes | **Target:** low-freq global light/shape |
| **Implementation maturity in SDX** | Default path | **Partial** hooks | **Partial** (CLIP guard, etc.) | **Partial** (SFP loss only) |
| **Inference cost** | Baseline | +small planner or stage | +VLM forwards (high) | Depends on architecture |

---

#### Suggested research order

1. **Cheap structure** — dual-stage + stronger captions + optional **auxiliary edge/depth** head (trainable).  
2. **Cheap critic** — CLIP guard + masks from **simple** saliency before a full VLM.  
3. **Spectral** — ablate SFP on your data; then consider **true** spectral blocks in the denoiser if loss-only gains plateau.

---

#### Machine-readable themes

[`utils/modeling/architecture.py`](../utils/modeling/architecture.py): `geometric_latent_split_blueprint`, `discriminative_denoise_vlm_loop`, `frequency_domain_global_coherence`.

---

## Diffusion stack: high-leverage upgrades (“10× class” levers)

**What this doc is:** A **prioritized** list of **model / training / inference** directions that can compound into **large** quality or efficiency gains. “10×” here means **orders-of-magnitude-style improvement across the full stack** (data + representation + objective + sampler + alignment)—not a single architectural swap.

**Companion reads:** [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) (what is already wired), [LANDSCAPE_2026.md](#image-generation-landscape-architecture-and-workflows-2026) (industry + architecture + workflow), [`utils/modeling/architecture.py`](../utils/modeling/architecture.py) (theme → path map).

---

### 1. Lever hierarchy (where gains actually come from)

| Tier | Lever | Why it dominates |
| :--- | :--- | :--- |
| **A** | **Data & captions** | Better pairs beat wider nets; panel/text/comic data quality caps lettering and layout. |
| **A** | **Latent representation** (VAE vs RAE, bit depth, decode artifacts) | Ceiling on fine detail, text edges, and color stability. |
| **B** | **Training objective & time sampling** | Wrong noise levels or loss weights waste capacity (you already have logit-normal, min-SNR-soft, v-pred). |
| **B** | **Conditioning depth** (text encoders, fusion, length, control signals) | Prompt following and style lock scale with **how much** structured signal reaches DiT blocks. |
| **C** | **Backbone capacity** (depth, width, MoE, attention tricks) | Marginal if A/B are weak; essential once data/objectives are sane. |
| **C** | **Inference** (steps, distillation, guidance fixes) | Latency and “snap” to prompt without retraining the whole base. |
| **D** | **Post-train alignment** (DPO, reward fine-tuning) | Steers an **already good** base; poor substitute for broken data or latents. |

**Practical rule:** Before chasing exotic architectures, **measure** A/B with fixed compute (same steps, same resolution buckets)—see `python -m scripts.tools training_timestep_preview` and your bucket manifests.

---

### 2. Representation & autoencoder (latent stack)

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **RAE / semantic latents** | Denoise in a space aligned to **vision semantics**, not only pixel compression. | `--autoencoder-type rae`, `models/conditioning.py`, [MODEL_STACK.md](MODEL_STACK.md) | Medium (train stack) |
| **REPA-style alignment** | Auxiliary loss matches internal DiT states to **frozen ViT/DINO** features. | `--repa-weight`, `train.py`, DiT flags in `models/dit_text.py` | Low–medium (toggle + tune) |
| **Stronger VAE for text/edges** | Less smear in speech balloons, sharper line art—often **ae** upgrades beat **dit** width. | External VAE swap + compatibility tests in `train.py` / `sample.py` | High if retraining AE |
| **Multi-scale latent / MAR hybrids** | Coarse tokens + diffusion refine (masked AR in latent) | Not in-repo; overlaps “hybrid AR + diffusion” in [LANDSCAPE_2026.md](#post-diffusion-and-structural-shifts-late-20252026) | Research |
| **Pixel-space DiT (PixelDiT-class)** | Reduce single-VAE bottleneck; heavy compute. | Listed in [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) | Long-term branch |

---

### 3. Conditioning & multimodal (prompt → denoiser)

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **Multi-encoder text** | T5 + CLIP-L + CLIP-bigG fusion improves adherence and style. | `train.py --text-encoder-mode triple`, fusion weights in checkpoint | Medium |
| **Longer / structured captions** | JSONL `parts`, region captions, layout-aware training data. | `data/t2i_dataset.py`, [REGION_CAPTIONS.md](PROMPT_STACK.md#regional--layout-captions-jsonl) | Data pipeline |
| **Native multimodal transformer** | Single stack for extra modalities (sketch, layout tensor) before DiT. | Not implemented (an unwired prototype was removed) | Medium–high |
| **Reference-image conditioning** | IP-Adapter / cross-attn image tokens for **character lock** without full finetune. | Would be new adapter path on `DiT_Text` or parallel cond channel | High |
| **Regional / compositional CFG** | Different guidance per region (subject vs background). | Not in core sampler; inference research | Research |
| **Control paths** | Depth, edge, segmentation as extra channels (PixArt-style `control_cond_dim` hooks exist in `dit_text.py`). | `models/dit_text.py` (`control_cond_dim`), training glue | Medium |

---

### 4. Objectives, schedules, and auxiliary tasks

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **Non-uniform timestep sampling** | Emphasize hard noise levels (high-noise, logit-normal). | `diffusion/schedules.py`, `train.py` | Low |
| **Loss weighting** | Min-SNR, soft min-SNR, SNR-aware curriculum. | `diffusion/schedules.py`, [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) | Low |
| **v-prediction / epsilon** | Match parameterization to schedule and CFG behavior. | `--prediction-type` | Low |
| **Auxiliary structure heads** | Side losses on **edges, depth, segmentation** for comics/layout. | New heads on DiT final layer or intermediate — **not wired** | High |
| **Self-forcing / bootstrapping** | Train on model’s own high-quality generations (filtered). | New data loop + gating | Medium |
| **Flow matching / rectified flow** | Replace VP objective with velocity field on simpler paths. | **Separate trainer**; see [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) | Very high |

---

### 5. Architecture inside the denoiser (DiT)

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **MoE FFN / routed experts** | Scale params without dense FLOPs per token. | `models/dit.py`, `models/dit_text.py` (`moe_num_experts`, aux loss) | Medium |
| **Block-causal / AR hybrid** | Raster or block order bias for layout coherence. | `--num-ar-blocks`, `models/attention.py`, [AR.md](MODEL_STACK.md#block-wise-autoregressive-ar-generation) | Medium |
| **RoPE / register tokens / token routing** | Better long-sequence and capacity use. | Flags on `DiT_Text` (`use_rope`, `num_register_tokens`, routing) | Low–medium |
| **SSM / Mamba backbone** | Subquadratic sequence models for huge latents. | Not in-repo ([LANDSCAPE_2026.md](#post-diffusion-and-structural-shifts-late-20252026)) | Research |
| **Matryoshka / nested depth** | Train shallow→deep (depth growth) for stability. | Trainer schedule change (SANA-style ideas in [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026)) | High |

---

### 6. Inference, sampling, and distillation

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **Better ODE/SDE solvers** | DPM-Solver++, UniPC, adaptive step where supported. | `sample.py` / diffusion scheduler modules | Medium |
| **CFG rescale / dynamic threshold** | Reduce oversaturation at high guidance. | `sample.py` | Low |
| **Test-time compute** | N candidates + CLIP/edge/OCR pick. | `--num`, `--pick-best`, `utils/quality/test_time_pick.py` | Low |
| **Consistency / DMD / turbo distillation** | Few-step student matching teacher distribution. | **No trainer**; roadmap [IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training) | Very high |
| **RF-style test-time refinement** | Extra gradient steps on scores (heavy). | Research / optional script | Research |

---

### 7. Alignment after base training

| Idea | Mechanism | SDX touchpoints | Effort |
| :--- | :--- | :--- | :--- |
| **Diffusion-DPO / preference pairs** | Steer with win/lose image pairs. | Discussed in [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026); **no default trainer** | High |
| **Automated pair mining** | Use `test_time_pick` scores to build preference datasets. | `utils/quality/test_time_pick.py` | Medium |
| **Human / model critique loop** | `orchestrate_pipeline.py` style multi-stage review. | `scripts/tools/generate.py` | Workflow |

---

### 8. Suggested sequencing (example roadmap)

1. **Lock metrics:** FID/CLIPScore on a **fixed** eval prompt set + optional OCR on text panels.  
2. **Exploit existing flags:** resolution buckets, logit-normal / high-noise sampling, min-SNR-soft, triple text encoders, REPA weight sweep.  
3. **Latent audit:** Compare standard VAE vs RAE path on **text-in-image** and **line art** subsets.  
4. **Add control or reference conditioning** if character consistency is the main product gap.  
5. **Only then** widen DiT or add MoE depth—otherwise you may be **noise-fitting**.  
6. **Long fork:** flow-matching trainer or distillation student as a **separate** experiment branch, not mixed into the main `GaussianDiffusion` loop without tests.

---

### 9. Paper / keyword index (starting points)

- **Flow / OT:** rectified flow, flow matching, stochastic interpolation.  
- **Schedules / training:** EDM preconditioning, VP vs VE, cosine/sigmoid betas (you have multiple in `diffusion/schedules.py`).  
- **Efficiency:** FasterDiT (SNR-aware training), SANA (efficient DiT), DiT-Air.  
- **Alignment:** Diffusion-DPO, Pick-a-Pic, reward-model distillation.  
- **Structure:** ControlNet-class adapters, T2I-CompBench / GenEval for compositional testing.  
- **Latent:** RAE, REPA, tokenizer+flow (RecTok-class).

Keep **arXiv numbers** and product names out of the critical path—verify citations when you implement.

---

### 10. What *not* to do first

- Swapping to a **10× wider DiT** without fixing **caption noise** and **latent smear**.  
- Adding **flow matching** on top of an **unstable VP baseline**—stabilize schedules first.  
- **DPO** on a model that cannot yet render **readable text** in-domain—alignment cannot invent capability you did not train.

This roadmap should stay **living**: when you land a feature, move it from “idea” to **`architecture_map.py`** + [MODERN_DIFFUSION.md](#modern-diffusion--flow-ideas-20242026) “Implemented” tables.

---

## Image quality: research levers (2025–2026) + SDX map

There is no single trick for a “perfect” image model. Quality is a **stack**: data → training objective → capacity → sampler → guidance → evaluation → iteration. This note compresses **recent research directions** (with primary links) and maps them to **what SDX already exposes** vs **what would be new work**.

---

### A. Inference / sampling (often the biggest ROI)

| Idea | Why it matters | Pointers | SDX today |
|------|----------------|----------|-----------|
| **CFG rescale / dynamic CFG** | High CFG improves prompt adherence but causes saturation, contrast blow-ups, and texture collapse. | Oversaturation analysis [arXiv:2410.02416](https://arxiv.org/abs/2410.02416); dynamic CFG with feedback [arXiv:2509.16131](https://arxiv.org/abs/2509.16131) | `--cfg-rescale` on `sample.py`; Holy Grail schedules per-step CFG in `diffusion/sampling`. |
| **Geometry-aware CFG for flows** | Standard CFG on rectified-flow velocities can push states **off the data manifold** → artifacts. | Rectified-CFG++ [project site](https://rectified-cfgpp.github.io/) | If you train with flow matching, pair **`flow_matching_sample`** paths in `diffusion/gaussian_diffusion.py` with conservative CFG + rescale; **native Rectified-CFG++** would be new implementation work. |
| **Fast schedules that preserve geometry** | Step count interacts with trajectory curvature; “obvious” schedules waste budget early/late. | TORS / fast sampling analysis [arXiv:2603.00763](https://arxiv.org/abs/2603.00763) | Experiment with `--steps`, solver choice (`ddim` / `heun`), and flow solver (`euler` / `heun`). Automating **learned or TORS-like** schedules → research feature. |
| **Frequency-decoupled guidance (FDG)** | Apply different guidance strength to low vs high frequency components to reduce CFG side effects. | [arXiv:2506.19713](https://arxiv.org/abs/2506.19713) | Not bundled as FDG; would need spectral decomposition in the denoise update. |
| **Adaptive projected guidance (APG)** | Decompose CFG into parallel vs orthogonal components; down-weight the part that drives oversaturation. | [arXiv:2410.02416](https://arxiv.org/abs/2410.02416) related line (APG cited in community summaries) | Would be **new** sampler math; Holy Grail is the right hook point for per-step blending. |

**Practical inference checklist (no new code):** sweep `--cfg-scale` **down** before chasing architecture; add **`--cfg-rescale 0.6–0.85`**; increase steps if mid-frequency detail is mushy; use `docs/QUALITY.md` for Civitai-style failure modes.

---

### B. Training objective & optimization

| Idea | Why it matters | Pointers | SDX today |
|------|----------------|----------|-----------|
| **Rectified flow / flow matching training** | Straighter paths → fewer NFE at similar quality if the field is well trained. | Microsoft “Improving the Training of Rectified Flows”; MeanFlow [arXiv:2511.23342](https://arxiv.org/abs/2511.23342) | Flow training + sampling hooks exist in `diffusion/flow_matching.py` and `gaussian_diffusion.py`. |
| **Perceptual / semantic auxiliaries** | Pure MSE on latents underweights structure humans care about. | LPIPS-style diffusion losses [arXiv:2401.00110](https://arxiv.org/abs/2401.00110); PixelGen (LPIPS + DINO) [arXiv:2602.02493](https://arxiv.org/abs/2602.02493) | **REPA** and related alignment hooks in training (`--repa-weight`, see `train.py` / config). Full pixel-space LPIPS in latent DiT training → optional extension. |
| **Preference / DPO-style alignment** | Moves the model toward **human-preferred** samples beyond likelihood. | Diffusion-SDPO [arXiv:2511.03317](https://arxiv.org/abs/2511.03317); Curriculum DPO (CVPR 2025); Rethinking DPO [arXiv:2505.18736](https://arxiv.org/abs/2505.18736) | `gaussian_diffusion` contains **Diffusion-DPO** hooks; treat as **advanced**—mis-tuned preference training can **hurt** base quality (see SDPO motivation). |

---

### C. Data & captions (the silent multiplier)

| Idea | Why it matters | Pointers | SDX today |
|------|----------------|----------|-----------|
| **Autocuration / online selection** | Better examples per step can beat bigger but noisier corpora. | Autoguided curation [arXiv:2509.15267](https://arxiv.org/abs/2509.15267) | Use tooling for **hard-case mining / manifests** (`scripts/tools`, book pipeline) as lightweight curation; full online JEST-style selection is not default. |
| **Quality as conditioning (not only filtering)** | Throwing away weak data loses coverage; conditioning on quality scores can preserve diversity. | LACON [arXiv:2603.26866](https://arxiv.org/abs/2603.26866) | Would need dataset fields + model conditioning; **greenfield** relative to stock DiT text conditioning. |
| **Caption hygiene** | Wrong or lazy captions teach the wrong associations. | Industry curation writeups (e.g. Datology multimodal curation blog) | Caption utilities, prompt domains, tag reordering in `sample.py` / `data/`—improve **your** captions before chasing new arch. |

---

### D. Evaluation (if you cannot measure it, you cannot optimize it)

- **Human / pairwise** remains gold for “looks good”; automate with **fixed protocols** (same seeds, same prompts).
- **Model-based metrics** (CLIP alignment, aesthetic classifiers, OCR for text-in-image) are **gameable**—use as **diagnostics**, not the only objective.
- SDX: **`utils/quality/test_time_pick.py`**, Holy Grail pick metrics, **TCIS** (`docs/TCIS_OVERVIEW.md`) for multi-candidate ranking ideas.

---

### E. “Insane tier” backlog (high effort, high upside)

Ordered roughly by **impact if executed well** vs **engineering cost**:

1. **Manifold-safe CFG for flow models** (Rectified-CFG++-class) in the flow sampler loop.  
2. **APG / FDG** as optional guidance modes in `diffusion/sampling` / Holy Grail.  
3. **Learned or optimized timestep schedules** (TORS-class) exposed as presets.  
4. **Safeguarded preference training** (SDPO-class) on top of your base DiT.  
5. **Quality-conditioned training** (LACON-class) with explicit metadata channels.  
6. **Joint pixel+latent perceptual losses** where VRAM allows (PixelGen-class intuition, adapted to your stack).

---

### F. What SDX will not magically fix

- A **bad or tiny dataset** will cap any architecture.  
- **Extreme CFG** without rescale / schedule will **always** fight you.  
- **No compute** means fewer ablations; prioritize **inference sweeps** and **small controlled train runs** over rewriting the world.

### See also

- [Sampling experiment grids](#sampling-experiments-backlog)

- [`docs/QUALITY.md`](QUALITY.md) — operational fixes (blur, saturation, hands, text).  
- [`docs/HOW_GENERATION_WORKS.md`](HOW_GENERATION_WORKS.md#holy-grail-adaptive-sampling-how-it-fits-together) — adaptive sampling.  
- [`docs/TCIS_OVERVIEW.md`](HOW_GENERATION_WORKS.md#tcis--tri-consensus-iterative-synthesis) — multi-candidate + ViT critique loop.  
- [`docs/QUALITY.md`](QUALITY.md#quick-evaluation-recipe-holy-grail--manifests) — minimal eval recipe.

---

*Disclaimer: links point to papers/projects for follow-up reading; implementation in SDX is not implied unless listed in the “SDX today” column.*

---

## Sampling experiments backlog

Concrete **grids** to run when you change samplers, CFG, or flow settings. Pair with
[`examples/eval_prompts_baseline.json`](../examples/eval_prompts_baseline.json) and
[`examples/run_baseline_eval.py`](../examples/run_baseline_eval.py).

### Grid A — CFG vs rescale (VP / DDIM-style)

Hold: seed, steps (e.g. 30), resolution, checkpoint.

| Run | `--cfg-scale` | `--cfg-rescale` |
|-----|---------------|-----------------|
| A1 | 5.0 | 0 |
| A2 | 7.5 | 0 |
| A3 | 7.5 | 0.7 |
| A4 | 9.0 | 0.7 |
| A5 | 9.0 | 0.85 |

**Pass if:** A3–A5 reduce saturation vs A2 without losing subject (eyeball or pick-best).

### Grid B — Steps vs solver

| Run | `--steps` | notes |
|-----|-----------|--------|
| B1 | 20 | baseline |
| B2 | 30 | |
| B3 | 50 | default-ish |
| B4 | 30 | change solver / flow `heun` vs `euler` if applicable |

### Grid C — Flow matching sample path

When `--flow-matching` (or equivalent) sampling is on:

- Sweep **CFG down** first (flows punish naive CFG).
- Compare **Heun vs Euler** at fixed NFE budget.
- Track **manifold drift** symptoms: plastic textures, color locking (see [IMAGE_QUALITY_LEVERS_2026.md](#image-quality-research-levers-20252026--sdx-map)).

### Grid D — Holy Grail presets

For each preset you care about:

1. Fixed 5 prompts from the baseline pack.
2. Same seed list `[0,1,2,4,8]`.
3. Export filenames that include **preset id + seed + prompt id**.

Use directory diff or `test_time_pick` metrics if you have ViT checkpoints wired.

### What not to do

- Change **three** knobs at once (CFG + steps + scheduler) — you will not know what helped.
- Tune on **one** prompt; use the **baseline pack** minimum.

---

See also [eval_baseline_prompts.md](QUALITY.md#baseline-prompt-pack-for-quick-eval).

---

## Next-gen image generation: four design pillars (and SDX mapping)

**Purpose:** Capture a **reasoning-first** view of image generation (not only denoising) and map each pillar to **what exists in SDX today**, **partial hooks**, and **real gaps**. This is a **design document**, not a promise of a trained “super-model.”

**Core tension:** *Creative flexibility* (diverse, plausible samples) vs *logical precision* (prompt geometry, counts, colors, text). Pure latent denoisers behave like strong **statistical imitators**; moving “beyond guessing” usually means **extra structure** (planning, critique, frequency/objective changes, or preference learning).

---

### 1. Semantic–geometric dual backbone (e.g. global planner + DiT)

**Idea:** A **long-context, subquadratic** module (industry narrative: **Mamba / SSM**) consumes the prompt and produces a **global layout prior** (where objects, light, relations live). A **DiT** then **renders** detail conditioned on that prior.

**Why it’s compelling:** Transformers scale **O(n²)** in sequence length; very long prompts + global spatial reasoning stress vanilla DiT cross-attn. A **linear-sequence** planner is a natural research direction for “red ball left ↔ mirror right” style constraints.

**SDX today (partial, not a separate Mamba stack):**

| Mechanism | Role |
| :--- | :--- |
| **`--num-ar-blocks`** + block masks | Raster / block **causal** bias in the DiT (hybrid AR + diffusion angle). |
| **`--dual-stage-layout`** | **Coarse latent** pass → upscale → **detail** pass (layout-first *workflow*, same DiT). |
| **`ssm_every_n` / ViT-Gen SSM swap** | Lightweight **SSM-like token mixer** in some DiT blocks — **not** a full Mamba “architect” network. |
| **`size_embed_dim`**, **control**, **scene blueprint** text | Conditioning and prompt structure — not a learned global latent map. |

**Gap for “full” pillar:** A **dedicated** Mamba (or other SSM) **planner** that outputs a **tensor blueprint** fused into every DiT layer (or into a separate low-res latent) is **not** implemented.

---

### 2. In-loop VLM / discriminative correction

**Idea:** Every *k* steps, a **vision–language model** compares **current latents / decodes** to the prompt and applies a **gradient or latent correction** so mistakes (wrong color, extra fingers) don’t get “polished in place.”

**Why it’s compelling:** Diffusion is **path-dependent**; early mistakes are expensive to undo. A **critic** is the “step back and check the reference” loop.

**SDX today (partial — no full VLM, no autograd critic in the loop):**

| Mechanism | Role |
| :--- | :--- |
| **`--clip-guard-threshold`** + **`utils/generation/guidance.py`** | **CLIP** image–text cosine; optional **short second `sample_loop`** if score is low (coarse alignment, not segmentation). |
| **`--volatile-cfg-boost`** | Heuristic **CFG bump** when latent updates are **spiky** vs recent steps (cheap “unstable step” signal). |
| **`--pick-best`** (`utils/quality/test_time_pick.py`) | **Post-hoc** multi-sample scoring (CLIP / edge / OCR / combos). |
| **Refine / hires / SAG / PBFM** | Quality and structure nudges without a VLM. |

**Gap for “full” pillar:** A **mini VLM** (or detector) on a **decode preview** or **auxiliary head**, with **differentiable** or **latent-space** steering **inside** the denoising loop, is **not** wired. That requires model API hooks, memory budget, and training or frozen critic calibration.

---

### 3. Fourier / operator view (“infinite” resolution narrative)

**Idea:** Operate in **frequency** or **function** space (neural operators) so **global** low frequencies and **local** high frequencies are explicit; **resolution** can be reframed as **bandwidth** / sampling of the same underlying field.

**Why it’s compelling:** Fixed-grid CNN/ViT biases can **tile** or lose coherence when extrapolating resolution; frequency-aware objectives can emphasize **layout at high noise** and **texture at low noise**.

**SDX today (partial — training-side, not full neural operator diffusion):**

| Mechanism | Role |
| :--- | :--- |
| **`--spectral-sfp-loss`** + `diffusion/flow_matching.py` | **FFT-weighted** training loss on pred−target in **latent** space; timestep-dependent radial weights. |
| **`--dual-stage-layout` / `--hires-fix`** | **Multi-scale** inference paths (not the same as continuous resolution operators). |

**Gap for “full” pillar:** **NOD-style** denoising **directly** on Fourier coefficients with **zero-shot** arbitrary output grids is **not** implemented. SFP is a **loss shaping** tool on the **existing** VP latent grid, not a replacement forward process.

---

### 4. Direct preference alignment (DPO-style on images / latents)

**Idea:** Train on **pairs** *(A, B)* with human or teacher labels (“A follows prompt better”, “B is more aesthetic”) so the model **optimizes preferences**, not only likelihood under a static dataset.

**Why it’s compelling:** “Quality” is **subjective** and **multi-objective**; pairwise feedback matches how humans judge images.

**SDX today:** Standard **supervised** diffusion in **`train.py`**. Stage-2 **Diffusion-DPO** runner: **`python -m scripts.tools train_diffusion_dpo`** (frozen ref DiT, shared `t` + noise, **`safeguarded_dpo_preference_loss`** on per-sample VP losses). Helpers: **`utils/training/preference.py`**, **`preference_jsonl.py`**, **`preference_image_dataset.py`**. **`GaussianDiffusion.per_sample_training_losses`** supports the DPO term.

**Gap:** Optional **triple-encoder** / CLIP-bundle path in the DPO script; full paper-faithful variants (e.g. multiple noise draws).

---

### Comparison table (qualitative — not verified benchmarks)

| Pillar | Main failure mode addressed | SDX status |
| :--- | :--- | :--- |
| Planner + DiT | Weak long-range layout / composition | **Partial** (AR blocks, dual-stage, SSM swap, conditioning) |
| VLM critic | Wrong semantics locked in early | **Partial** (CLIP guard, volatile CFG, pick-best) |
| Fourier / operator | Resolution / global coherence | **Partial** (spectral SFP loss; not full NOD) |
| DPO / preferences | “Average” or bland dataset mimicry | **Partial** (loss + JSONL loader; no integrated trainer) |

---

### Suggested build order (if you implement for real)

1. **Stronger layout priors** — dual-stage + AR + data with captions that stress relations; optional **auxiliary layout** head.  
2. **Cheap critic** — CLIP guard + pick-best + orchestration; graduate to **small VLM** only if metrics justify cost.  
3. **Spectral / multi-scale losses** — ablate SFP vs baseline on your domain.  
4. **Preference fine-tune** — once base model is stable, add pairwise data and DPO-class objective.

---

### See also

- [`LANDSCAPE_2026.md`](#post-diffusion-and-structural-shifts-late-20252026) — broader 2025–2026 architecture themes.
- [`BLUEPRINTS.md`](#part-2-prompt-accurate-generation-gls-critics-frequency) — GLS, in-loop discriminative loops, frequency/FNO narrative.
- [`MODERN_DIFFUSION.md`](#modern-diffusion--flow-ideas-20242026) — VP vs flow, SFP, timestep sampling.  
- [`utils/modeling/architecture.py`](../utils/modeling/architecture.py) — machine-readable theme IDs (includes **nextgen_*** entries).

---

## Superior Stack — Research Map (2025–2026)

How recent diffusion alignment and inference papers map to SDX modules. This is an **implementation index**, not a claim of parity with frontier closed models.

### Alignment (training)

| Idea | Paper / trend | SDX module |
|------|---------------|------------|
| Timestep-aware DPO weighting | Rethinking DPO in Diffusion (AAAI 2026 oral) | `utils/training/preference.py` → `--timestep-dpo-weight high_noise` |
| Safeguarded / SDPO loser scaling | Diffusion-SDPO | `safeguarded_dpo_preference_loss` → `--safeguarded-dpo 0.85` |
| Dynamic reference (EMA ref) | Online DPO variants | `ema_update_reference` → `--ref-ema-alpha 0.01` |
| Preference mining + ViT | Internal benchmark flywheel | `vit_mining.py`, `hard_negative.py` |
| Model soup promotion | Model soups (Wortsman et al.) | `model_soup.py` |

### Inference (sampling)

| Idea | Paper / trend | SDX module |
|------|---------------|------------|
| Composable guidance stack | APG, FDG, CFG rescale, CFG-Zero★, TCFG, PAG, SLG and the guidance interval compose | `guidance_stack.py` (see below) |
| Feature / step caching | SpeCa, TeaCache | `feature_cache.py` → `--feature-cache-delta 0.012` |
| Batched CFG | Standard throughput win | `cfg_batched.py` |
| Multi-metric best-of-N | Test-time scaling | `composite_ranker.py` over `test_time_pick` terms (incl. `ai_look.py`, `art_critic.py`) |
| Self-correction refine | Align-and-refine loops | `self_correct.py` |
| Block-wise DiT cache | BWCache, TeaCache | `block_cache.py` → `--block-cache-thresh` |
| Consistency / LCM distill | Latent Consistency Models | `train_consistency_distill.py` |
| Flow-GRPO online RL | NeurIPS 2025 Flow-GRPO | `train_flow_grpo.py`, `flow_grpo.py` |
| LADD adversarial distill | Flux Schnell family | `train_ladd_distill.py` |
| TaylorSeer block forecast | ICCV 2025 cache-and-forecast | `taylor_cache.py` → `--taylor-cache` |
| DenseGRPO step rewards | 2026 dense alignment | `dense_grpo.py` → `--dense-grpo` |
| Adaptive Projected Guidance (APG) | Sadat et al., ICLR 2025 | `--apg-parallel-eta 0 --apg-momentum-beta -0.5 [--apg-norm-threshold r]` |
| Flash-GRPO | Iso-temporal grouping + grad rectification | `flash_grpo.py` → `--flash-grpo` |
| BranchGRPO | Tree branching rollouts | `branch_grpo.py` → `--branch-grpo` |
| Frequency-decoupled guidance (FDG) | Sadat et al. 2025, Laplacian split | `--fdg-low 0.6 --fdg-high 1.3` |
| CFG-Zero★ | Fan et al. 2025; zero-init on flow only | `--cfg-zero-star` |
| QSilk micrograin | Latent quantile clamp + detail gate | `latent_filters.py` → `--qsilk-micrograin 0.12` |
| TurningPoint-GRPO | Step incremental + turning-point credit | `turning_point_grpo.py` → `--tp-grpo` |
| DyDiT dynamic width | Timestep-wise width scaling | `dynamic_dit.py` → `--dynamic-dit-width` |
| SLA linear attention | Softmax/linear blend (elu+1 kernel) | `models/attention.py` → `--linear-attn-fraction 0.25` |
| LCM few-step student | Consistency distill inference | `--lcm-ckpt` + `train_consistency_distill` |
| Guidance interval | Kynkäänniemi et al. 2024, keyed on noise level | `--cfg-interval 0.28,5.42` |
| CFG-Rejection | Early-path filter for best-of-N | `cfg_rejection.py` |
| GRPO-Guard | Ratio-norm + timestep reweight | `grpo_guard.py` → `--grpo-guard` |
| TCFG | Kwon et al. 2025, SVD of [cond; uncond] | `--tcfg-damping 1` |
| SLG | Skip Layer Guidance (SD3.5) | `--slg-scale 2.8` |
| PAG | Perturbed-Attention Guidance (Ahn et al. 2024) | `--pag-scale 3` (batched with the CFG forward) |
| DBCache CFG split | Cache-DiT cond-only fingerprint | `--dbc-separate-cfg` + block cache |
| CFG-Rejection rerank | Early CFG gap multi-sample order | `--cfg-rejection-rerank` |

### Data flywheel

| Stage | SDX tool |
|-------|----------|
| Curate JSONL | `superior_curate` |
| Benchmark | `scripts/tools/bench.py` (`bench suite`) |
| Mine prefs | `scripts/tools/superior.py` (`superior mine-pairs`), `vit_mining.py` |
| DPO train | `train_diffusion_dpo.py` |
| Promote | `model_soup.py`, `run_flywheel.py` |

### Honest ceiling

No wrapper replaces **pretraining scale**, **proprietary data**, or **frontier VAE/text encoders**. The Superior Stack maximizes what you can extract from **your checkpoint + your corpus + your compute** via alignment loops and inference orchestration.

### Recommended defaults (`--preset superior`)

```bash
python sample.py --ckpt ... --preset superior --num 4 \
  --pick-best superior_composite \
  --apg-parallel-eta 0 --fdg-low 0.6 --fdg-high 1.3 --cfg-rescale 0.3 \
  --cfg-zero-star --qsilk-micrograin 0.1 \
  --local-rag-jsonl datasets/facts.jsonl --superior-self-correct
```

**Guidance composition** (`utils/generation/guidance.py`): every enabled technique runs, in this order. TCFG and the CFG-Zero★ scale adjust the uncond branch in the model's output space; both branches map to x0; APG (momentum, norm cap, projection) runs per FDG band; PAG and SLG add `s*(cond - perturbed)`; CFG rescale (Lin et al.) runs last. Outside the `--cfg-interval` sigma band the uncond forward is skipped. CFG-Zero★ zero-init applies to flow-matching checkpoints only.

DPO stage after benchmark:

```bash
python -m scripts.tools train_diffusion_dpo \
  --ckpt results/best.pt --preference-jsonl prefs.jsonl \
  --timestep-dpo-weight high_noise --safeguarded-dpo 0.85 \
  --ref-ema-alpha 0.01 --dpo-beta 300
```

---

## Agentic Research Map (2025–2026)

| Paper / trend | Idea | SDX module |
|---------------|------|------------|
| GenEvolve | Tool-orchestrated trajectories + visual experience distillation | `utils/agentic/agent.py`, `agentic_evolve` |
| VisionCreator | UTPC: Understand, Think, Plan, Create | `utils/agentic/agent.py` |
| VisionCreator-R1 | Act–Reflect–Think–Act self-correction | `utils/agentic/agent.py`, reflect loop in `agent.py` |
| Vision-SR1 | Self-reward GRPO (perception + language) | `utils/training/preference.py`, `train_flow_grpo` |
| Designer/Verifier/Reasoner | Multi-role pipeline | `utils/agentic/agent.py` + agent tools |

### Folder layout

```
utils/agentic/          ← operational agent + GenerationPlan DAG
utils/agentic/quality_loop.py ← quality tools the agent calls
config/presets.py
docs/HOW_GENERATION_WORKS.md
```

### When to use what

| Goal | Command |
|------|---------|
| One-shot best image | `agentic_generate` |
| Learn from multiple trajectories | `agentic_evolve` |
| Full self-improve loop | `agentic_flywheel` or `run_flywheel` |
| Manual control | `sample.py` + Superior flags |

---

## APEX — Adaptive Predictive EXtrapolation

Co-designed **sampler + scheduler** for SDX VP diffusion and rectified flow.

### Why not “another Euler”

Fixed steppers (Euler, DDIM, DPM++, UniPC) use a predetermined step sequence. APEX keeps a strong **static** default (reproducible / batchable) and optionally **adapts remaining timesteps** from embedded error, curvature, and phase.

Grounded in:

- DPM-Solver / DPM-Solver++ embedded order error (Lu et al.)
- Karras EDM σ spacing
- Align-Your-Steps mid-band density
- Classical adaptive ODE step control (atol / rtol)

### Components

| Piece | Path |
|-------|------|
| Core math + controller | [`diffusion/solvers.py`](../diffusion/solvers.py) |
| VP schedule `apex` | [`diffusion/solvers.py`](../diffusion/solvers.py) |
| Flow schedule `apex` | [`diffusion/solvers.py`](../diffusion/solvers.py) |
| Loop wiring | [`diffusion/gaussian_diffusion.py`](../diffusion/gaussian_diffusion.py) |

### Modes

#### Static (`--solver apex` + `--scheduler apex`)

1. Build three-phase timestep density (composition → reconstruction → refinement).
2. Each step: one model eval → DPM++ order-1 and order-2+ from the **same** `x0` (no extra NFE).
3. Normalized error `E = ‖x_hi − x_lo‖ / (atol + rtol‖x‖)`.
4. If `E` is large, blend toward the safer low-order update (CFG-stable).
5. Phase picks recommended multistep order (structure ≤2, mid ≤3, refine ≤2).

#### Adaptive (`--solver apex_adaptive`)

Same as static, plus after each step **rebuild the remaining** discrete indices with difficulty

`D ∝ 1 + αE + βκ + γV + spatial_topk(Δx0) + phase`

so the NFE budget stays fixed while spacing concentrates where the trajectory is hard.

### CLI

```bash
# Recommended APEX stills
python sample.py --ckpt ... --prompt "..." --steps 28 \
  --scheduler apex --solver apex

# Adaptive remaining-grid
python sample.py --ckpt ... --prompt "..." --steps 28 \
  --scheduler apex --solver apex_adaptive

# Flow
python sample.py --ckpt ... --flow-matching-sample \
  --flow-schedule apex --flow-solver apex --steps 28
```

### Honest limits

- Not a trained neural scheduler; difficulty is hand-designed.
- Spatial term is a **global** difficulty scalar, not per-pixel ODE stepping.
- True reject-and-retry with new NFEs is avoided to keep step budget predictable; soft blend + redistribute approximate classical adaptive solvers.
- Always benchmark against `dpmpp_2m` + `ays_dit` / `unipc` on your ckpt before changing defaults.

### Benchmark checklist

Compare at equal NFE (e.g. 12 / 20 / 28 steps):

1. `dpmpp_2m` + `ays_dit`
2. `unipc` + `ays_dit`
3. `apex` + `apex`
4. `apex_adaptive` + `apex`

Metrics: CLIP/prompt adherence, aesthetic/ViT pick scores, hand/face failure rate, wall time.

---

## Board-scale training + corpus retrieval, multi-character, Ideogram & edit skills

### Training corpora (your plan)

Target boards/sites for captions + images (JSONL rows with `image` + `text`/`tags`):

| Source | Notes |
|--------|--------|
| Danbooru | Tag-rich; best for structure + artist styles |
| Gelbooru / Safebooru | Similar tag schema |
| e621 | Furry / anthro; keep separate rating filters |
| Rule34 / Rule34xyz | NSFW-heavy; consent + rating gates at sample |
| DeviantArt / ArtStation | Style/moodboard diversity; noisier captions |
| Pixiv / others | Add via same JSONL schema |

**Caption tip:** mirror multi-character structure from `utils/prompt/scene.py` (`TRAINING_CAPTION_GUIDE`) so retrieval and T5 see the same ownership phrases you use at sample time.

### Corpus retrieval at sample time

Train on boards → at inference, **retrieve nearest rows** and inject:

1. Caption facts into the prompt (RAG)
2. Existing image files as InstantStyle / moodboard refs

```bash
# Build a slim index (optional)
python -m scripts.tools corpus_retrieve build \
  --jsonl data/danbooru_train.jsonl --out data/corpus_index.jsonl

# Sample with live retrieval
python sample.py --ckpt ... --prompt "1girl, catgirl, black lingerie, eipril style" \
  --corpus-jsonl data/corpus_index.jsonl --corpus-top-k 4 \
  --scheduler apex --solver apex
```

Code: `utils/generation/inventions.py`.

### Character consistency + multi-character anti-blend

```bash
python -m scripts.tools scene_compile \
  --cast examples/multi_character_scene.example.json \
  --print-argv

python sample.py --ckpt ... --cast-json examples/multi_character_scene.example.json \
  --anti-bleed --label-multi-character-sheets
```

Compiles:

- Per-character **outfit / pose / action** ownership phrases
- Anti-blend negatives (no outfit swap, face swap, merged bodies)
- Box layout regions for regional CFG / layout attention
- Optional per-character sheets + face refs

Code: `utils/generation/inventions.py` (+ existing sheets / InstantStyle).

### Ideogram-like text & layout

```bash
python -m scripts.tools scene_compile \
  --prompt 'poster with "SUMMER SALE" and "50% OFF"' \
  --ideogram --glyph-text "SUMMER SALE" --palette "#FF3366,#111111" \
  --print-argv
```

Uses design brief + glyph canvas + optional palette lock (`utils/generation/editing.py`).

### Photoshop-like edit skills

```bash
python -m scripts.tools scene_compile \
  --edit-image out.png --fix-region face --fix-region hands \
  --change-outfit "red jacket" --print-argv
```

Heuristic region masks → MDM inpaint argv (face/hands/clothing/background). Stack with `--reference-image` for identity lock.

### Perfect-gen combo

Corpus refs + cast + APEX:

```bash
python sample.py --ckpt ... \
  --cast-json my_cast.json \
  --corpus-jsonl data/corpus_index.jsonl \
  --scheduler apex --solver apex_adaptive \
  --steps 28
```

---

## Invention stack

`sample.py --invention-stack auto|all|off|<comma list>` rewrites the prompt with passes aimed at the
failure modes text-to-image models are known for. Code: `utils/generation/inventions/`.

| Pass | Module | What it does |
|------|--------|--------------|
| ORACLE | `failure_oracle.py` | Scores prompt risks (count, binding, negation, anatomy, glyph, multi-character, plastic, long prompt) and picks passes in `auto` mode. |
| NEGATRON | `negatron.py` | Moves "no X" / "without X" clauses out of the positive and into the negative. |
| BINDLOCK | `bindlock.py` | Binds colours to their nouns and negates attribute swaps. |
| COUNTGATE | `countgate.py` | Exact-count tags plus a grid box layout (regional CFG) for counted subjects. |
| HYDRA-SLOTS | `countgate.py` | One regional-CFG slot per counted instance. |
| ANATOMON | `priors.py` | Joint-chain anatomy language when the prompt shows a person. |
| FRICTION | `priors.py` | Anti-plastic microtexture language. |
| Scene priors | `priors.py` | Composition, depth layers, eye/teeth/contact-shadow/skin cues (people only), filmic highlights and lens look (photographic or unstated medium only). |

Runtime glue (`wire.py`):

- `--invention-adaptive-steps`: step budget from oracle risk (+3 steps per risk).
- Oracle risks fill unset guidance knobs: `--guidance-schedule piecewise` for count/binding/negation/multi-character
  prompts (not with `--holy-grail`), `--apg-parallel-eta 0` for binding/plastic prompts.
- `--invention-self-heal`: writes `self_healing_plan.json` (rewritten prompt, APEX scheduler/solver argv,
  regions to redo when the critic fires).
- `--cast-json` / `--corpus-jsonl`: multi-character cast compiler and training-corpus retrieval.

The OCR repair re-run receives the already-rewritten prompt, so invention, cast and corpus flags are not forwarded.

### CLI

```bash
python -m scripts.tools invention_lab --prompt "four red cubes left of blue spheres, no text, hands" --enable auto --print-argv
python -m scripts.tools invention_lab --prompt "2girls" --self-heal --print-argv
python -m scripts.tools invention_lab --atlas-out outputs/failure_atlas.jsonl   # failure-mode eval prompts
```

---

## Tools, repos, and inspiration

References and ideas we use to make the model and images better.

### Cloned reference repos (`external/`)

| Repo | What we take from it |
|------|----------------------|
| [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | Base transformer for diffusion; patch embed, timestep embed, adaLN. |
| [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | Structural conditioning (depth, edge, pose); control scale blending. |
| [black-forest-labs/flux](https://github.com/black-forest-labs/flux) | Modern diffusion design; fast inference; img2img/editing. |
| [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) | SD3 / MM-DiT; official stack; architecture and training ideas. |
| [PixArt-alpha/PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) | **PixArt-α:** T5 + DiT, efficient T2I on a fraction of SD-style budget. |
| [PixArt-alpha/PixArt-sigma](https://github.com/PixArt-alpha/PixArt-sigma) | **PixArt-Σ:** 4K T2I, weak-to-strong training, token compression attention. |
| [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image) | **Z-Image (S3-DiT):** Single-stream DiT; text and image in one sequence; efficient. |
| [willisma/SiT](https://github.com/willisma/SiT) | **SiT:** Flow matching + DiT backbone; interpolant framework; faster convergence. |
| [Alpha-VLLM/Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X) | **Lumina-T2I / Next-DiT:** Next-gen DiT scaling; multi-resolution; Rectified Flow. |

Run `scripts/setup/clone_repos.ps1` (Windows) or `scripts/setup/clone_repos.sh` (Linux/macOS) to clone all into `external/`. SDX does not import them at runtime.

### [PixAI.art](https://pixai.art/en/generator/image) (website — not a repo)

**PixAI.art** is the AI art generator site we take **prompt and tag style** from. It is **not** the same as PixArt-alpha (a separate T5+DiT research repo on GitHub; we don’t clone it and it’s unrelated to PixAI.art).

- **PixAI-style prompts**: Tag-based, `(tag)` / `((tag))` emphasis, `[tag]` de-emphasis, subject-first order.
- **Model lineup**: XL and DiT families; named lines (Haruka, Tsubaki, Hoshino, etc.).

### ComfyUI / A1111 ideas

- **CFG rescale / dynamic threshold**: Internal sampling uses fixed behavior; not exposed as CLI options.
- **Post-process**: Optional `--sharpen`, `--contrast` in `sample.py`. See `utils/quality/finishing.py`.

### Features that make images look better and match the user

1. **Negative prompt** — Model subtracts negative conditioning so it avoids unwanted content.
2. **Quality tags** — Boost tags like `masterpiece`, `best quality` in data so the model learns to improve quality.
3. **Style + ControlNet + LoRA** — Blended with scales (style_strength, control_scale, per-LoRA scale) so output isn’t messy.
4. **CFG rescale / dynamic threshold** — (Internal only; not exposed in sample CLI.)
5. **Post-process** — Optional sharpen and contrast for a crisper look (see `utils/quality/finishing.py`).
6. **Step-based training + save best** — More steps → better checkpoint; no epoch ceiling.
7. **Refinement** — Train on small-t to fix imperfections; optional refinement pass at inference.
8. **Img2img + inpainting** — Training and sample.py `--init-image`, `--mask`.
9. **Aesthetic / sample weighting** — In JSONL, use `weight` or `aesthetic_score` so high-quality samples count more in loss.

### Optional dependencies for quality

- **scipy** — For `--sharpen` (unsharp mask). Install with `pip install scipy` if you use post-process sharpen.

### File map and roadmap

- **[FILES.md](FILES.md)** — Map of all SDX project files and key files in external repos (DiT, ControlNet, flux, generative-models) with links.
- **[IMPROVEMENTS.md](#improvement-ideas-quality-fixes-and-better-training)** — Structured improvement ideas (quality, fixes to old techniques, features from other SD/DiT/FLUX models).

---

## SDX Website Design Brief

Image-generation platform design reference for SDX (not UI code). Each page includes a **detailed image-gen prompt** for mockups, concept art, or marketing visuals.

---

### Brand & Logo — SDX

**Name:** SDX — short, technical, memorable (diffusion / studio / eXperiment).

**Logo concept:** A minimal mark combining a **diffusion spiral** and a **layout grid**.

- Primary form: rounded-square app icon, 1024×1024 export, soft radial gradient deep indigo `#1a1a2e` → violet `#6c5ce7` with electric cyan `#00d4ff` bloom at 15% opacity in lower-right.
- Center glyph: three concentric arcs (grainy/noisy outer ring → crisp inner ring) that resolve into a **2×2 box grid** occupying the lower-right quadrant of the mark — represents regional box prompting and spatial control.
- Wordmark: **SDX** in geometric sans (Satoshi / General Sans / similar to Ideogram). Letter-spacing −2%. The **X** has a 12° diagonal cut through the right stroke (cross-attention metaphor).
- Favicon: grid-only variant at 32px; arcs dropped for legibility.
- Motion: on hover, arcs denoise from film grain to sharp over 400ms ease-out; grid boxes fade from 0% → 20% opacity staggered 50ms each.
- Dark-mode first. Light mode: white `#fafafa` field, indigo `#4c3aff` glyph, no gradient.

**Taglines:** “Layout-aware diffusion.” · “Prompt with precision.” · “Generate with structure.”

#### Logo image-gen prompt

> App icon and wordmark design for “SDX”, an AI image generation platform. Rounded square icon on dark indigo-violet gradient background. Center symbol: three concentric semicircular arcs suggesting noise transforming into a clean signal, merging into a 2×2 grid of four small squares in the lower-right quadrant, like a layout composition tool. Minimal flat vector style, subtle cyan glow accent, no 3D bevel. Beside the icon, wordmark “SDX” in modern geometric sans-serif, tight kerning, the letter X has a subtle diagonal slash cut. Premium tech startup aesthetic, Figma/Dribbble quality, dark mode, 4K, crisp edges, purple and cyan color palette only, no photographs, no clutter.

---

### Global Design Language

| Element | Specification |
|--------|----------------|
| **Viewport** | Desktop primary 1440×900; content max-width 1280px centered in canvas area |
| **Chrome** | Left sidebar 240px (64px collapsed), right inspector 320px, center fluid |
| **Surfaces** | Base `#0d0d12`, elevated `#16161f`, hover `#1e1e2a`, border `#2a2a38` |
| **Accent** | Primary violet `#6c5ce7`, secondary cyan `#00d4ff`, success `#2ecc71`, warning `#f39c12` |
| **Type** | UI 13–14px/1.4, prompts 15–16px/1.5, labels 11px uppercase tracking 0.06em muted `#9aa0b5` |
| **Radius** | Cards 12px, buttons 8px, inputs 8px, modals 16px |
| **Shadow** | `0 8px 32px rgba(0,0,0,0.45)` on modals only |
| **Icons** | Lucide-style 1.5px stroke, 20px default |
| **Imagery** | Masonry thumbs 1:1 default; blur-up skeleton `#1a1a24` → fade image |
| **Inspiration** | PixAI density for feeds; Ideogram clarity for create; Linear.app polish for settings |

**Sidebar nav items (icon + label):** Home · Create · Edit · Explore · Styles · Characters · Benchmarks · Taste · Workflows · Docs · Settings (bottom-pinned).

---

### Page 1 — Home / Landing

**Purpose:** Convert cold visitors; communicate differentiation (layout control, character lock, quality stack) without login.

**Anatomy:**
1. **Sticky header (64px):** SDX logo left; nav links Docs, Explore, GitHub; right: “Sign in” ghost button + “Start creating” filled violet CTA.
2. **Hero (100vh min):** Full-bleed background = slow crossfade of 5 curated generations (parallax 8%). Foreground left-aligned copy block max 560px:
   - Eyebrow pill: `v11 · Layout-aware diffusion`
   - H1: “Structure your imagination.”
   - Subhead (18px muted): “Draw regions. Lock characters. Refine automatically. SDX is diffusion built for composers—not gamblers.”
   - CTAs: primary “Start creating →”, secondary “Browse gallery”
   - Micro trust row: GitHub stars, open-source badge, “No credit card”
3. **Hero visual right:** Floating UI card mockup showing box layout on a landscape image (cyan/violet boxes labeled “sky”, “figure”, “foreground”).
4. **Feature strip (5 cols):** Icon + title + 2-line desc each — Box layout · Character sessions · Auto-refine · Style genomes · Regional inpaint.
5. **Comparison section:** Split panel “Plain txt2img” (muddy composition, wrong object placement) vs “SDX box layout” (clean regions, readable scene). Caption: “Same prompt. Different control.”
6. **Model cards row:** 3 cards — SDXL preset, Anime, Photoreal — each with sample thumb, bullet features, “Use in Create” link.
7. **Workflow diagram:** 4 steps — Prompt → Layout → Generate → Refine — horizontal timeline with icons.
8. **Footer (4 col):** Product, Resources, Community, Legal. Discord + GitHub social icons.

**States:** Hero carousel loading skeleton; CTA hover glow `0 0 24px rgba(108,92,231,0.4)`.

#### Page 1 image-gen prompt

> Full landing page UI mockup for “SDX” AI image generator website, desktop 1440px, dark theme. Sticky top navigation with SDX logo (diffusion spiral + grid icon), links, violet “Start creating” button. Hero section: left side large headline “Structure your imagination”, subtext about regional prompts and character lock, two call-to-action buttons. Right side floating product screenshot showing image generation canvas with colored bounding boxes labeled sky, character, foreground on a cinematic landscape. Below hero: five feature cards in a row with icons. Then split comparison section: left messy AI image badly composed, right clean well-composed AI image with visible layout boxes overlay. Modern SaaS marketing page, PixAI meets Ideogram aesthetic, indigo and cyan accents on charcoal background, subtle glassmorphism, professional typography, no lorem ipsum, high fidelity Figma export style, 4K.

---

### Page 2 — Create (main generator)

**Purpose:** Primary txt2img / img2img workspace. Highest traffic page.

**Anatomy:**
1. **Top bar (56px):** Breadcrumb `Create / New`; model dropdown “SDXL · best.pt”; aspect pills `1:1` `4:5` `3:2` `16:9` `Custom`; seed field with lock icon; steps `50` and CFG `7.5` as compact numeric steppers; queue indicator `0 jobs`.
2. **Center column:**
   - **Preview stage (min 480px tall):** Empty state = dotted border, icon, “Your image appears here”, 3 rotating tips. Populated = image with optional box overlay toggle. Bottom toolbar: download, upscale, send to Edit, copy seed, fullscreen.
   - **Prompt composer:** Tab `Prompt` | `Negative`. Large textarea 4 rows, placeholder “Describe the scene… Use (emphasis) or [de-emphasis].” Below: token count, lint chips (“⚠ negation detected: no glasses”, “✓ count: 3 people”).
   - **Generate bar:** Batch `1–4` stepper, est. time `~18s`, primary **Generate** button (violet, 120px wide), keyboard hint `⌘↵`.
3. **Right inspector (320px, tabs):**
   - **Layout:** Mini canvas 1:1 mirroring aspect ratio; tools: draw box, select, delete; region list with color swatch, name field, per-region prompt textarea, sketch upload thumb; “Import JSON layout” link; “Add global prompt” field.
   - **Style:** Holy-grail preset cards (balanced, photoreal, anime, aggressive); style pack dropdown; style genome slider `chaos 0.35`; LoRA slots ×3.
   - **Advanced:** Frontier toggle + serendipity dial; per-region CADS toggle; guidance schedule dropdown; scheduler `ddim` / `euler`.
   - **Character:** Session dropdown “None · Yuki · Marcus”; ref image strip (3 slots); “Lock prompt additions” textarea; negative lock field.
4. **Bottom drawer (collapsed default):** Negative prompt, dissect refs, control image upload.

**Interactions:** Drag box corners with snap to thirds; double-click region to edit prompt inline; live wireframe overlay on preview when Layout tab active; Generate pulses while running with step progress `Step 24/50`.

#### Page 2 image-gen prompt

> Detailed UI mockup of AI image generation “Create” workspace, app name SDX, dark mode desktop 1440px. Three-column layout: narrow left sidebar navigation with icons, large center canvas showing a generated fantasy portrait with three colored rectangular bounding boxes overlaid (cyan, violet, amber) each with small text labels, below canvas a prompt text area with sample text about a woman in a forest, emphasis syntax hints. Top bar with model selector, aspect ratio pills 1:1 4:5 16:9, seed and steps controls. Right panel 320px with tabs “Layout”, “Style”, “Advanced”, “Character” — Layout tab active showing list of regions with mini prompt fields and color swatches, draw tools. Bottom purple Generate button. Ideogram-style regional prompting interface combined with PixAI generator density, charcoal UI, violet and cyan accents, crisp Figma-quality components, subtle borders, professional product design, 4K.

---

### Page 3 — Edit / Inpaint

**Purpose:** Region fix, img2img, mask painting, adherence debugging.

**Anatomy:**
1. **Top bar:** Back to Create; mode pills `Inpaint` `Img2img` `Fix region`; undo/redo; zoom 50–200%.
2. **Center canvas:** Source image full width; mask layer in red 40% overlay where painted; optional box region outlines from layout JSON in cyan. Brush cursor circle. Layer toggles: Image / Mask / Boxes / Heatmap.
3. **Left tool rail (48px):** Brush, eraser, box select, pan; brush size slider 4–128px; hardness slider.
4. **Right panel:**
   - **Fix region dropdown:** lists region names from layout file (`subject`, `background`, `sky`).
   - **Regional prompt override** textarea (pre-filled from box).
   - **Strength** slider 0–1 (img2img), **Inpaint mode** `MDM` | `Legacy`.
   - **Adherence heatmap** toggle — when on, preview shows inferno colormap overlay where cross-attention landed.
   - **Generate fix** CTA (amber accent to distinguish from Create).
5. **Filmstrip bottom:** Original | Mask | Result v1 | Result v2 — click to compare.

**States:** Empty = “Upload image or send from Create”; heatmap legend gradient bar.

#### Page 3 image-gen prompt

> AI image editing and inpainting UI mockup, SDX app, dark theme. Center large image of a photorealistic street scene with semi-transparent red mask painted over a storefront sign area. Left vertical toolbar with brush, eraser, selection tools and brush size slider. Top mode tabs: Inpaint, Img2img, Fix region. Right sidebar: dropdown “Fix region: storefront_sign”, prompt override text field “neon sign reading OPEN”, strength slider, MDM inpaint mode toggle, “Show adherence heatmap” toggle enabled with orange-red heatmap overlay visible on part of the image. Bottom filmstrip comparing original, mask, and result thumbnails. Charcoal interface, cyan box outlines, amber generate button, professional creative tool aesthetic like Photoshop meets Ideogram editor, 1440px desktop, 4K Figma mockup.

---

### Page 4 — Explore / Gallery

**Purpose:** Discovery, remix culture, community feed (PixAI-style).

**Anatomy:**
1. **Header:** Search bar full-width “Search prompts, styles, users…”; sort `Trending` `New` `Top`.
2. **Filter chips (horizontal scroll):** All · Photoreal · Anime · Illustration · Layout-heavy · Character · Text-in-image · Frontier.
3. **Masonry grid:** 4 columns @ 1440px; card = image (hover scale 1.02), gradient scrim bottom, truncated prompt 2 lines, username `@pixelnova`, seed copy icon. Hover overlay buttons: Remix · Upscale · Variation · Save ♡.
4. **Modal on click:** Large image left; right meta panel — full prompt (copy button), negative, model, steps, seed, layout JSON download, “Open in Create”.
5. **Infinite scroll** loader at bottom = 3 pulsing skeleton cards.

**Card aspect mix:** 1:1, 4:5, 16:9 varied for visual rhythm.

#### Page 4 image-gen prompt

> Social gallery feed page for AI art platform SDX, dark mode, Pinterest/PixAI style masonry grid of diverse AI generated images — anime character, photoreal portrait, typography poster with text, fantasy landscape. Each card shows image thumbnail, two lines of prompt text, username, small icons. Top search bar and horizontal filter chips: Photoreal, Anime, Illustration, Layout-heavy. Hover state on one card showing overlay buttons Remix, Upscale, Save. Right side optional detail drawer open on one image with full prompt text and seed metadata. Charcoal background #0d0d12, subtle card borders, violet accent on active filter, clean sans typography, infinite scroll aesthetic, 1440px wide desktop UI mockup, 4K, high detail.

---

### Page 5 — Styles & Genomes

**Purpose:** Browse, invent, and apply style genomes; export galleries.

**Anatomy:**
1. **Header:** “Style Genomes” + “Invent new” violet button; search; view toggle grid/list.
2. **Featured carousel:** 3 large hero cards — auto-scrolling style packs with 3 sample images each.
3. **Grid cards (280px):** Genome name “Neon Noir 07”; DNA string preview monospace `palette:#0ff,#f0f;grain:0.2;line:ink`; 2×2 thumb quad; palette swatches row (5 circles); tags `cinematic` `high-contrast`; actions: **Apply** · **Compare** · **Export HTML**.
4. **Compare mode (split view):** Same prompt, slider A|B between two genome outputs; genome selectors below.
5. **Invent drawer:** Chaos level slider, mode `normal` `wild`; “Generate 3 candidates” — shows loading helix animation.

#### Page 5 image-gen prompt

> Style genome library UI for SDX AI art platform, dark theme. Grid of style cards each showing 2x2 grid of sample images, title “Neon Noir 07”, monospace DNA code string, row of five color palette swatches, tags cinematic and high-contrast, Apply button. Top featured carousel with three large style packs. Split-screen compare mode visible on right: same portrait prompt with before/after slider between two different color grades. “Invent new” button violet. Charcoal surfaces, cyan and purple accents, creative director tool vibe, 1440px desktop, Figma-quality component library, 4K mockup.

---

### Page 6 — Characters

**Purpose:** Persistent character identity across sessions (character lock).

**Anatomy:**
1. **List view (left 360px):** Character cards — avatar circle, name “Yuki”, lock badge, last used `2h ago`; + New character card dashed border.
2. **Editor (right):**
   - Name, description
   - **Prompt additions** (locked text): `silver hair, amber eyes, mole under left eye, red scarf`
   - **Negative lock:** `different hair color, aged up, extra fingers`
   - **Reference images:** 4-up upload grid with dissect preview
   - **Last outputs timeline:** horizontal scroll of 8 thumbs with seed under each
   - Sticky footer: **Generate with Yuki** (opens Create preloaded) · Save session JSON
3. **Empty state:** Illustration + “Create a character session to keep identity consistent across generations.”

#### Page 6 image-gen prompt

> Character management UI for SDX AI generator, dark mode. Two-panel layout: left list of character cards with circular avatars, names Yuki and Marcus, selected state violet border. Right editor panel for character “Yuki”: text fields for prompt additions describing silver hair amber eyes red scarf, negative prompt lock field, grid of four reference photo thumbnails of anime-style character, horizontal timeline of recent generated images. Large violet button “Generate with Yuki”. Session lock iconography, consistency-focused character sheet tool, charcoal UI, clean typography, visual novel / game asset pipeline aesthetic, 1440px desktop mockup, 4K.

---

### Page 7 — Benchmarks & Quality

**Purpose:** Power users — checkpoint comparison, regression tracking.

**Anatomy:**
1. **Header:** “Benchmark Suite” · Run config dropdown `default suite` · **Run benchmark** (shows queue).
2. **Leaderboard table:** Columns — Rank, Checkpoint, Composite score, Text OCR, Count accuracy, Anatomy, Sparkline 7d trend. Sortable. Row expand → per-case breakdown.
3. **Charts row:** Line chart composite over time (from benchmark_history); bar chart per prompt case.
4. **Prompt cases panel:** List `text_sign`, `people_count`, `anatomy_fullbody` with expected constraints badges.
5. **Actions footer:** Export CSV · Export JSON · Mine DPO pairs · Append to history.

**Colors:** Score ≥0.8 green, 0.6–0.8 amber, <0.6 red.

#### Page 7 image-gen prompt

> Data dashboard UI for AI model benchmarking, SDX platform, dark theme. Large table leaderboard comparing three model checkpoints with columns Rank, Composite score, OCR accuracy, Count accuracy, sparkline trend graphs in purple. Above table line chart showing score over time. Side panel listing prompt test cases: text_sign, people_count, anatomy_fullbody with badges. Top bar with Run benchmark button and suite selector. Developer analytics aesthetic like Grafana meets Linear, charcoal background, green amber red score colors, monospace numbers, 1440px desktop, 4K Figma mockup, no photographs.

---

### Page 8 — Taste / Preferences

**Purpose:** Personal preference learning — likes/dislikes → DPO training pairs.

**Anatomy:**
1. **Header:** “Taste Profile” · progress `47 ratings` · **Export DPO pairs** CTA.
2. **Split pane 50/50:**
   - **Liked** (green header): masonry of liked images, prompt snippet, score slider 0–1 on hover.
   - **Disliked** (muted red header): same layout, thumbs down icon.
3. **Rating mode toggle:** Grid select vs swipe (Tinder card stack with Like/Dislike buttons).
4. **Pair preview drawer:** Shows proposed win/lose pair from same prompt; confirm/export.
5. **Profile JSON preview** collapsible monospace block.

#### Page 8 image-gen prompt

> Personal taste training UI for SDX AI platform, dark mode. Split screen: left column “Liked” with grid of beautiful AI portraits and landscapes green accent header, right column “Disliked” with lower quality images red accent header. Center floating card swipe interface like Tinder for rating images. Top bar “Taste Profile” with Export DPO pairs button and rating count. Bottom drawer showing win vs lose image pair comparison with same prompt label. RLHF training data curation tool aesthetic, charcoal UI, green and muted red section colors, violet primary buttons, 1440px desktop, 4K mockup.

---

### Page 9 — Workflows (Comfy bridge)

**Purpose:** Bridge power users to ComfyUI node graphs; export CLI settings.

**Anatomy:**
1. **Header:** Import JSON · Export from last run · Paste CLI flags.
2. **Center:** Read-only node graph canvas (pan/zoom) — nodes `SDXLoadCheckpoint`, `CLIPTextEncode` ×2, `SDXSampler`, `SaveImage` connected with colored wires (Comfy-style but SDX branded node headers).
3. **Right JSON editor:** Syntax-highlighted workflow JSON with copy button.
4. **Bottom strip:** Last export timestamp; mapping note “Map SDXLoadCheckpoint to your Comfy DiT bridge node.”
5. **CLI paste area:** `sample.py` flags mirrored as form fields with sync toggle.

#### Page 9 image-gen prompt

> ComfyUI-style node workflow editor UI branded for SDX, dark theme. Center canvas with connected nodes: Load Checkpoint, CLIP Text Encode, SDX Sampler, Save Image, violet and cyan node headers, colored connection wires on dark grid background. Right panel syntax-highlighted JSON workflow. Top import export buttons. Bottom CLI flags text area showing sample.py command. Power user automation tool aesthetic, node-based programming interface, charcoal background, 1440px desktop, 4K Figma mockup.

---

### Page 10 — Docs & Learn

**Purpose:** In-app onboarding; reduce support burden.

**Anatomy:**
1. **Left doc nav (tree):** Getting started · Box layout JSON · Regional prompting · Frontier · Holy grail · Character sessions · CLI reference.
2. **Content area:** MDX-style prose, code blocks with copy, inline screenshots at 80% width.
3. **Interactive embed — Prompt diff:** Two textareas side by side; diff highlights `only_in_a` cyan, `only_in_b` violet, shared gray.
4. **“Try it” chips:** Open Create with this example preloaded.
5. **Search** `⌘K` command palette overlay for doc search.

#### Page 10 image-gen prompt

> Documentation page UI inside SDX AI app, dark theme. Left sidebar documentation tree navigation: Getting started, Box layout, Frontier, Character sessions. Main content area with heading “Regional Box Prompting”, prose text, code block showing JSON layout example with syntax highlighting. Below interactive two-column prompt diff tool with highlighted differing words in cyan and purple. Top search command palette overlay. Developer docs aesthetic like Vite or Stripe docs, charcoal background, clean typography, embedded screenshots of generator UI, 1440px desktop, 4K mockup.

---

### Page 11 — Account / Settings

**Purpose:** Defaults, API (future), cache, notifications.

**Anatomy:**
1. **Sections (vertical nav):** General · Models · Generation defaults · Notifications · API keys (locked) · Appearance.
2. **General:** Display name, email, theme `Dark` `Light` `System`.
3. **Models:** Checkpoint path browser, default preset `sdxl`, VRAM hint badge.
4. **Generation defaults:** steps, CFG, holy-grail preset, auto-refine count, negative prompt template.
5. **Notifications:** Email on queue complete toggle; browser push toggle.
6. **Danger zone:** Clear cache, reset taste profile — red outline buttons.

#### Page 11 image-gen prompt

> Settings page UI for SDX AI application, dark mode, Linear app style. Left section navigation: General, Models, Generation defaults, Notifications. Main panel showing form fields: default checkpoint path, preset dropdown SDXL, theme selector dark light system, sliders for default steps and CFG, toggle switches for email notifications. Clean minimal forms, charcoal surfaces, violet toggle active state, subtle dividers, 1440px desktop settings aesthetic, 4K Figma mockup.

---

### Mobile (375×812)

**Bottom nav (56px + safe area):** Create · Explore · Characters · Profile — center Create icon larger.

**Create mobile:** Full-screen prompt first; FAB **Generate**; layout via bottom sheet “Add region” simplified 2-box mode; preview swipe up fullscreen.

**Explore mobile:** 2-column masonry; pull-to-refresh.

**Characters mobile:** Stacked list → tap opens full-screen editor sheet.

#### Mobile image-gen prompt

> Mobile iPhone UI mockup for SDX AI image generator app, dark mode, bottom navigation bar with Create Explore Characters Profile, center screen showing prompt input and generated square image with two colored bounding boxes, floating purple Generate button, iOS status bar, 375px width, modern native app aesthetic, indigo and cyan accents, 4K.

---

### Master prompt — full product suite (all pages)

> Comprehensive UI design system presentation for “SDX” AI image generation platform, dark mode, showing six screens arranged on charcoal canvas: (1) marketing landing page with hero and feature grid, (2) main Create workspace with bounding box regional prompts, (3) inpaint editor with brush mask and heatmap, (4) PixAI-style explore masonry gallery, (5) character session manager with reference images, (6) style genome library grid. Consistent violet #6c5ce7 and cyan #00d4ff accents, geometric SDX logo with diffusion spiral and grid, Ideogram clarity plus PixAI density, Figma-quality components, 1440px desktop screens, subtle glassmorphism, professional SaaS product design, 8K presentation board, no watermarks.

---

*This document describes product/design intent only. Implementation is separate from the SDX training/sampling codebase.*
