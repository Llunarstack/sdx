# Competitive Analysis & Solution Roadmap

> What the leading text-to-image / video systems do well and badly, how **SDX**
> compares as a **trainable framework**, and concrete architecture-specific
> levers in this repo (`models/`, `diffusion/`, `vit_quality/`, `utils/`,
> `pipelines/video/`, `native/`).
>
> **Snapshot:** mid‑2026 public products and open weights. Names, Elo boards,
> and licenses churn; treat version labels as approximate. Prefer *capability
> classes* over marketing demos. README summary: [../README.md](../README.md#how-sdx-compares).

---

## 0. Landscape research (mid‑2026)

### 0.1 Closed image leaders — what they can do

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

### 0.2 Open-weight image leaders — what they can do

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

### 0.3 Video leaders — what they can do

| System | What it can do well | Gaps |
|--------|---------------------|------|
| **Veo 3.1** (Google) | Hosted fidelity + **native audio**; strong all-round clips | Closed; cloud; watermarking / ToS; no open director OS |
| **Kling 3** | Motion, multi-shot value, competitive 4K-class paths | Closed API; residency / ToS; not forkable |
| **Runway Gen-4.5** | Motion brush, editor UX, character/control workflows | Hosted product economics; not a training framework |
| **Luma Ray / Seedance** | Atmospheric I2V; multi-shot narrative experiments | Closed; limited scientific reproducibility |
| **Wan / LTX** (open-ish) | Emerging **self-host** video weights | Model-centric; little open continuity / scene-graph film studio |

**Pattern:** video quality crowns sit with hosted labs. Open video is catching up on *weights*, not on *open directing*.

### 0.4 Cross-product capability matrix

Legend: **✓** strong · **◐** partial · **ext** via community · **✗** no · **n/a** wrong shape

#### Image

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

#### Video

| Capability | Veo | Kling | Runway | Luma | Wan/LTX | **SDX** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Hosted clip + audio quality | **✓** | **✓** | ✓ | ✓ | ◐ | n/a |
| Editor / motion-brush UX | ◐ | ◐ | **✓** | ◐ | ✗ | ◐ |
| Open JSON scene director | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Continuity validators + retry | ◐ | ◐ | ✓ | ◐ | ✗ | **✓** |
| Forkable train+compose stack | ✗ | ✗ | ✗ | ✗ | ◐ | **✓** |

### 0.5 What SDX uniquely (or rarely) offers

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

### 0.6 What SDX should *not* claim

- Winning Midjourney / Veo aesthetics Elo without a published bake-off.
- Being a drop-in replacement for FLUX.2 or Ideogram OCR demos out of the box.
- Shipping a single mega-checkpoint that “is” GPT Image / Kling.

SDX’s bet: **own the loop** (data → train → sample → critique → direct). Theirs: **own the default pixels**.

### 0.7 Sources (orientation, not endorsement)

Public mid‑2026 surveys and vendor docs informing §0 (verify before purchasing decisions):

- Teamday — *Best AI Image Models 2026* (job-based ranking; GPT Image 2, MJ V8, Ideogram 4, FLUX.2, Seedream, Firefly, …)
- Thunder Compute / BentoML / Second Talent — open-weight guides (FLUX.2, SD3.5, Qwen-Image, Z-Image, Hunyuan, …)
- BFL docs — FLUX.2 klein LoRA training
- AI video roundups — Veo 3.1, Kling 3, Runway Gen-4.5, Luma, Seedance, Wan/LTX; Sora API sunset messaging in 2026 press
- In-repo: [research/LANDSCAPE_2026.md](research/LANDSCAPE_2026.md)

---

## 1. The landscape at a glance (legacy early‑2026 notes)

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

## 2. Cross-cutting weaknesses → sdx opportunities

Each item = a weakness shared by most competitors, then the concrete sdx lever.

### 2.1 Prompt adherence collapses on long / compositional prompts
Everyone degrades as prompts get longer, more relational ("A left of B, holding
C"), or contain counts ("exactly three"). Attention dilutes; late tokens get
ignored.

- **What sdx already has:** `prompt_reinject_every_n`, timestep-aware prompt
  scaling (`prompt_early_scale`/`late_scale`) in `dit_text.py`, TCIS count/OCR
  scoring in `hybrid_dit_vit_generate.py`, and now a **CLIP-conditioned ViT
  adherence head** (see `vit_quality/text_encoder.py`).
- **Solution A (evaluator → generator feedback):** the ViT adherence head now
  sees real prompt semantics. Wire its *per-region* disagreement back into TCIS
  as a spatial reweighting signal, not just a scalar accept/reject. Land in
  `utils/superior/` + `hybrid_dit_vit_generate.py`.
- **Solution B (structured prompts):** add a lightweight prompt→scene-graph
  parser (subject / relation / attribute / count) feeding regional conditioning
  (`utils/generation/regional_box_prompting.py` already exists). This is a
  *small tool*, not a model — see §4.
- **Solution C (count adherence):** competitors have no count guarantee. sdx can
  close-loop it: generate → native object-count metric → resample the failing
  region. The counter belongs in `native/` (fast, deterministic) — see §5.

### 2.2 Text-in-image rendering (only Ideogram/Recraft nail it)
Most diffusion models still smear multi-word text.

- **Solution:** sdx already has an OCR term in TCIS consensus. Promote it to a
  *first-class glyph-conditioning* path: render the requested string to a mask
  and inject via the existing ControlNet stack (`control_image`/`control_type`
  in `dit_text.py`), then verify with the OCR score and iterate. No new backbone
  needed — it reuses control + critic infra.

### 2.3 "House style" sameness (MJ, DALL·E, Imagen all have a detectable look)
Their strength is also a ceiling: outputs are recognizable and hard to push off.

- **Solution:** sdx's `creativity_embed_dim` + `style_embed_dim` + MoE experts
  are a genuine differentiator. Formalize a **style-dispersion objective**:
  during preference mining (`utils/superior/vit_mining.py`), reward *intra-prompt
  diversity* that stays on-prompt (ViT adherence high, pairwise embedding
  distance high). This directly attacks the sameness competitors can't escape.

### 2.4 Controllability is bolted on, not native
FLUX/SD control needs external ControlNets/LoRAs; MJ/DALL·E barely expose it.

- **What sdx has:** native ControlNet injection, REPA, AR blocks, RAE latents.
- **Solution:** lean into this as the *product wedge*. Ship a single
  "control bundle" API (pose + depth + glyph + region + count) that competitors
  can't match without a plugin zoo. The pieces exist; the gap is a unified,
  documented surface.

### 2.5 No self-critique loop (all competitors are single-shot)
None of the majors verify their own output against the prompt and retry. This is
sdx's biggest structural edge.

- **What sdx has:** TCIS (`docs/TCIS.md`) + ViT critic + auto/flywheel loops.
- **Solution:** make the critic *cheaper and better* so the loop can run more
  iterations per second: (a) the CLIP-semantic adherence head (done), (b) native
  scoring kernels for the non-neural TCIS terms (count/saturation/diversity) so
  the consensus math isn't a Python bottleneck (§5), (c) uncertainty-triggered
  early-exit (already partially present) tuned against the new head.

### 2.6 Anatomy / small-face / far-figure degradation
Shared weakness (SD3.5 especially) at distance and in crowds.

- **Solution:** multi-crop high-res ViT scoring. Today `vit_quality` resizes
  everything to 224 and `book_model_readiness.py` already warns about it. Add a
  tiled/multi-crop scoring path so the critic actually *sees* small faces, then
  let TCIS resample crops that fail. Land in `vit_quality/tta.py` + dataset.

---

## 3. Where sdx is already ahead — protect and sharpen

1. **Closed-loop critic (TCIS).** No major competitor self-verifies. Keep
   investing here; it compounds.
2. **Separable adherence vs quality heads.** Competitors conflate "good" and
   "on-prompt"; sdx scores them independently — enables targeted fixes.
3. **Native-accel discipline.** `native/rust/` + ctypes/NumPy-fallback is a real
   moat for loop throughput. Extend it (§5).
4. **AR-regime awareness.** ViT↔DiT AR calibration (`ar_block_conditioning.py`)
   is unusually principled; keep the evaluator and generator regimes in sync.

---

## 4. Small tools to build (Python, "not huge") — model-improving utilities

These improve the *model via better data/eval/loops*, not by shipping big weights.

| Tool | Purpose | Home |
|---|---|---|
| **Prompt scene-graph parser** | subject/relation/attribute/count → regional cond | `utils/prompt/` |
| **Caption↔image alignment auditor** | flags mislabeled training pairs using the CLIP featurizer already added | `scripts/tools/data/` |
| **Adherence-failure miner** | runs the CLIP-ViT head over a manifest, buckets the worst prompts by failure type (count/relation/text) → targeted training set | `vit_quality/` |
| **Diversity/duplication filter** | near-dup detection on training data via embedding + pHash to fight style collapse | `scripts/tools/data/` |
| **Eval harness vs competitors** | fixed prompt suite scoring adherence/text/count/anatomy so we track progress on *their* weak points | `scripts/tools/` |

Every one of these is bounded, testable, and reuses infra that now exists.

---

## 5. Native (Rust) speedups — real hotspots, small crates

The `native/rust/` pattern (C-ABI cdylib + ctypes wrapper + NumPy fallback,
see `docs/NATIVE_AND_SYSTEM_LIBS.md`) is the right home. Candidates, in rough
value order — **each should be profiled first** (`native/benchmark_suite.py`):

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

## 6. Suggested sequencing

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
