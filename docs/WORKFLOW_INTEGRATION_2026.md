# Workflow integration & mathematical efficiency (March 2026)

---

## Disclaimers

| | |
| :--- | :--- |
| **Medical** | This page is **informational only**. It is **not** medical advice, diagnosis, or treatment. For health decisions, **consult a licensed professional**. |
| **Technical / legal** | **Product and assistant names** in industry commentary are **illustrative**; capabilities and benchmarks **change frequently**. SDX does **not** integrate third-party APIs or weights unless explicitly documented. |

---

## Context

As of **early 2026**, much of the field’s energy has shifted from “make prettier pixels” to **workflow integration** (fits real pipelines) and **mathematical efficiency** (fewer steps, better scaling, less waste). Below are **themes** often discussed in that context — with a **straight mapping to SDX** where applicable ([`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py), extended rows `workflow_*` / `llada_*` / `test_time_*` / `live_grounding_*`).

---

## 1. High coherency & “structural” understanding (illustrative product stacks)

**Talking points in industry commentary**

- **Reverse-engineering / 3D-style logic:** Some stacks claim stronger **consistency** under **viewpoint or pose** changes (e.g. “side view” vs “above”) without identity or clothing **melting** — often via **better conditioning**, **multi-view data**, or **internal 3D/pose** priors (details vary by vendor).
- **4K native:** **Single-pass** high resolution vs **upscale-only** workflows — shifts professional photo and arch-viz pipelines.

**SDX**

- **Native resolution:** `sample.py` **width/height**, **VAE tiling** for large decodes; training **`--resolution-buckets`** ([`data/t2i_dataset.py`](../data/t2i_dataset.py)) for multi-aspect / multi-res **single-GPU** training.
- **Multi-angle consistency:** **Not** a built-in 3D/pose engine; use **data + captions**, **Control**-style paths where supported, **character lock** helpers ([`utils/consistency/character_lock.py`](../utils/consistency/character_lock.py)), and **[`docs/LANDSCAPE_2026.md`](LANDSCAPE_2026.md)** for consistency themes.

---

## 2. Discrete diffusion & text (e.g. “LLaDA”-class ideas)

**Concept**

- **Classic LLMs:** often **autoregressive** (one token after another).
- **Classic image models:** often **diffusion** in latent or pixel space.
- **Unified / “LLaDA”-style narratives:** **diffusion-style** processes applied to **text** or **shared** weights across modalities — **whole-sequence** reasoning vs strict left-to-right only.

**Why it matters**

- Drives interest in **one stack** for **prompt reasoning** + **image generation** (tighter **adherence** in complex instructions).

**SDX**

- **Text:** **T5** (+ optional **CLIP fusion**) — **autoregressive** text encoding, **not** discrete diffusion over text tokens in this repo.
- **Image:** **DiT** + **VP diffusion** — see [`docs/MODERN_DIFFUSION.md`](MODERN_DIFFUSION.md), [`docs/ARCHITECTURE_SHIFT_2026.md`](ARCHITECTURE_SHIFT_2026.md).

---

## 3. Test-time compute (“think before paint”)

**Concept**

- **Inference scaling:** extra **compute at sample time** (e.g. internal **critique** / **refine** loops), inspired by **reasoning**-style LLM narratives — **layout vs anatomy** checks before finalizing.

**SDX (partial)**

- **Refinement** pass in **`sample.py`** (small **re-noise** + **denoise**).
- **Multi-candidate + selection:** `--num K` + **`--pick-best`** ([`utils/quality/test_time_pick.py`](../utils/quality/test_time_pick.py)) — e.g. `clip`, `edge`, `ocr`, `combo`, **`combo_exposure`**.
- **Pipeline wrapper:** [`scripts/tools/ops/orchestrate_pipeline.py`](../scripts/tools/ops/orchestrate_pipeline.py), [`utils/generation/orchestration.py`](../utils/generation/orchestration.py).

This is **not** a full internal **latent self-critique** loop like a closed proprietary stack — it’s **explicit** test-time **scoring** and **optional** refinement.

---

## 4. Real-time grounding & RAG-for-images

**Concept**

- **Stale training data:** mitigated by **retrieval** — **facts** or **references** injected at **generation time**.
- **“Live web”** narratives: **product-layer** integrations (policy, safety, licensing) — **not** “just bigger weights.”

**SDX (partial)**

- **Merge user facts into the prompt:** [`utils/prompt/rag_prompt.py`](../utils/prompt/rag_prompt.py) (`merge_facts_into_prompt`, `load_facts_from_jsonl`). **You** supply retrieval; the model sees **frozen** encoders after that.
- **No** built-in web browser or API in core `train.py` / `sample.py`.

---

## 5. Mamba / SSM backbones (linear scaling)

**Concept**

- **Transformers:** **quadratic** cost in sequence length — painful at **8K+** or long **video**.
- **State-space models (e.g. Mamba):** **subquadratic** / **linear** scaling narratives — **candidate** backbones for **vision** encoders or decoders.

**SDX**

- **DiT** uses **transformer** attention (see [`docs/ARCHITECTURE_SHIFT_2026.md`](ARCHITECTURE_SHIFT_2026.md)). **No Mamba** backbone in-tree **today**.

---

## See also

- [`LANDSCAPE_2026.md`](LANDSCAPE_2026.md) — production & authenticity trends  
- [`ARCHITECTURE_SHIFT_2026.md`](ARCHITECTURE_SHIFT_2026.md) — flow, bridges, hybrid AR+DiT, RAE  
- [`MODERN_DIFFUSION.md`](MODERN_DIFFUSION.md) — timestep sampling & roadmap  
- [`utils/architecture/architecture_map.py`](../utils/architecture/architecture_map.py) — theme IDs + `workflow_*` rows  
