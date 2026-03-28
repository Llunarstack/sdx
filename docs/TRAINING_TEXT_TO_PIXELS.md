# Training: text tokens ↔ latent “pixels” (what’s actually coupled)

This doc matches a mental model—**words you care about (“tokens”)** and **image structure (“pixels”)**—to what SDX does during **training**, so you can steer **faithful dissection** vs **original / varied** outputs.

---

## Two kinds of “tokens”

| Side | What it is in SDX | Notes |
|------|-------------------|--------|
| **Text tokens** | Sequence from the text encoder (e.g. T5 `last_hidden_state`, shape `(B, L, D)`). With **triple text** + fusion, T5 is fused with CLIP-style signals before `text_embedder` in DiT. | One “token” ≈ a **subword** (SentencePiece), not always one English word. Order and punctuation matter. |
| **Vision / patch tokens** | Image → VAE (or RAE bridge) → **latent** → DiT **patch embedding**: one vector per spatial patch (your “pixels” are really **latent patches**, not raw RGB). | Cross-attention lets every patch **query** the full text sequence. There is **no fixed** word→region map unless you add losses or metadata. |

So: the model **learns** which text positions matter for which image regions from data + objective (noise prediction), not from a built-in parser that “cuts up” the scene.

---

## Making “dissection” more accurate (faithful structure)

1. **Captions that already dissect the scene**  
   Compositional prompts in training data (subject, attributes, pose, background, lighting, style) give the cross-attn a cleaner signal than one vague sentence. Use JSONL `caption` / `positive` fields consistently; optional tools: [`utils/prompt/scene_blueprint.py`](../utils/prompt/scene_blueprint.py), [`utils/prompt/content_controls.py`](../utils/prompt/content_controls.py), [`docs/PROMPT_STACK.md`](PROMPT_STACK.md).

2. **Negative prompts**  
   Training with `encoder_hidden_states_negative` teaches what to **suppress**; improves separation of “wanted” vs “unwanted” concepts ([`train.py`](../train.py) passes both into DiT).

3. **Per-token emphasis (train + infer)**  
   `(word)` → scale **1.2**, `[word]` → **0.8** on cross-attn conditioning after `text_embedder` (`models/dit_text.py`). **Inference:** `sample.py` builds weights from the cleaned prompt. **Training:** enable **`--train-prompt-emphasis`** so JSONL captions with the same syntax are stripped for T5 and **`token_weights`** are passed in the loop (matches sampling). **Triple text:** two extra CLIP tokens get weight **1.0**. Implementation: [`utils/prompt/prompt_emphasis.py`](../utils/prompt/prompt_emphasis.py). If the tokenizer has no `offset_mapping`, weights are skipped (captions are still cleaned when the flag is on).

4. **Control / structure**  
   If you train with **control** inputs (edges, depth, pose), patch tokens get an extra structural prior so text doesn’t have to carry geometry alone.

5. **Stronger alignment (research direction)**  
   Auxiliary objectives (e.g. contrastive patch–phrase, grounding boxes, segmentation masks) are **not** required defaults in SDX but are how some systems get sharper “this word → that region” behavior.

---

## Making outputs more **original** (less copy-paste, more variation)

1. **`creativity` / diversity embedding**  
   If you train with `--creativity-embed-dim` > 0, the model gets a scalar **creativity** channel; at sample time use `--creativity` ([`sample.py`](../sample.py)). **`--creativity-jitter-std`** (training) and **`--creativity-jitter`** (sampling, especially with `--num` > 1) add Gaussian noise so each image sees a slightly different creativity value—less “same face” repetition when the checkpoint supports creativity.

2. **Originality phrases (train + sample)**  
   **`sample.py --originality 0.3`** inserts random tokens from **`ORIGINALITY_POSITIVE_TOKENS`** (composition / lighting / art-direction) after subject tags—same insertion rules as training. **`train.py --train-originality-prob 0.15 --train-originality-strength 0.5`** applies that stochastically to JSONL captions so the DiT learns from “fresher” prompts, not only at inference. Shared code: [`utils/prompt/originality_augment.py`](../utils/prompt/originality_augment.py).

3. **Caption dropout (and schedules)**  
   Randomly dropping or shortening captions forces the model to rely on latents + prior → more **inventive** fill-in; scheduled dropout is documented in [`docs/IMPROVEMENTS.md`](IMPROVEMENTS.md) (`--caption-dropout-schedule`).

4. **Data diversity**  
   Originality is largely **data**: many subjects, styles, and compositions; dedup ([`scripts/tools/data/data_quality.py`](../scripts/tools/data/data_quality.py)) reduces memorization of single images.

5. **Min-SNR, timestep sampling, EMA**  
   Stabilize learning so the model generalizes rather than fitting spurious caption–image quirks ([`docs/MODERN_DIFFUSION.md`](MODERN_DIFFUSION.md)).

---

## Practical recipe

| Goal | Lean on |
|------|--------|
| Crisper “this part of the prompt → that part of the image” | Better captions; negatives; optional control training; `( )` / `[ ]` at train (`--train-prompt-emphasis`) and sample |
| More novel / less dataset memorization | Caption dropout; diverse data; `creativity_embed_dim` + `--creativity` / jitter; `--originality` or `--train-originality-prob` |
| Understand the stack end-to-end | [`docs/HOW_GENERATION_WORKS.md`](HOW_GENERATION_WORKS.md) (diagram + wiring §13) |

---

## Caption dropout + emphasis

If you use **`--caption-dropout-schedule`**, dropped samples still get **uniform** token weights (all **1.0**) for that step so conditioning stays consistent with the empty-caption embedding.
