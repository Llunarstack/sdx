# How Our Model Generates an Image

This doc walks through the pipeline from **prompt → image** at inference (e.g. `sample.py`). Training learns the same forward pass in reverse (denoising).

**Visual pipeline:** See **[GENERATION_DIAGRAM.md](GENERATION_DIAGRAM.md)** for a flowchart (Mermaid + ASCII) of text → T5 → diffusion loop (DiT) → VAE → image and optional paths (img2img, control, LoRA).

---

## 1. High-level flow

```
Prompt (+ optional negative, style, control)
    → Text encoder (T5) → embeddings
    → Diffusion loop: start from noise → DiT denoises step-by-step → clean latent
    → VAE decode → pixel image
    → Optional post (sharpen, contrast, resize) → save PNG
```

- **Latent space:** We never diffuse pixels. We diffuse **VAE latents** (e.g. 4×32×32 for 256×256). The DiT sees and predicts in this space; the VAE turns the final latent into an image.
- **Denoising:** The model was trained to predict **noise** (or **velocity**, if `prediction_type="v"`) given noisy latent + timestep + text. At inference we run that many times, step by step, from high noise down to zero.

---

## 2. Load checkpoint and encoders

- **Checkpoint:** Contains `config` (TrainConfig) and `ema` or `model` state. From `config` we know `model_name`, `text_encoder`, `vae_model`, `image_size`, etc.
- **DiT:** Built with `get_dit_build_kwargs(cfg)` and loaded with the checkpoint state. It’s a **transformer** that takes noisy latent patches + timestep + text and outputs a prediction (noise or v).
- **T5:** Tokenizer + `T5EncoderModel` (e.g. T5-XXL). Encodes the prompt (and negative prompt) to fixed-size embeddings `(1, seq_len, 4096)`.
- **VAE:** `AutoencoderKL` (e.g. SD VAE). Used only at the end: **decode** the final latent → RGB image. Latents are scaled by `latent_scale` (e.g. 0.18215) before decode.

---

## 3. Encode text

- **Positive prompt** → tokenize (e.g. max_length=300) → T5 encoder → `cond_emb` shape `(1, L, 4096)`.
- **Negative prompt** (or default like “blurry, worst quality, bad anatomy…”) → same pipeline → `uncond_emb`.
- Optional **style** prompt → encoded and blended into conditioning (if `style_embed_dim` and `--style` are used).
- Optional **creativity** scalar 0–1 is added as extra conditioning if the model was trained with `creativity_embed_dim`. **`--creativity-jitter`** adds per-image noise (useful with **`--num`** > 1 for more spread).
- Optional **`--originality`** (0–1) prepends random **composition / lighting** tokens after subject tags ([`utils/prompt/originality_augment.py`](../utils/prompt/originality_augment.py)); train with **`--train-originality-prob`** for the same distribution at learning time.

These embeddings are passed as `encoder_hidden_states` (and negative as `encoder_hidden_states_negative` for CFG).

- Optional **`token_weights`** (per T5 position): `sample.py` parses `(word)` / `[word]` in the positive prompt, strips brackets for T5, and scales cross-attn conditioning (1.2 / 0.8). Training can mirror this with **`train.py --train-prompt-emphasis`** ([`utils/prompt/prompt_emphasis.py`](../utils/prompt/prompt_emphasis.py), [TRAINING_TEXT_TO_PIXELS.md](TRAINING_TEXT_TO_PIXELS.md)).

**Preview without a GPU:** `python scripts/tools/preview_generation_prompt.py --prompt "..."` runs `utils.prompt.content_controls` + the same pos/neg token conflict filter as `sample.py` (subset of flags; no checkpoint). Set `SDX_DEBUG=1` to print a traceback if `apply_content_controls` fails.

---

## 4. Diffusion object

- **GaussianDiffusion** is created with the same `num_timesteps`, `beta_schedule`, and `prediction_type` as in training (e.g. 1000 steps, linear, epsilon).
- It knows how to:
  - **Add noise:** `q_sample(x0, t)` → `x_t` (forward diffusion).
  - **One denoise step:** `p_step(model, x_t, t, model_kwargs)` (DDPM) or the DDIM-style **step_with_pred** used in the loop.
  - **Predict x0 from model output:** `_predict_x0_and_noise(model_out, x_t, t)` so we can compute the next `x_{t-1}` (or next DDIM state).

---

## 5. Sampling loop (denoising)

- **Initial latent:**
  - **Text-to-image:** `x = randn(1, 4, H_latent, W_latent)` (pure noise).
  - **Img2img / from-z:** Start from a latent (or from an image encoded with the VAE), add noise up to a timestep `t_start`, then denoise from there.
- **Timesteps:** For 50 steps we don’t use all 1000; we **subsample** (e.g. every 20th step) from 999 down to 0. So we get 50 steps: high noise → low noise → clean.
- **Per step:**
  1. **CFG (Classifier-Free Guidance):**  
     - Run the DiT **twice**: once with `encoder_hidden_states = cond_emb` (prompt), once with `uncond_emb` (negative).  
     - Prediction = `uncond + cfg_scale * (cond - uncond)`. Typical `cfg_scale=7.5` pushes the sample toward the prompt and away from the negative.
  2. **DDIM step:** Using the combined prediction, compute predicted `x_0` and then the next latent `x_{t_next}` (deterministic if `eta=0`).
  3. Replace `x` with that next latent and repeat.

- Loop ends when we’ve reached the last timestep. The result is **x_0**: the predicted clean latent (no noise).

---

## 6. What the DiT does each step

For each call `model(x, t, encoder_hidden_states=..., ...)`:

1. **Patchify:** `x` is latent `(B, 4, H, W)`. It’s split into patches (e.g. 2×2) and linearly embedded → sequence of tokens plus 2D sin/cos position encoding.
2. **Timestep:** `t` is embedded (e.g. MLP + sin/cos) → `t_emb`.
3. **Text:** `encoder_hidden_states` (B, L, 4096) is projected to the model’s hidden size (e.g. 1152) via `text_embedder` → `text_emb`. If a negative prompt is used for CFG, that’s handled in `sample_loop` by two separate forward passes; inside the DiT we only see one conditioning at a time.
4. **Blocks:** A stack of transformer blocks. Each block:
   - **Self-attention** over the spatial tokens. If `num_ar_blocks` > 0, a **block-causal mask** is applied so that patches only attend to earlier blocks (and earlier positions within their block) in raster order; see §9 and [AR.md](AR.md).
   - **Cross-attention:** query = spatial tokens, key/value = `text_emb` → the image tokens “look at” the text.
   - **MLP** and residual connections, with **AdaLN** (timestep `t_emb` modulates scale/shift).
5. **Final layer:** Projects the sequence back to patch-sized predictions (same shape as input patches), then **unpatchify** → `(B, 4, H, W)` or `(B, 8, H, W)` if `learn_sigma`. That output is the predicted **noise** (or **v**), which the diffusion code converts to a predicted `x_0` and then to the next `x`.

So at every step the DiT is answering: “Given this noisy latent, this timestep, and this text, what is the noise (or v)?” The diffusion loop uses that to move the latent toward a clean image that matches the prompt.

---

## 7. From latent to image

- **Unscale:** `x_0 = x_0 / latent_scale` (e.g. 0.18215).
- **VAE decode:** `image = vae.decode(x_0).sample` → `(1, 3, H_img, W_img)` in [-1, 1].
- **To uint8:** `(image * 0.5 + 0.5).clamp(0, 1)` → permute to HWC → multiply by 255, round, uint8.
- Optional: resize to `--width`/`--height`, then **sharpen** and **contrast** (e.g. `utils/quality`) and save as PNG.

---

## 8. Optional paths (same pipeline, different start/cond)

- **Img2img:** Encode initial image with VAE → add noise up to `t_start` → run the same loop from that `x_init` and `start_timestep`.
- **Inpainting:** Same as img2img but the initial noisy latent is a blend of (noise in mask region, noisy image in unmasked).
- **ControlNet:** A control image (depth/edge/pose) is encoded and **added** to the DiT’s patch embeddings inside the model; the rest of the loop is unchanged.
- **LoRA:** Applied on top of the loaded weights before sampling; the forward pass is the same.
- **Style / creativity:** Extra conditioning inputs to the DiT; the diffusion loop and VAE decode are unchanged.

---

## Summary

**Generation = iterative denoising in latent space, guided by text.**  
The DiT predicts noise (or v) from (noisy latent, timestep, text); the diffusion loop uses that prediction and the schedule to go from random noise to a clean latent; the VAE decodes that to the final image.

---

## 9. Where AR (autoregressive) fits in

If the model was trained with **`num_ar_blocks` > 0**, then during **every** denoising step the DiT’s **self-attention** over the spatial patches is **block-causal**: the patch grid is split into blocks (e.g. 2×2 or 4×4), and each block can only attend to earlier blocks (and within its own block, in raster order). So “top-left” is generated first, then “top-right” (seeing top-left), then bottom-left, then bottom-right. **Cross-attention to the text stays bidirectional** — all patches can see all text tokens.

**Code:**

- **Mask:** `models/attention.py` → `create_block_causal_mask_2d(h, w, num_ar_blocks)`. Builds an (N, N) mask with `-inf` where attention is disabled.
- **DiT:** In `models/dit_text.py` and `models/dit_predecessor.py`, when `num_ar_blocks > 0` we set `num_patches = (latent_size // patch_size) ** 2`, `p = int(sqrt(num_patches))`, and register `self._ar_mask = create_block_causal_mask_2d(p, p, num_ar_blocks)`. That buffer is passed as `attn_mask` into every block’s self-attention (see the `for i, block in enumerate(self.blocks): ... block(x, c, text_emb, attn_mask=attn_mask, ...)` loop).
- **SelfAttention** in `models/attention.py` takes that mask and uses it in `memory_efficient_attention(..., attn_mask=attn_mask)` or SDPA so that causal positions get `-inf` and are ignored.

So AR does **not** change the diffusion loop or text encoding; it only restricts **which spatial positions can see which** inside the DiT at each step. Full details: [docs/AR.md](AR.md).

---

## 10. Code from the repos (external/)

**SDX does not import from `external/` at runtime.** All runnable code lives in the project (e.g. `models/`, `diffusion/`, `config/`). The repos in `external/` are **reference only** — we reimplement or adapt their ideas.

| Repo (clone into `external/`) | What we take from it (concepts / reference) |
|------------------------------|---------------------------------------------|
| **facebookresearch/DiT** | Patch embed, timestep embed, AdaLN blocks; diffusion respacing. Our [models/dit.py](../models/dit.py), [models/dit_text.py](../models/dit_text.py), [diffusion/gaussian_diffusion.py](../diffusion/gaussian_diffusion.py) are our own implementation. |
| **lllyasviel/ControlNet** | Structural conditioning (depth/edge/pose), control scale. Our [models/controlnet.py](../models/controlnet.py) and control path in the DiT are our own. |
| **black-forest-labs/flux** | Sampling loop, guidance, img2img/fill, structural conditioning docs. Our sampling is in [diffusion/gaussian_diffusion.py](../diffusion/gaussian_diffusion.py) and [sample.py](../sample.py). |
| **Stability-AI/generative-models** | SD3 / official stack as architecture and training reference. |

Clone them with `scripts/setup/clone_repos.ps1` (Windows) or `scripts/setup/clone_repos.sh` (Linux/macOS). For a file-level map of what in each repo corresponds to our features, see [docs/FILES.md](FILES.md) (section “Key files in external repos”). For a short list of ideas we use (PixAI, ComfyUI, etc.), see [docs/INSPIRATION.md](INSPIRATION.md).

---

## 11. Ported code (no runtime dependency on external/)

The following are **adapted into our codebase** so they work with our config, training, and sampling. We do not import from `external/` at runtime.

| Our file | Ported from | What it does |
|----------|--------------|--------------|
| **diffusion/respace.py** | DiT | `space_timesteps(num_timesteps, "ddim50" or "10,15,20")` for flexible inference timesteps. |
| **diffusion/sampling_utils.py** | ControlNet | `norm_thresholding`, `spatial_norm_thresholding` for x0 dynamic thresholding. |
| **diffusion/loss_weighting.py** | generative-models | `get_loss_weight(alpha, "unit" \| "edm" \| "v" \| "eps")` for alternative timestep loss weights. |
| **models/pixart_blocks.py** | PixArt-alpha/sigma | `SizeEmbedder` for (h, w) conditioning; optional building block for multi-res. |

**Config / usage:** Respace: set `timestep_respacing="ddim50"` or `"10,15,20"`. Dynamic threshold: `sample_loop(..., dynamic_threshold_type="norm" or "spatial_norm", dynamic_threshold_value=1.0)` or `--dynamic-threshold-*` in sample.py. Loss weighting: `TrainConfig.loss_weighting="edm"` (or unit/v/eps) and `--loss-weighting` in train.py. SizeEmbedder: import from `models.pixart_blocks` when adding variable-resolution conditioning.
