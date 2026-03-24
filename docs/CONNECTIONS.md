# How files and models connect

Single reference for how config, data, and models flow through the project. See [FILES.md](FILES.md) for the file list and [README.md](../README.md) for usage.

---

## 1. Config flow

```
config/train_config.py (TrainConfig + get_dit_build_kwargs)
         │
         ├── train.py          (args → TrainConfig; saves cfg in checkpoint)
         ├── sample.py         (loads cfg from checkpoint)
         ├── inference.py      (loads cfg from checkpoint)
         └── scripts/training/self_improve.py  (loads cfg from checkpoint)
```

- **Single source of truth:** `config/train_config.py` defines `TrainConfig` and `get_dit_build_kwargs(cfg, *, class_dropout_prob=None)`.
- **Training:** CLI args are converted to `TrainConfig`; the same object is saved in every checkpoint as `ckpt["config"]`.
- **Inference:** `sample.py`, `inference.py`, and `self_improve.py` load the checkpoint, read `cfg = ckpt["config"]`, and build the DiT with `model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))`. They also read `cfg.text_encoder`, `cfg.vae_model`, `cfg.image_size`, etc. for T5, VAE, and diffusion.
- **Why this matters:** Adding a new model option (e.g. `creativity_embed_dim`) only needs to be added in `TrainConfig` and in `get_dit_build_kwargs`; all entry points then stay in sync.

---

## 2. Model build (DiT)

All of these build the **same DiT** from the same checkpoint config:

| Entry point | How it builds DiT |
|-------------|-------------------|
| **train.py** | `model_fn(**get_dit_build_kwargs(cfg))` (uses `cfg.caption_dropout_prob`) |
| **sample.py** | `model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))` |
| **inference.py** | `model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))` |
| **scripts/training/self_improve.py** | `model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))` |

`get_dit_build_kwargs` returns: `input_size`, `text_dim`, `class_dropout_prob`, `num_ar_blocks`, `use_xformers`, `style_embed_dim`, `control_cond_dim`, `creativity_embed_dim`, `size_embed_dim`.  
Model registry: `models/__init__.py` → `DiT_models_text`: `DiT-XL/2-Text` (base), `DiT-P/2-Text`, `DiT-P-L/2-Text` (predecessor), **`DiT-Supreme/2-Text`**, **`DiT-Supreme-L/2-Text`** (supreme: RMSNorm + QK-norm self+cross + SwiGLU + AdaLN-Zero + optional size embed).

---

## 3. Components used where

| Component | train.py | sample.py | inference.py | self_improve.py |
|-----------|-----------|-----------|--------------|-----------------|
| **TrainConfig / cfg** | Yes (from args) | From ckpt | From ckpt | From ckpt |
| **T5 (text encoder)** | Yes (encode captions) | Yes (encode prompt) | Yes | Yes (encode prompt) |
| **VAE** | Yes (encode images → latents) | Yes (decode latents → image) | Yes | Yes (decode) |
| **DiT** | Yes (train) | Yes (sample) | Yes (refine/sample) | Yes (sample) |
| **diffusion** | Yes (training_losses) | Yes (sample_loop) | Yes | Yes (sample_loop) |
| **caption_utils** | Via dataset | — | — | — |
| **ControlNet** | If control_cond_dim | If --control-image | — | — |
| **LoRA** | — | If --lora | — | — |
| **BLIP / VLM** | — | — | — | Optional (captioner) |

T5 and VAE are always loaded from **cfg.text_encoder** and **cfg.vae_model** (Hugging Face IDs or local paths like `model/T5-XXL`).

---

## 4. Data flow

**Training**

```
data_path or manifest_jsonl
    → Text2ImageDataset (data/t2i_dataset.py)
    → collate_t2i → batch { pixel_values, captions, negative_captions, styles, (difficulty), (control_image), (latent_values) }
    → train.py: encode_text(captions) → T5; optional --train-prompt-emphasis → cleaned captions + token_weights in model_kwargs
    → encode_images or latent_values → VAE or cache
    → diffusion.training_losses(model, latents, t, model_kwargs)
    → model(x_t, t, **model_kwargs)  [DiT forward]
```

**Sampling**

```
Checkpoint → cfg + DiT state
cfg.text_encoder, cfg.vae_model → T5, VAE (from HF or model/)
Prompt → encode_text([prompt]) → cond_emb
model_kwargs_cond = { encoder_hidden_states: cond_emb, (style), (control_image), (creativity) }
diffusion.sample_loop(model, shape, model_kwargs_cond, model_kwargs_uncond, ...)
    → x0 (latents)
VAE.decode(x0 / latent_scale) → image
```

**Self-improvement**

```
Checkpoint → cfg + DiT; T5, VAE, diffusion (same as sample)
For each prompt: sample_loop → x0 → decode → save image
Optional: BLIP captioner → overwrite caption in manifest
Write manifest.jsonl + images → use with train.py --manifest-jsonl
```

---

## 5. Where models live

| Model | Config key | Default | Can be local |
|-------|------------|---------|--------------|
| **T5** | `cfg.text_encoder` | `google/t5-v1_1-xxl` | Yes (e.g. `model/T5-XXL`) |
| **VAE** | `cfg.vae_model` | `stabilityai/sd-vae-ft-mse` | Yes (e.g. `model/sdxl-vae-fp16-fix`) |
| **DiT** | Built from cfg; weights in checkpoint | — | Checkpoint only |
| **LLM** | Not in cfg | — | `model/SmolLM2-360M-Instruct`, `model/Qwen2.5-7B-Instruct` (for prompt expansion; not used inside train/sample) |

Download T5 + VAE + LLMs: `python scripts/download/download_models.py --all` → `model/`.

---

## 6. Scripts and what they plug into

| Script | Reads | Writes / uses |
|--------|-------|----------------|
| **train.py** | TrainConfig (args), data_path/manifest_jsonl, optional latent_cache_dir | Checkpoint (model, ema, config, steps), logs |
| **sample.py** | Checkpoint, prompt, optional style/control/creativity | Image, optional attn .pt |
| **inference.py** | Checkpoint | Programmatic API (refine, etc.) |
| **scripts/download/download_models.py** | — | model/T5-XXL, model/sd-vae-ft-mse, model/sdxl-vae*, model/SmolLM*, model/Qwen* |
| **scripts/download/download_llm.py** | — | model/SmolLM2-360M-Instruct or model/Qwen2.5-7B-Instruct |
| **scripts/training/self_improve.py** | Checkpoint, prompts or prompts-file | out_dir/images/*.png, out_dir/manifest.jsonl |
| **scripts/training/precompute_latents.py** | data_path, vae_model | latent_cache_dir/*.pt (per image) |
| **scripts/setup/clone_repos.ps1 / .sh** | — | external/ (reference repos) |
| **scripts/tools/dev/ckpt_info.py** | checkpoint path | — (prints config) |
| **scripts/tools/dev/quick_test.py** | — | — (smoke test) |

---

## 7. Quick checklist: adding a new DiT option

1. Add the field to **TrainConfig** in `config/train_config.py`.
2. Add it to **get_dit_build_kwargs()** in the same file (with a getattr default).
3. Use it in **DiT constructor** in `models/dit_text.py` (and `models/dit_predecessor.py` if applicable).
4. Add CLI arg in **train.py** and pass into TrainConfig (if training-time only, that’s enough).
5. For inference: **sample.py** / **inference.py** already get it from checkpoint config via get_dit_build_kwargs; add any new CLI (e.g. `--creativity`) and pass into model_kwargs.

No need to touch self_improve.py for DiT build (it uses get_dit_build_kwargs); only touch it if you add a new conditioning signal it should pass (e.g. creativity).
