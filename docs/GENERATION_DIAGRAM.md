# Text-to-Image Generation Pipeline

How **text** and optional inputs become an **image** in SDX (`sample.py`). All components live in this repo; see [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) for step-by-step detail.

---

## Pipeline overview (Mermaid)

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        P["Prompt"]
        N["Negative prompt"]
        S["Style (optional)"]
        C["Control image (optional)"]
        I["Init image / mask (optional)"]
        L["LoRA weights (optional)"]
    end

    subgraph load["Load from checkpoint & config"]
        CKPT["Checkpoint (.pt)"]
        CFG["Config (model_name, text_encoder, vae_model, image_size, ...)"]
        CKPT --> CFG
        CFG --> DIT["DiT (transformer)"]
        CFG --> T5["T5 tokenizer + encoder"]
        CFG --> VAE["VAE (decoder)"]
    end

    subgraph text["Text encoding"]
        P --> T5
        N --> T5
        S --> T5
        T5 --> COND["Conditioning embeddings (cond + uncond)"]
    end

    L -.->|"apply to weights"| DIT

    subgraph init["Initial latent"]
        NOISE["Random noise (B,4,H,W)"]
        IMG2IMG["Img2img: VAE encode init → add noise to t_start"]
        INPAINT["Inpainting: mask blend noise + image"]
        NOISE --> XINIT["x_init"]
        IMG2IMG --> XINIT
        INPAINT --> XINIT
    end

    I -.-> IMG2IMG
    I -.-> INPAINT
    C --> DIT

    subgraph diffusion["GaussianDiffusion"]
        CREATE["create_diffusion(timestep_respacing, num_timesteps, beta_schedule, prediction_type)"]
    end

    subgraph loop["Sampling loop (N steps)"]
        direction TB
        STEP["For each timestep t (high → low noise):"]
        DIT_COND["DiT(x, t, cond_emb)"]
        DIT_UNCOND["DiT(x, t, uncond_emb)"]
        CFG_COMBINE["CFG: pred = uncond + cfg_scale × (cond − uncond)"]
        RESCALE["Optional: CFG rescale / dynamic threshold"]
        DDIM["DDIM step: predicted x₀ → next x"]
        STEP --> DIT_COND --> DIT_UNCOND --> CFG_COMBINE --> RESCALE --> DDIM
        DDIM --> STEP
        COND --> DIT_COND
        COND --> DIT_UNCOND
        XINIT --> loop
    end

    loop --> X0["Clean latent x₀ (B,4,h,w)"]

    subgraph decode["Decode to image"]
        SCALE["x₀ / latent_scale"]
        VAE_DEC["VAE.decode(x₀)"]
        TO_RGB["(×0.5+0.5), clamp, HWC"]
        SCALE --> VAE_DEC --> TO_RGB
        X0 --> SCALE
        VAE --> VAE_DEC
    end

    subgraph post["Post-process & save"]
        RESIZE["Resize to --width / --height"]
        SHARP["Optional: sharpen, contrast"]
        SAVE["Save PNG (and optional grid)"]
        TO_RGB --> RESIZE --> SHARP --> SAVE
    end

    style inputs fill:#e8f4e8
    style load fill:#e8e8f4
    style text fill:#f4e8e8
    style loop fill:#f4f4e8
    style decode fill:#e8f4f4
    style post fill:#f0e8f4
```

---

## Simplified linear flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PROMPT + NEGATIVE (+ style, control path, init/mask)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  T5 ENCODER  (tokenize → encoder_hidden_states)                              │
│  → cond_emb (positive), uncond_emb (negative); optional style_emb           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  INITIAL LATENT  (noise, or img2img/inpainting from init image + VAE/mask)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DIFFUSION LOOP  (GaussianDiffusion.sample_loop)                             │
│  • Each step: DiT(x, t, cond) and DiT(x, t, uncond)                          │
│  • CFG: pred = uncond + cfg_scale × (cond − uncond)                          │
│  • Optional: CFG rescale, dynamic threshold on x₀                            │
│  • DDIM step → next x; repeat until t = 0                                    │
│  • Output: clean latent x₀                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VAE DECODE  (x₀ / latent_scale → vae.decode → image in [-1,1])              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  POST-PROCESS  (→ [0,1], resize, optional sharpen/contrast) → PNG (+ grid)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components and where they live

| Step | Component | Code / config |
|------|-----------|----------------|
| Load | Checkpoint, config | `sample.py`: `load_model_from_ckpt`; `ckpt["config"]`, `ckpt["ema"]` |
| Load | DiT | `models/` (e.g. `dit_text.py`); `get_dit_build_kwargs(cfg)` |
| Load | T5 | `cfg.text_encoder` → `AutoTokenizer`, `T5EncoderModel` |
| Load | VAE | `cfg.vae_model` → `AutoencoderKL` |
| Optional | LoRA | `models/lora.py`: `apply_loras(model, lora_specs)` |
| Encode | Text | `encode_text([prompt], tokenizer, text_encoder, device)` → (1, L, 4096) |
| Loop | Diffusion | `diffusion/gaussian_diffusion.py`: `create_diffusion`, `sample_loop` |
| Loop | DiT forward | `model(x, t, encoder_hidden_states=..., ...)` → noise or v prediction |
| Decode | VAE | `vae.decode(x0).sample` |
| Post | Quality | `utils/quality.py`: `sharpen`, `contrast` |

---

## Optional paths (same pipeline)

- **Img2img:** Init image → VAE encode → add noise to `t_start = strength × T` → run loop from that `x_init`.
- **Inpainting:** Init image + mask → encode, blend (noise in mask, noisy image outside) → run loop.
- **ControlNet:** Control image loaded and passed as `control_image`; DiT adds it to patch embeddings (see `models/controlnet.py`).
- **Style / creativity:** Extra keys in `model_kwargs_cond`; DiT uses them in the conditioning path.

For full narrative and code pointers, see [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md).
