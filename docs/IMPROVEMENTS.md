# Improvement ideas: quality, fixes, and better training

Ideas to make SDX better—quality gains, modern replacements for old techniques, and features borrowed from how other SD/DiT/FLUX models train and infer. Things we **already have** are noted so you know the baseline.

---

## Already in place (baseline)

- **CFG rescale** (`--cfg-rescale`) and **dynamic threshold** (`--dynamic-threshold`) in sampling — reduce oversaturation at high guidance (ComfyUI/SD3-style). **Configurable CFG** (`--cfg-scale`, default 7.5) so realistic/SDXL-style models can use 5–7. **Batch generation** (`--num N`), **VAE tiling** (`--vae-tiling` or auto when output > 512²) for large decodes, and **safetensors export** (`scripts/tools/export_safetensors.py`) for ComfyUI/A1111. **Civitai-style default negative prompt** (quality + anatomy/hands when negative is empty). **Resolution note** when output size is far from model native (blur warning). See [CIVITAI_QUALITY_TIPS.md](CIVITAI_QUALITY_TIPS.md).
- **Min-SNR weighting** in training — stabilizes learning; avoids overfitting to easy timesteps (SD/SDXL).
- **Noise offset** — better light/dark balance in latents (SD/SDXL).
- **EMA** — inference from EMA weights; smoother, better generalization.
- **Passes-based training** — “N passes over data” instead of raw steps/epochs.
- **Validation + early stopping** — best checkpoint by val loss; stop when quality plateaus.
- **Negative prompt** — model learns to avoid unwanted content.
- **Refinement** (small-t training + optional refinement pass) — fix imperfections.
- **Post-process** — optional sharpen / contrast (`utils/quality.py`).
- **Aesthetic / sample weighting** — `weight` or `aesthetic_score` in JSONL.
- **Precomputed latents** — `--latent-cache-dir` for faster training.

---

## 1. Training: quality and stability

### 1.1 Multi-resolution / aspect-ratio bucketing (SDXL / FLUX / SD3)

- **Idea:** Train on multiple resolutions or aspect ratios (e.g. 256², 512×256, 256×512) instead of a single square. Improves composition and “native” non-square generation.
- **How others do it:** SDXL uses resolution buckets; FLUX/Kontext use fixed preferred resolutions per aspect. We currently have a single `--image-size`.
- **Add:** Optional resolution buckets (e.g. list of (H,W) or aspect ratios) and batch by resolution so each batch is same size; or multi-scale training with random crop to one of several sizes.

### 1.2 Smarter crop strategy — **coded**

- **Idea:** Center crop can cut off important content; random crop can miss the subject. “Largest center crop” that fits the short side, or “face/subject-aware” crop (if metadata exists), often gives better training signal.
- **Coded:** `--crop-mode center|random|largest_center` in `train.py`; `data/t2i_dataset.py` uses `_center_crop`, `_random_crop`, or `_largest_center_crop` (resize to cover then center crop).

### 1.3 Caption dropout schedule (curriculum) — **coded**

- **Idea:** Start with higher caption dropout so the model learns structure first; reduce dropout over time so it learns prompt adherence later. Reduces overfitting to captions early.
- **We have:** Fixed `caption_dropout_prob`. Curriculum for **caption length** exists (`curriculum_caption_steps`).
- **Coded:** `--caption-dropout-schedule 0,0.2,10000,0.05` (comma-sep pairs step,prob); linear interpolation. When set, model uses 0 built-in dropout and we apply scheduled dropout in the loop.

### 1.4 V-prediction as default or option

- **Idea:** Some papers and SD2 use v-prediction (predict velocity); can behave better at high CFG and with certain schedulers. We support `--prediction-type v` but default is epsilon.
- **Add:** Document when to prefer v; optionally A/B test or make v the default for new configs.

### 1.5 Checkpoint averaging (Polyak) — **coded**

- **Idea:** Average the last K checkpoints (or exponential moving average of weights over last N steps) and save as the “final” model. Often improves robustness and reduces variance.
- **Coded:** `--save-polyak N` keeps a running average of model weights (buffer = (1-1/N)*buffer + (1/N)*model each step); saves `polyak.pt` in the checkpoint dir every `ckpt_every`.

### 1.6 Data quality pipeline (dedup, aesthetic, caption filter)

- **Idea:** Deduplicate by embedding or perceptual hash; filter or downweight low-aesthetic / low-caption-quality samples. Used by SD, SDXL, and many fine-tuners.
- **Add:** Script or dataset option: (a) dedup (e.g. by CLIP or image hash), (b) aesthetic score from a small rater (or use existing `weight` in JSONL), (c) caption length/quality filter (drop or weight by length, bad words, etc.). Document in data prep section.

---

## 2. Inference: quality and control

### 2.1 More samplers / schedules

- **Idea:** DDIM is standard; DPM++ 2M, Euler, or flow-matching-style schedules can give better quality or fewer steps (e.g. 20–30 steps with a good schedule).
- **We have:** DDIM-style step in `gaussian_diffusion.py` with CFG rescale and dynamic threshold.
- **Add:** Optional second scheduler (e.g. DPM++ 2M or simple Euler) and a `--scheduler` flag so users can compare.

### 2.2 Self-attention guidance (SAG) or cross-attention control

- **Idea:** Reduce attention to background (SAG) or scale cross-attention per token (prompt emphasis/de-emphasis at inference). Gives more control without retraining.
- **Add:** Optional SAG strength in sampling (attenuate non-subject attention); or cross-attn scale per token from prompt (e.g. (word) → 1.2, [word] → 0.8). Requires hooking into model attention in `sample.py`.

### 2.3 Default CFG rescale / dynamic threshold when CFG is high — **coded**

- **Idea:** When user sets high `--cfg-scale` (e.g. > 10), auto-enable a mild rescale or dynamic threshold so outputs don’t blow out. ComfyUI often does this.
- **Coded:** In `sample.py`, if `--cfg-scale` > 10 and user did not set `--cfg-rescale` or `--dynamic-threshold-percentile`, we auto-set `cfg_rescale=0.7` and `dynamic_threshold_percentile=99.5` and log.

### 2.4 Fewer-step inference (distillation or consistency)

- **Idea:** Train a distilled model (e.g. consistency model or progressive distillation) for 4–8 step inference. Bigger change; often a second training stage.
- **Add:** Long-term: optional consistency/distillation training script that consumes our DiT and produces a few-step model; document as “advanced.”

### 2.5 Style and artist tags (PixAI, Danbooru, Gelbooru) — **coded**

- **Idea:** Make the model really good at styles by using artist/style tags from tag boards; extract style from captions and prompts for the style conditioning head.
- **Coded:** (a) **config/style_artists.py**: `ARTIST_STYLE_PATTERNS` (e.g. "by X", "style of X", `artist:name`), `ARTIST_STYLE_TAGS` (known tags), `extract_style_from_text()`; (b) **data/caption_utils.py**: `DOMAIN_TAGS["style_artist"]` so style/artist tags are boosted; (c) **data/t2i_dataset.py**: when `style` is empty, auto-fill from caption via `extract_style_from_text` (`extract_style_from_caption=True`); (d) **sample.py**: `--auto-style-from-prompt` to extract style from prompt at inference; (e) **docs/STYLE_ARTIST_TAGS.md** and README.

### 2.6 Prompt adherence for complex, NSFW, weird prompts — **coded**

- **Idea:** Improve quality and adherence for long/complex prompts, mature content, and surreal/abstract/weird prompts without censoring.
- **Coded:** (a) **config/prompt_domains.py**: `COMPLEX_PROMPT_TIPS`, `CHALLENGING_PROMPT_TIPS`, and `RECOMMENDED_PROMPTS_BY_DOMAIN` entries for "complex" and "challenging"; (b) **data/caption_utils.py**: domain tags "complex" and "challenging" (surreal, abstract, detailed, etc.) so `boost_domain_tags` reinforces them; `QUALITY_PREFIX` and `prepend_quality_if_short()` for short captions; (c) **sample.py**: `--boost-quality` (prepend "masterpiece, best quality"), long-prompt token count warning (>250 tokens); (d) **docs/CIVITAI_QUALITY_TIPS.md**: sections “Complex and long prompts” and “Challenging content (NSFW, surreal, abstract, weird)” with tips and no-censorship guidance.

---

## 3. Architecture and conditioning

### 3.0 Block-wise AR (expand)

- **We have:** `num_ar_blocks` 0 / 2 / 4 with raster block order; mask in `models/attention.py` → `create_block_causal_mask_2d`; see [docs/AR.md](AR.md).
- **Idea:** **Alternative block orders** — spiral (center-out or corner-in), row-by-row, or “content-aware” order (e.g. from a saliency map). Could improve composition for certain layouts.
- **Idea:** **AR only in early layers** — use block-causal mask in the first K transformer blocks and full bidirectional in the rest; early layers fix structure, later layers refine globally. Requires config (e.g. `ar_layers: Optional[int]`) and per-block mask in the forward.
- **Idea:** **Non-square AR** — support `h != w` for the patch grid (e.g. 32×16) so portrait/landscape latents get a sensible block order; currently mask is built for square `p×p`.
- **Add:** Document 3×3 blocks (`num_ar_blocks=3`) as valid; optionally add `--ar-block-order raster|spiral` and implement spiral in attention.py.

### 3.1 Dual text encoder (T5 + CLIP)

- **Idea:** SDXL uses two text encoders (CLIP + T5). CLIP gives a strong global embedding; T5 gives long context. Combining both can improve both prompt adherence and “style” alignment.
- **Add:** Optional second encoder (CLIP) whose embedding is concatenated or projected into the conditioning path; config flag and extra projection layer. Requires more VRAM and data.

### 3.2 Cached T5 / text encoder caching

- **Idea:** For large batches or many steps, encoding text once and reusing (caching) saves compute. Common in production and in FLUX-style pipelines.
- **We have:** Encode every batch in training.
- **Add:** In inference, cache encoder output per (prompt, negative_prompt) and reuse across steps; optional training path that caches per epoch (more complex).

### 3.3 Inpainting / latent masking in training

- **Idea:** Train with randomly masked latent regions (model predicts only in the mask or the full image with mask conditioning) so the model can do inpainting/outpainting at inference.
- **Add:** Optional `--inpaint-prob` and mask generation (random rectangle or from dataset mask); pass mask to model (e.g. as extra channel or conditioning). Requires model change to accept mask.

### 3.4 RoPE or other positional encodings for latent space

- **Idea:** DiT uses 2D sinusoidal or learned pos for patches. RoPE (rotary) is used in FLUX/LLMs and can improve length/resolution generalization.
- **Add:** Experimental: replace or augment patch pos with RoPE in the DiT block; compare at 256 vs 512.

### 3.5 Latent size conditioning + patch channel gate — **coded (DiT-Text)**

- **Idea (PixArt-style):** Encode latent grid height/width into the timestep conditioning vector so the denoiser knows absolute scale (helps multi-res finetunes and resolution extrapolation).
- **Coded:** `TrainConfig.size_embed_dim` + `--size-embed-dim`; `SizeEmbedder` in `models/pixart_blocks.py` wired in `models/dit_text.py`. Training and `sample.py` pass `size_embed` `(B,2)` = latent `H,W` when `size_embed_dim > 0`. If `size_embed_dim > 0` but `size_embed` is omitted, the model infers `(H,W)` from the input latent tensor.
- **Idea (lightweight quality):** A channel-wise gate on patch tokens (SE-style) with **zero-init last layer** so training starts from identity and can learn calibration.
- **Coded:** `patch_se` / `--patch-se` and `patch_se_reduction` / `--patch-se-reduction` on `TrainConfig`; `ZeroInitPatchChannelGate` after patch/control fusion in `DiT_Text`.

---

## 4. Data and scaling

### 4.1 WebDataset / streaming for huge datasets

- **Idea:** For 10M+ images, loading a single big JSONL or folder can be slow. WebDataset or tar-based streaming loads shards on the fly and scales better.
- **Add:** Optional `DataLoader` backed by WebDataset (or similar) when `--data-format webdataset` and shard path are set; keep current folder/JSONL as default.

### 4.2 Booru tag normalization and quality tags

- **Idea:** For booru-style data: normalize tag order (e.g. subject, character, style, quality), merge synonyms, and enforce quality/rating tags so the model sees consistent conditioning.
- **We have:** PixAI-style emphasis and tag order in captions; quality boost.
- **Add:** Script or dataset option: tag vocabulary, synonym map, and “canonical” ordering for known booru tags; optional rating/quality prefix (e.g. `score_9, safe, ...`).

### 4.3 Video / temporal (future)

- **Idea:** Train on video frames with temporal consistency (e.g. same noise across frames, or temporal attention). Makes the model suitable for video or animated sequences.
- **Add:** Long-term: video dataset loader (frame sampling + optional temporal loss or conditioning); document in HARDWARE.md with the rest of video notes.

---

## 5. Ops and reproducibility

### 5.1 Optional WandB / TensorBoard — **coded**

- **Idea:** Log loss, LR, val loss, and optional samples to WandB or TensorBoard for easier monitoring and comparison.
- **Coded:** `--wandb-project NAME` and/or `--tensorboard-dir DIR`; loss and lr logged every `log_every`. Optional sample images can be added later.

### 5.2 Export to ONNX / TensorRT / Safetensors

- **Idea:** Export DiT (and optionally VAE + text encoder) to ONNX or TensorRT for fast deployment; safetensors for smaller, portable checkpoints.
- **Add:** Script `scripts/export_onnx.py` or `export_safetensors.py` that loads a checkpoint and exports; document in README.

### 5.3 Reproducible runs (full seed control)

- **Idea:** We have `--deterministic` and worker seeds. Extend to dataloader generator, CUDA seeds, and env flags (CUBLAS_WORKSPACE_CONFIG) so the same command gives bit-identical results.
- **Add:** Document and, if needed, set `torch.use_deterministic_algorithms(True)` (where supported) and seed everything in one place; note any non-deterministic ops.

---

## 6. Fixes to “old” techniques

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

## 8. Novel ideas (rare or never in open image gen)

Ideas that are uncommon or absent in mainstream DiT/SD/FLUX pipelines—worth experimenting with to differentiate.

### 8.1 RLHF / preference learning for images

- **Idea:** Train a **reward model** on human (or model) preferences: “A is better than B” for prompt adherence, aesthetics, or safety. Then optimize the diffusion model (e.g. DPO-style, or REINFORCE with the reward) so generations score higher. Common in LLMs; almost never applied end-to-end in open image models.
- **Add:** Optional stage-2 training: freeze or low-lr base, add reward head or use reward to weight/ reject samples and fine-tune toward preferred distribution. Requires preference data or a proxy (e.g. CLIP score, aesthetic scorer).

### 8.2 Constitutional / rule-based auxiliary loss

- **Idea:** Define **rules** (e.g. “no text in image”, “correct hand count”, “no duplicate face”) and add an auxiliary loss or reward that penalizes violations. Can use a small classifier or VLM to score generated samples and backprop or use as a rejection signal. Rarely baked into open image gen training.
- **Add:** Optional `--rule-loss` with a list of rule modules (e.g. “no text”: OCR on decode, penalize if text detected; “anatomy”: hand/face detector, penalize bad count). Blend with denoising loss or use in a second phase.

### 8.3 Mixture of experts (MoE) in the DiT

- **Idea:** Replace some or all transformer blocks with **MoE**: multiple “expert” sub-networks and a router (e.g. from prompt or latent) that selects which experts run. Standard in LLMs; very rare in image DiT. Could specialize experts by “style”, “object”, “background”, etc.
- **Add:** Experimental MoE block: 2–4 experts, router conditioned on text embedding or pooled latent; same FLOPs budget as one block but more capacity. Requires careful load balancing (aux loss or capacity constraint).

### 8.4 Retrieval-augmented generation (RAG) for style or layout

- **Idea:** At inference (or training), **retrieve** a small set of relevant vectors (e.g. from a database of style codes, or “similar prompt” embeddings) and condition the model on them via cross-attention or concatenation. “What would this look like in the style of these N reference embeddings?” Rare in open text-to-image.
- **Add:** Optional “style memory”: user or system provides a few reference image embeddings; model gets a small KV set to attend to in addition to T5. Or retrieve similar captions from training and inject their conditioning.

### 8.5 Spatial “avoid” conditioning (negative regions)

- **Idea:** User specifies **regions** where certain things must not appear (e.g. “no text here”, “no face here”). Negative prompt is global; **spatial negative** conditioning is rare. Could be a latent mask or per-patch “avoid” embedding that the model is trained to respect.
- **Add:** Optional `avoid_mask` or `avoid_regions` in training (and inference): binary or soft mask + optional class (e.g. “text”). Model gets an extra conditioning channel or loss term that discourages those classes in those regions. Requires data with region annotations or user-drawn masks.

### 8.6 Self-improvement loop (synthetic caption bootstrap)

- **Idea:** Generate images with the current model, caption them with a **vision–language model** (e.g. LLaVA, BLIP-2), then train on (generated_image, synthetic_caption) to improve coherence and “understand” its own outputs. Used in some closed systems; rarely in open DiT.
- **Add:** Optional pipeline: sample from current checkpoint, caption with VLM, add to a synthetic dataset; mix with real data (e.g. 10% synthetic) and continue training. Risk: drift if VLM is biased; use with validation and cap on synthetic ratio.

### 8.7 “Creativity” or diversity knob (temperature by region or step)

- **Idea:** A single scalar (or per-step schedule) that controls how much the model **deviates** from the mode—e.g. higher noise in the sampler, or a learned “diversity” embedding that shifts the distribution. Lets users ask for “safe, on-prompt” vs “weirder, more varied” without changing the prompt. Partially overlaps CFG; explicit “creativity” parameter is rare.
- **Add:** Optional conditioning: `creativity` or `diversity` embedding (e.g. scalar → MLP → vector) added to timestep or text path; train with random scalar so model learns to vary style/interpretation. At inference, user sets 0–1 for “strict” vs “creative”.

### 8.8 Learning from rejection (negative signal from user feedback)

- **Idea:** When a user **rejects** an image (thumbs down, or “regenerate”), use it as **negative** signal: contrastive loss (push embedding away from this) or fine-tune to reduce likelihood of this output given the prompt. Requires logging and a feedback loop; almost never in open image gen.
- **Add:** Optional “rejection set”: store (prompt, negative_prompt, latent or image) for rejected samples; in a separate training phase, add a small loss term that discourages the model from producing similar outputs for that prompt. Or use rejection samples to train a small discriminator and use it as in 8.1.

### 8.9 Dual output: image + explanation (attention or keyword heatmap)

- **Idea:** Model outputs not only the image but a lightweight **explanation**: e.g. which prompt tokens most influenced which spatial regions (attention rollout or gradient-based), or a short “why” caption. Interpretability for text-to-image is rare in open models.
- **Add:** At inference, optionally compute and return cross-attention weights (token → patch) or integrated gradients; visualize as heatmap or “key words per region”. No training change; just hooks and a small export format.

### 8.10 Unlearning / concept removal

- **Idea:** Fine-tune the model to **reduce** the probability of generating a specific concept (e.g. a person’s face, a logo, watermarks) without hurting general quality. Machine unlearning for diffusion is emerging; not standard in open pipelines.
- **Add:** Optional script: given a “concept” (e.g. set of images or a text description), run a constrained fine-tune that minimizes likelihood of that concept while regularizing toward the base model (e.g. gradient ascent on concept, plus KL or L2 toward base). Requires careful tuning to avoid catastrophic forgetting.

### 8.11 Adversarial refinement head (discriminator on top of diffusion)

- **Idea:** Keep diffusion as the main generator but add a **discriminator** (e.g. PatchGAN or small CNN) on decoded images and train with a small GAN loss (real vs generated) to sharpen details and fix blur. Used in some super-resolution and old GAN–diffusion hybrids; rare in modern open DiT.
- **Add:** Optional `--adversarial-refine`: freeze or low-lr DiT, add discriminator, train with combined denoising + adversarial loss (e.g. 0.9 * L_denoise + 0.1 * L_adv). Risk: mode collapse; use small weight and short phase.

### 8.12 Difficulty-weighted curriculum — **coded**

- **Idea:** Score each training sample by **difficulty** (e.g. caption length, rarity of tokens, number of objects, or failure rate of a baseline model). Train first on easy samples, then gradually mix in harder ones. Some curriculum exists elsewhere; explicit difficulty scoring and scheduling is uncommon in image gen.
- **Coded:** JSONL may include `"difficulty": 0.0–1.0`. Config `curriculum_difficulty_steps` (e.g. [0, 5000, 10000]), `curriculum_difficulty_easy_first`; train.py multiplies sample_weights by difficulty schedule (early=easy, late=hard). `--curriculum-difficulty-steps 0,5000,10000`.

---

## 7. Suggested priorities

- **Quick wins:** Default CFG rescale when CFG is high (2.3); caption dropout schedule (1.3); checkpoint averaging / Polyak (1.5).
- **Medium effort:** Multi-resolution or aspect bucketing (1.1); data quality script (1.6); WandB/TensorBoard (5.1); export safetensors/ONNX (5.2).
- **Larger projects:** Dual encoder (3.1); inpainting training (3.3); WebDataset (4.1); second scheduler (2.1); SAG / cross-attn control (2.2).
- **Novel / research:** RLHF or rule-based loss (8.1, 8.2); RAG for style (8.4); creativity knob (8.7); self-improvement loop (8.6).
- **Push quality toward “insane”:** Read **§11** — flow/EDM objectives, few-step distillation, best-of-N + quality gates, layout/box conditioning, multi-anchor REPA, RAE adapter + decode refine, VLM data flywheel.

---

## 9. Ideas you might add next

- **Sample grid:** **Done** — `sample.py --num 4 --grid` saves individual files and a single `stem_grid.png` 2×2 (or N-up) grid.
- **Export safetensors:** **Done** — `scripts/tools/export_safetensors.py` saves DiT `state_dict` to `.safetensors` for ComfyUI / other tools.
- **WandB or TensorBoard:** **Done** — `--wandb-project`, `--tensorboard-dir` in train.py (see 5.1).
- **Default negative prompt:** Done — when `--negative-prompt` is empty, sample.py uses `config.prompt_domains.DEFAULT_NEGATIVE_PROMPT`.
- **Checkpoint inspector:** Done — `python scripts/tools/ckpt_info.py path/to/best.pt` prints config and steps; `--keys` lists checkpoint keys.
- **Smoke test:** Done — `python scripts/tools/quick_test.py` runs one forward pass to verify imports and model.
- **Dry-run training:** **Done** — `--dry-run` runs one step and exits (sets max_steps=1).
- **.env.example:** **Done** — `.env.example` documents `HF_TOKEN` and `CUDA_VISIBLE_DEVICES`; copy to `.env` (in .gitignore).
- **Reproducible sampling:** **Done** — `sample.py --deterministic` sets cudnn deterministic + benchmark off and (when supported) deterministic algorithms so same seed → same image.

---

## 10. What the model is still missing (suggested next)

Things commonly expected in production or in ComfyUI/A1111 that we don’t have yet. Pick by priority.

| Gap | What it is | Where (IMPROVEMENTS) | Effort |
|-----|------------|----------------------|--------|
| **Alternative samplers** | **Done** — `--scheduler ddim|euler` in sample.py; Euler uses linear timestep spacing. | §2.1 | — |
| **Prompt emphasis at inference** | **Done** — Use `(word)` and `[word]` in prompt; per-token scale 1.2 / 0.8 (DiT token_weights). | §2.2 | — |
| **Multi-res / aspect bucketing** | Train on 256², 512×256, etc. in one run for better composition | §1.1 | Medium |
| **Data quality pipeline** | **Done** — `scripts/tools/data_quality.py`: dedup (phash/md5), min/max caption length, bad-words, min-weight. | §1.6 | — |
| **T5 caching (server)** | **Done** — `sample.py` caches T5 output per (prompt, negative, style); `--no-cache` to disable. | §3.2 | — |
| **Log sample images (train)** | **Done** — `--log-images-every N` and `--log-images-prompt`; logs to WandB/TensorBoard. | §5.1 | — |
| **Export ONNX / TensorRT** | **Done** — `scripts/tools/export_onnx.py` exports DiT to ONNX; `--dynamic-batch` supported. | §5.2 | — |
| **Full reproducibility** | **Done** — `--deterministic` in sample.py and train.py; [REPRODUCIBILITY.md](REPRODUCIBILITY.md). | §5.3 | — |
| **Dual text encoder (T5+CLIP)** | SDXL-style; second encoder for style/alignment | §3.1 | Large |
| **Inpainting in training** | Train with random masks so model learns inpainting natively (MDM-style patch masking). Inference already supports MDM-style known-region freezing via `sample.py --mask --inpaint-mode mdm`. | §3.3 | Large |
| **WebDataset** | Tar-based streaming for 10M+ images | §4.1 | Medium |

Use this doc as a roadmap: pick items that match your goals (quality vs speed vs scale) and implement in small steps, then validate with the same dataset and metrics (e.g. val loss, visual samples).

---

## 11. Next-tier: “insane” image quality (research → practical SDX hooks)

High-impact ideas from recent text-to-image / DiT / flow work. Many compose with what SDX already has (**REPA, MoE, MDM, AdaGen/PBFM, register tokens, RoPE, KV-merge, token routing, OCR/book pipelines**).

### 11.1 Training objective upgrades (often > raw architecture for “wow”)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **Rectified flow / flow matching** | Straighter noise→data paths; often **fewer steps** and **calmer** optimization than vanilla epsilon prediction on some setups. | Add optional **velocity / flow** training target (you already have `prediction-type` and loss weighting knobs—extend toward RF-style schedules and sampling). |
| **EDM / preconditioning end-to-end** | Cleaner scaling of \( \sigma \), better high-frequency detail when tuned. | You have `loss_weighting` hints; document an **EDM-first** preset and match **sampler** to the same preconditioning. |
| **Multi-scale x0 consistency** | Forces the net to be self-consistent across noise levels; reduces “mushy” midsteps. | Auxiliary loss on decoded previews: **LPIPS / SSIM** vs EMA teacher \(\hat x_0\) (expensive—cap batch or low-res decode). |
| **Beyond REPA: multi-anchor alignment** | One frozen encoder misses texture or typography; multi-encoders catch different failure modes. | **REPA stack**: e.g. **DINOv2 (structure)** + **SigLIP/CLIP (semantic)** with **small fused projector** and schedule (warm-start REPA weight). |

### 11.2 Few-step generation (where “insane” meets “instant”)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **Consistency / distillation (LCM, SDXL-Turbo, Hyper-SD spirit)** | **4–8 step** quality usable in prod. | Stage-2 script: student DiT with **consistency loss** against a frozen teacher + your latents; export a separate `*-fast.pt`. |
| **Guidance distillation** | Reduces **CFG blowout** and can remove **two-forward** cost. | Train a student that absorbs **_CFG + negative_** into weights (pairwise synthetic data or distillation loss). |
| **Step-wise adversarial / discriminator refine** | Sharper micro-detail (texture, eyes, lettering edges) without wrecking diversity if regularized. | Light **PatchGAN** on **short decode crops** + tiny weight; or **stage-only** after main training (see §8.11). |

### 11.3 Inference-time scaling (cheap code, big perceived quality)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **Best-of-N with a cheap judge** | Pick the winner from **K** samples; huge win for **hands / faces / text**. | `sample.py --num K` + **pick**: **CLIPScore**, **aesthetic head**, **OCR match** (you already have OCR repair—use score before repair). |
| **Test-time x0 refinement loop** | One or two **extra denoise** passes at low noise fixes artifacts. | You have **refinement**; add a **quality gate**: only refine if **blur/edge** heuristic or **small CLIP margin** fails. |
| **Dynamic schedule** | Allocate steps where the latent **changes most** (related to AdaGen). | Generalize AdaGen to **per-step budget**: more steps when \(\Delta z\) large; fewer when flat (full “adaptive schedule” vs early-exit only). |

### 11.4 Conditioning & layout (prompt adherence without bigger DiT)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **Box / sketch / layout tokens (GLIGEN-style)** | **Composition** beats raw scale for multi-object scenes. | Extra cross-attn tokens from **bounding boxes** + **class** embeddings in JSONL (`layout` field). |
| **Segmentation or depth as first-class control** | Reduces **wrong object boundaries** and **merged bodies**. | Beyond edges: **sky/ground/character** maps as **control_cond** channels (you have control conditioning—extend channel semantics). |
| **Token-level routing from saliency** | Spend compute on **subject** not **sky**. | **Token routing** exists; feed a **cheap saliency prior** (blur map, face box, text mask) to **bias** `token_router` inputs at inference. |

### 11.5 Data Engine (often beats a 10% wider model)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **VLM re-caption + filter** | Fixes **wrong associations** and improves **rare concepts**. | Batch: **describe → filter by confidence → dedupe**; mix at **5–15%** with real captions. |
| **Synthetic hard negatives** | Teaches **what not to merge** (two faces, six fingers). | Generate **failure cases** on purpose; label with **negative_caption**; small weight in JSONL. |
| **Difficulty-aware mixing** | You have **curriculum difficulty**—pair with **MoE load balancing** so hard samples route to **“detail experts.”** | Joint schedule: **curriculum ↑** + **moe balance loss** tuned together. |

### 11.6 Autoencoder / latent space (ceiling on “insane”)

| Idea | Why it helps | SDX angle |
|------|----------------|-----------|
| **RAE + learned adapter to DiT** | RAE latents are **high-dim**; a **linear or tiny conv adapter** aligns channels to your **4×h×w** DiT without rewriting the world. | Document + implement **explicit adapter** when `autoencoder_type=rae` (currently guarded). |
| **Decoder refinement net** | Fixes **VAE mush** on text and fine lines. | Tiny U-Net on **RGB residual** after VAE (train frozen DiT, only refine decode—fast iteration). |

### 11.7 Suggested “insane mode” stacks (preset combos)

1. **Quality ceiling (slow, best pixels):** REPA (multi-anchor) + MDM + high-step sampler + best-of-4 + optional PBFM + refinement gate.  
2. **Speed ceiling (fast, still strong):** Distilled student + RF-style sampler + AdaGen + KV-merge + token routing + **no CFG** (if guidance-distilled).  
3. **Book / comic ceiling:** Face + bubble anchoring + OCR loop + layout tokens (when added) + edge control.

When you implement anything from §11, add a **one-line “wired in” note** here (like §1.2) so the roadmap stays honest.

### 11.8 Wired in (this repo)

| Idea | Where |
|------|--------|
| **RAE ↔ DiT latent bridge** | `models/rae_latent_bridge.py`; `train.py` creates the bridge when RAE `encoder_hidden_size != 4`, adds optional `--rae-bridge-cycle-weight`, saves `rae_latent_bridge` in checkpoints; `sample.py` / `inference.py` load it and run `dit_to_rae` before decode. |
| **Test-time best-of-N** | `sample.py --num N --pick-best clip\|edge\|ocr\|combo` (+ `--pick-save-all`, `--pick-clip-model`); scores in `utils/test_time_pick.py`. |
