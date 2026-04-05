# Prompt cookbook: copy‑paste recipes using SDX features

Short, opinionated recipes that use the flags and presets you’ve added (`--preset`, `--op-mode`, `--hard-style`, `--naturalize`, `--anti-bleed`, `--diversity`, etc.).

**How the prompt is built (modules and order):** see **[PROMPT_STACK.md](PROMPT_STACK.md)**. **Preview without sampling:** `python scripts/tools/preview_generation_prompt.py --prompt "..."`.

All examples assume:

```bash
cd sdx
```

---

## 1. Photorealistic portrait (SDXL‑style, minimal AI slop)

**Goal:** Natural, realistic portrait with strong adherence and minimal plastic look.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "portrait of a woman, soft window light, shallow depth of field, sharp eyes, 50mm lens" \
  --negative-prompt "" \
  --preset sdxl \
  --op-mode portrait \
  --width 768 --height 1024 \
  --out portrait_sdxl.png
```

What this does:
- `--preset sdxl`: photorealistic defaults (cfg-scale ~6.5, cfg-rescale, steps, hard-style realistic, naturalize on).
- `--op-mode portrait`: turns on diversity, anti-artifacts, naturalize for faces.

---

## 2. Flux‑style gritty photo (safer CFG, no grid)

**Goal:** FLUX‑like realism but with explicit safeguards against CFG burn / grid artifact.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "street photo of a man walking in the rain, city lights, cinematic, moody" \
  --preset flux \
  --op-mode portrait \
  --width 896 --height 1152 \
  --out flux_style.png
```

Preset `flux`:
- Uses CFG ≈ 3.5 and cfg-rescale to avoid FLUX‑style grid artifacts.
- Enables `--naturalize`, `--anti-bleed`, `--diversity`, `--anti-artifacts`, `--strong-watermark`.

---

## 3. 2.5D / semi‑realistic anime character

**Goal:** Anime‑leaning character with semi‑realistic shading and strong tag control.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "1girl, school uniform, medium shot, dynamic lighting, detailed hair" \
  --preset anime \
  --op-mode anime_char \
  --tags "1girl, teen, medium chest, long hair, dynamic lighting" \
  --width 768 --height 1024 \
  --out anime_25d.png
```

Preset `anime` + `anime_char`:
- Hard-style `style_mix` (2.5D / semi‑realistic).
- Anti‑bleed + diversity + anti‑artifacts + strong watermark negative.

---

## 4. Hard 3D/CG shot (no bleed, strong structure)

**Goal:** 3D render that doesn’t mush colors/objects together.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "3d render of a red car parked next to a blue house, sunset, detailed reflections" \
  --preset sdxl \
  --hard-style 3d \
  --anti-bleed \
  --anti-artifacts \
  --width 1024 --height 768 \
  --out hard_3d.png
```

Key pieces:
- `--hard-style 3d`: prepends 3D‑specific tags (octane, 3d render, etc.).
- `--anti-bleed`: adds distinct-colors positive + color‑bleed negative.
- `--anti-artifacts`: adds artifact negative (white dots, spiky, pixel-stretch).

---

## 5. Full‑body character without two‑head

**Goal:** Standing full‑body character in portrait aspect, avoiding two‑head and missing legs.

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "1girl, standing, long dress, legs, shoes, full body, outdoors, sunset" \
  --preset anime \
  --op-mode fullbody \
  --width 768 --height 1152 \
  --out fullbody.png
```

Tips (from `FULL_BODY_AND_TWO_HEAD_TIPS`):
- Describe lower body explicitly: `standing, long dress, legs, shoes`.
- Use portrait aspect for full body; 1:1 for head/shoulder.

---

## 6. Seed exploration (find your god seeds)

Use the seed explorer script to rapidly try many seeds with presets:

```bash
python -m scripts.tools.seed_explorer \
  --ckpt results/.../best.pt \
  --prompt "portrait of a woman, studio lighting" \
  --preset sdxl \
  --op-mode portrait \
  --rows 2 --cols 4 \
  --out-dir seed_explorer/portrait
```

This will:
- Generate 8 seeds (2×4 grid), each saved to `seed_explorer/portrait/seed_*.png`.
- Save a grid image and `seeds.json` listing the seeds.

Pick the best seed(s) and re‑run `sample.py` with your favorite settings.

---

## 7. Checkpoint regression suite (eval prompts)

Run a fixed prompt set so you can compare checkpoints/presets quickly:

```bash
python -m scripts.tools.eval_prompts \
  --ckpt results/.../best.pt \
  --preset sdxl \
  --op-mode portrait \
  --out-dir eval_prompts/sdxl_v1
```

This writes `eval_prompts/sdxl_v1/index.json` plus PNGs for each prompt.

---

## 8. Normalize/OP-ify captions before training

If you have a big JSONL manifest, normalize tag order and apply hard-style/quality boosting in one pass:

```bash
python -m scripts.tools.normalize_captions \
  --in manifest.jsonl \
  --out manifest_normalized.jsonl
```

The script updates:
- `caption` (subject/age/height/build/anatomy ordering + hard-style/quality/domain boosts)
- `negative_caption` / `negative_prompt` (when present)

---

## 9. More novel results (Originality)

If you notice your generations feel too templated / repetitive, use `--originality`.

It:
- injects deterministic “unique composition” tokens near the start of the prompt
- auto-sets `--creativity` (when supported) for extra diversity
- slightly lowers CFG so the model explores beyond the most literal token match

Example:

```bash
python sample.py \
  --ckpt results/.../best.pt \
  --prompt "portrait of a woman, studio lighting" \
  --preset sdxl \
  --op-mode portrait \
  --originality 0.6 \
  --out novel.png
```

