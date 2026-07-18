# Editing phase (post-generation refine loop)

Implements the flowchart:

**prompts + text encoders + model/VAE/size → sampler → image → break into pieces → editing phase → (prompt gaps / RAG / text fix / art rules) → loop until cohesive & non-AI-looking.**

## Quick start

```bash
# Plan-only (masks, diagnosis, art post) — no checkpoint required
python scripts/tools/editing_phase.py \
  --image outputs/sample.png \
  --prompt 'a samurai at sunset, sign says "OPEN"' \
  --dry-run --out-dir outputs/edit_demo

# Full loop (img2img / inpaint via sample.py)
python scripts/tools/editing_phase.py \
  --image outputs/sample.png \
  --prompt "..." \
  --ckpt results/best.pt \
  --scheduler ays_dit --solver dpmpp_2m \
  --out-dir outputs/edit_demo
```

## What it does each iteration

1. **Diagnose** — sharpness / exposure / CLIP gates; expected quoted text; anatomy cues; missing prompt tokens (vs optional caption)
2. **Break into pieces** — REVE-style region masks (`face`, `hands`, `clothing`, `subject`, `background`) under `pieces/`
3. **Plan** — OCR fix, piece inpaint, prompt realign, RAG prompt delta, artistic post
4. **Apply** — `sample_edit_runner` (DPM++ / AYS defaults) or dry-run
5. **Stop** when gates pass or `max_iters`

## Code

| Piece | Path |
|-------|------|
| Orchestrator | [`utils/generation/editing_phase.py`](../../utils/generation/editing_phase.py) |
| CLI | [`scripts/tools/editing_phase.py`](../../scripts/tools/editing_phase.py) |
| Roles | [`utils/generation/orchestration.py`](../../utils/generation/orchestration.py) (`editor`) |
| Edit backend | [`utils/generation/sample_edit_runner.py`](../../utils/generation/sample_edit_runner.py) |

Pairs with modern sampling defaults in [SAMPLING.md](SAMPLING.md).
