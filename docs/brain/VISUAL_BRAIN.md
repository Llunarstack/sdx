# Visual Brain — Understand, Edit, Generate

The **Visual Brain** is SDX's orchestration layer for **understanding** reference images and **iteratively generating/editing** until the output matches the user's prompt.

## Pipeline

```
User prompt
    ↓
[Web search] ──→ download reference images (optional)
    ↓
[Understand] ──→ OCR (pytesseract) + VLM caption (Moondream2) + canny control maps
    ↓
[Dissect] ──→ parse "use X from image N"; GDINO+SAM2 crops/masks when available
    ↓
[Scene brief] ──→ merge user intent + facts (never drop original prompt)
    ↓
[Generate] ──→ sample.py (creative RAG, dissection init, ControlNet, Superior stack)
    ↓
[Verify] ──→ composite + CLIP + OCR (if expected text)
    ↓
[Edit loop] ──→ OCR-fix / inpaint until coverage threshold
```

## Quick start

```bash
# Full brain: web search + understand + generate + edit
python -m scripts.tools visual_brain \
  --ckpt results/best.pt \
  --prompt "neon diner sign reading OPEN at night, vintage cars" \
  --expected-text OPEN \
  --web-search \
  --out diner.png

# With local references (dissection + inpaint init)
python -m scripts.tools visual_brain \
  --ckpt results/best.pt \
  --prompt "use the hat from image 1, portrait of a woman" \
  --reference-images ref_hat.png,ref_face.png \
  --out portrait.png

# Dry-run (build scene brief only, no GPU sampling)
python -m scripts.tools visual_brain \
  --ckpt results/best.pt \
  --prompt "test" \
  --dry-run
```

## Modules

| Module | Role |
|--------|------|
| `utils/brain/image_search.py` | DuckDuckGo + Wikimedia image search |
| `utils/brain/understand.py` | OCR, VLM caption, control map extraction |
| `utils/brain/scene_brief.py` | Synthesized plan of final image contents |
| `utils/brain/visual_brain.py` | Main orchestrator |
| `utils/agentic/tools.py` | Agent tools: `web_search`, `understand_refs`, `dissect_refs`, … |
| `utils/agentic/planner.py` | `plan_visual_brain()` DAG |

## Agent integration

```bash
python -m scripts.tools visual_brain --use-agent \
  --ckpt results/best.pt \
  --prompt "..." \
  --reference-images a.png \
  --web-search
```

Or extend the agentic stack with `ImageGenerationAgent.run_visual_brain(ctx)`.

## Optional models (under `pretrained/`)

- **Moondream2** — VLM captions (`--creative-rag`)
- **GroundingDINO + SAM2** — dissection / segmentation masks
- **Qwen** — LLM reflector in agent loop (`--qwen-path`)
- **pytesseract** — OCR read + verify

Without these, the brain degrades gracefully: heuristics + facts-only dissection + CLIP verify.

## Outputs

Each run writes to `--work-dir`:

- `scene_brief.json` — what the brain decided must appear
- `brain_trace.json` — step log (search, understand, metrics per loop)
- `search/` — downloaded reference images
- `dissection/` — crops and masks
- Final image at `--out`
