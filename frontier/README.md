# Frontier — outside-the-box generation

Research playground + ideas to try before promoting to production.

## Modules

| Folder | What | Research source |
|--------|------|-----------------|
| `logic/` | Contradictions, absence (negative space) | — |
| `narrative/` | Witness POV, temporal moment | — |
| `chaos/` | Serendipity, entropy budget | — |
| `memory/` | Generation echo (failure memory) | — |
| **`guidance/`** | Dynamic CFG picker, guidance intervals | arXiv:2509.16131, 2404.13040 |
| **`layout/`** | Omost canvas, `<loc_*>` tokens, LAMIC schedule, IN-R/FI-R metrics | Omost, ConsistCompose, LAMIC |
| **`attention/`** | Cross-attn layout plan (hook for DiT) | BoxDiff, Dense Diffusion |
| **`compose/`** | Per-region reference images | Regional-Prompting-FLUX + PULID |
| `registry.py` | Idea catalog + status | — |

## Quick start

```python
from frontier.registry import list_ideas, idea_by_id

# Browse what to try next
for idea in list_ideas(status="planned"):
    print(idea.id, idea.url)

# Omost canvas → box JSON
from frontier.layout import OmostCanvas, canvas_to_box_layout
import json

c = OmostCanvas()
c.set_global_description("fantasy battlefield at dusk")
c.add_local_description("armored knight", anchor="left", name="knight")
c.add_local_description("burning tower", anchor="right", name="tower")
with open("my_layout.json", "w") as f:
    json.dump(canvas_to_box_layout(c), f, indent=2)

# Then: python sample.py --box-layout my_layout.json
```

### Box layout extras (from online research)

```json
{
  "mask_inject_steps": 10,
  "base_ratio": 0.15,
  "coordinate_tokens": true,
  "lamic_isolation": true,
  "regions": [
    {
      "name": "hero",
      "box": [0.05, 0.1, 0.5, 0.95],
      "prompt": "knight with sword",
      "reference": "refs/face.png",
      "strokes": [{"points": [[0.5, 0.1], [0.5, 0.9]], "width": 0.03}]
    }
  ]
}
```

- **`mask_inject_steps`** — Regional-Prompting-FLUX: regional blend only early steps
- **`base_ratio`** — global vs regional CFG weight (lower = stronger boxes)
- **`coordinate_tokens`** — ConsistCompose `<loc_x1_y1_x2_y2>` in regional prompts
- **`reference`** — per-box identity image (PULID-style; path relative to JSON)

## Layout QA (before sampling)

```python
from frontier.layout import score_layout_masks
from utils.generation.regional_box_prompting import load_box_layout_file, build_latent_region_masks

spec = load_box_layout_file("my_layout.json")
rm, bg = build_latent_region_masks(spec, 64, 64, device=torch.device("cpu"))
print(score_layout_masks(rm, bg))
```

## Planned next (from literature)

See `frontier/registry.py` — **metapoint**, **dense_diffusion**, per-region CADS, VLM refine loop.
