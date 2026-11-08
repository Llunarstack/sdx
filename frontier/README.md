# Frontier — outside-the-box generation

Prompt-side planners used by `sample.py` (`--frontier`, `--frontier-subject`, `--frontier-perfect`, `--frontier-creative`) and box-layout tooling. `registry.py` lists each idea with its module and status.

## Modules

| Module | What | Research source |
|--------|------|-----------------|
| `engine.py` | Contradictions, absence, witness POV, temporal moment, serendipity, entropy budget, generation echo; `FrontierEngine` and the sampler hooks | — |
| `layout.py` | Omost canvas, `<loc_*>` tokens, LAMIC schedule, IN-R/FI-R metrics, Dense Diffusion cross-attention bias (`--box-attn-layout`), storyboard chains | Omost, ConsistCompose, LAMIC, Dense Diffusion, BoxDiff |
| `planning.py` | Deep synthesis (causality, compute tiers, uncertainty, scene relations, hard-token emphasis), subject planners (anatomy, creatures, mature quality, brushwork, anti-slop realism), perfect mode (atmosphere, materials, motion, era, typography, adult-NSFW steering tiers), video bridge | World-model T2I, test-time scaling, scene graphs |
| `imagination.py` | Creative orchestrator — `--frontier-creative` (surreal, paradox, mutation, constraint, synesthesia, cinema, mood, weathering, glitch, fusion, archetype, rhythm, focal, crowds, scale) | — |
| `multiview.py` | SDX-Spatial 3D core: tri-plane field, SDF fitting, NeuS rendering, mesh export/validation, SH radiance, rigging, asset graph | EG3D, NeuS |
| `registry.py` | Idea catalog + status | — |

The minor block in `utils/safety/minor_guard.py` always runs before the adult-NSFW steering tiers in `planning.py`.

## Quick start

```python
from frontier.registry import list_ideas, idea_by_id

# What is implemented vs partial
for idea in list_ideas(status="partial"):
    print(idea.id, idea.module, idea.summary)

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
      "strokes": [{"points": [[0.5, 0.1], [0.5, 0.9]], "width": 0.03}]
    }
  ]
}
```

- **`mask_inject_steps`** — Regional-Prompting-FLUX: regional blend only early steps
- **`base_ratio`** — global vs regional CFG weight (lower = stronger boxes)
- **`coordinate_tokens`** — ConsistCompose `<loc_x1_y1_x2_y2>` in regional prompts

## Layout QA (before sampling)

```python
from frontier.layout import score_layout_masks
from utils.generation.regional_box_prompting import load_box_layout_file, build_latent_region_masks

spec = load_box_layout_file("my_layout.json")
rm, bg = build_latent_region_masks(spec, 64, 64, device=torch.device("cpu"))
print(score_layout_masks(rm, bg))
```

### Deep frontier (all analyzers)

```python
from frontier.planning import analyze_deep, deep_sample_kwargs

plan = analyze_deep("rain on empty street at night, logo on sign", layout_regions=2)
kw = deep_sample_kwargs(plan, base_negative="blurry")
# kw: prompt, negative_prompt, frontier_guidance_tiers, frontier_recommend_best_of_n, ...
```

### Subject-aware (bodies, creatures, NSFW quality, mediums, realism)

```python
from frontier.planning import analyze_subject, subject_sample_kwargs

plan = analyze_subject(
    "hyperreal boudoir portrait, hands on silk, oil impasto background",
    medium_mode="auto",
)
kw = subject_sample_kwargs(plan, base_negative="blurry")
# body_mode, anatomy_risk, mature_class, realism_tier, merged prompts
```

### Perfect mode (CLI)

```bash
python sample.py --ckpt ... --prompt "..." --frontier-perfect --safety-tier moderate
python sample.py --ckpt ... --prompt "..." --frontier-subject   # subject-only (no full scene stack)
```

### Creative mode (different from perfect — no art-medium duplicate)

```bash
python sample.py --ckpt ... --prompt "surreal Escher jazz club OTS shot VHS rust" --frontier-creative
python sample.py --ckpt ... --prompt "sunset knight" --frontier-creative --creative-mutate 4 --creative-random-constraint
```

```python
from frontier.imagination import analyze_imagination, imagination_sample_kwargs

plan = analyze_imagination("melting dreamscape, techno rave", mutate_count=3, mutate_seed=0)
# plan.serendipity_dial, plan.mutations, plan.step_emphasis
```

### Creative auto-refine (mutations, not just seeds)

```bash
python sample.py --ckpt ... --prompt "sunset knight over fog" \
  --frontier-creative --creative-mutate 4 --auto-refine 4 --out best.png
```

Preview mutations without GPU:

```bash
python -m scripts.tools creative_explore --prompt "steampunk cyberpunk crowd" --mutate 6
python -m scripts.tools creative_explore --prompt "..." --mutate 4 --run -- --ckpt path/to.pt
```
