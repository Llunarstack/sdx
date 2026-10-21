# Frontier — outside-the-box generation

Prompt-side planners used by `sample.py` (`--frontier`, `--frontier-subject`, `--frontier-perfect`, `--frontier-creative`) and box-layout tooling. `registry.py` lists each idea with its module and status.

## Modules

| Module | What | Research source |
|--------|------|-----------------|
| `logic.py` | Contradictions, absence (negative space) | — |
| `narrative.py` | Witness POV, temporal moment | — |
| `chaos.py` | Serendipity, entropy budget | — |
| `memory.py` | Generation echo (failure memory) | — |
| `layout.py` | Omost canvas, `<loc_*>` tokens, LAMIC schedule, IN-R/FI-R metrics | Omost, ConsistCompose, LAMIC |
| `attention.py` | Dense Diffusion cross-attention bias for box layouts (`--box-attn-layout`) | Dense Diffusion, BoxDiff |
| `causality.py` | Rain→wet, fire→smoke plausibility scan | World-model T2I |
| `economy.py` | SKIP/LITE/FULL/HEAVY compute tiers per step | Step-skip / product |
| `uncertainty.py` | Ambiguity score → CFG boost + best-of-N | Test-time scaling |
| `semantics.py` | Subject-relation-object graph → layout hints | Scene graphs |
| `adherence.py` | Hard-token (text, hands, counts) emphasis | Prompt weighting |
| `anatomy.py` | Human vs stylized vs mecha body modes | Anatomy routing |
| `creatures.py` | Dragon, insectoid, eldritch body plans | Creature VFX |
| `mature.py` | NSFW/boudoir *quality* (lighting, skin, form) | Adult art craft |
| `medium.py` | Brush strokes (medium packs live in `config/defaults/art_mediums.py`) | Brush craft |
| `realism.py` | Anti-AI-slop, lens/sensor photoreal stack | Uncanny valley |
| `subject.py` | Compose anatomy + creature + mature + medium + realism | — |
| `atmosphere.py` | Fog, god rays, rain, snow, dust | Environmental |
| `materials.py` | Metal, glass, fabric, skin, wood, liquid, hair | PBR cues |
| `motion.py` | Pan blur, action freeze, impact frames | Action photo |
| `era.py` | Medieval → cyberpunk anachronism guards | Period accuracy |
| `typography.py` | Quoted text legibility + CFG boost | Ideogram-class text |
| `safety.py` | Adult-NSFW steering tiers (`--safety-tier`); the minor block in `utils/safety/minor_guard.py` always runs first | Platform diligence |
| `perfect.py` | **Everything** — `--frontier-perfect` | — |
| `surreal.py` | Dream logic, metamorphosis, scale paradox | vs contradiction resolver |
| `paradox.py` | Escher, infinite loops — keep don't fix | Beautiful impossibility |
| `mutation.py` | Prompt variants for explore / best-of-N | Not static tags |
| `constraint.py` | Art-school limits (monochrome, silhouette…) | Creative restriction |
| `synesthesia.py` | Music → serendipity/CFG knobs | Cross-modal physics |
| `cinema.py` | OTS, POV, ECU shot grammar | Director language |
| `vibe.py` | Mood → step curves + CFG | Mood physics |
| `weathering.py` | Rust, repair, graffiti layers | Object biography |
| `glitch.py` | VHS, datamosh, RGB split | Intentional artifacts |
| `imagination.py` | Creative orchestrator — `--frontier-creative` | No art-medium dup |
| `fusion.py` | Genre mashups with dominant/accent rule | Steampunk+cyber etc. |
| `archetype.py` | Threshold, mirror, labyrinth symbols | Mythic composition |
| `rhythm.py` | Visual beat, repetition, spiral | Pattern composition |
| `focal.py` | DOF as emotional story + CFG hint | Focus intent |
| `collective.py` | Crowds without clone faces | Group grammar |
| `scale.py` | Titan / miniature / cosmic magnitude | Size contrast |
| `registry.py` | Idea catalog + status | — |

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
from frontier.synthesis import analyze_deep, deep_sample_kwargs

plan = analyze_deep("rain on empty street at night, logo on sign", layout_regions=2)
kw = deep_sample_kwargs(plan, base_negative="blurry")
# kw: prompt, negative_prompt, frontier_guidance_tiers, frontier_recommend_best_of_n, ...
```

### Subject-aware (bodies, creatures, NSFW quality, mediums, realism)

```python
from frontier.subject import analyze_subject, subject_sample_kwargs

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
