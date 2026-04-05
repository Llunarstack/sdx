# Generating 3D, realistic, interior/exterior and other hard domains

Many models struggle with **3D-rendered** looks, **photorealistic** images, **interior** and **exterior** scenes (architecture, rooms, landscapes), and **style mixes** (2.5D, semi-realistic, photorealistic anime). SDX is set up so these domains are learned well.

## How we handle them

1. **Hard-style boosting** — 3D, photorealistic, and style-mix tags are boosted **first** (stronger than general domains) so the model anchors on the right look. See `data/caption_utils.py`: `HARD_STYLE_TAGS`, `boost_hard_style_tags`. Training applies this before quality/domain boosts.
2. **Domain tag boosting** — When your training captions (or inference prompts) contain tags like `interior design`, `exterior`, the data pipeline also **repeats them** via `DOMAIN_TAGS`, `boost_domain_tags`.
3. **Recommended prompts and negatives** — `config/prompt_domains.py` lists `HARD_STYLE_RECOMMENDED_PROMPTS`, `HARD_STYLE_NEGATIVES`, and `RECOMMENDED_PROMPTS_BY_DOMAIN` per domain. At inference use `--hard-style 3d | realistic | 3d_realistic | style_mix` to prepend the right tags (and set `--negative-prompt` from `HARD_STYLE_NEGATIVES` for best results).

## Training data

- **3D**: Include images with captions like `3d render, octane render, ...` or `blender 3d, isometric, ...`.
- **Realistic**: Use `photorealistic`, `realistic`, `raw photo`, `natural lighting`, etc.
- **Interior**: Use `interior design`, `indoor`, `room`, `living room`, `modern interior`, etc.
- **Exterior**: Use `exterior`, `outdoor`, `architecture`, `building`, `street view`, etc.

The pipeline automatically boosts these when present, so the model gets a strong signal for these domains.

## Inference

**Hard styles (3D, realistic, style mixes):** Use `--hard-style 3d | realistic | 3d_realistic | style_mix` to prepend recommended tags. Optionally set `--negative-prompt` from `config/prompt_domains.py` → `HARD_STYLE_NEGATIVES`.

| Hard style   | Example (prepended by --hard-style)                    | Example negative |
|--------------|--------------------------------------------------------|------------------|
| **3d**       | `3d render, octane render, masterpiece, best quality, 8k` | flat, 2d, blurry, bad proportions |
| **realistic**| `photorealistic, raw photo, natural lighting, masterpiece, 8k uhd` | unrealistic, cartoon, blurry |
| **3d_realistic** | `3d render, photorealistic, octane render, natural lighting, masterpiece` | flat, wrong lighting, plastic look |
| **style_mix** | `2.5d, semi-realistic, masterpiece, best quality, detailed` | blurry, incoherent style, messy |

**Other domains (interior, exterior):** Use the same tags in your prompt and the suggested negatives from `RECOMMENDED_NEGATIVE_BY_DOMAIN`:

| Domain    | Example prompt prefix                                      | Example negative |
|-----------|------------------------------------------------------------|------------------|
| **Interior**  | `interior design, indoor, room, modern interior, masterpiece` | empty room, distorted perspective, blurry |
| **Exterior**  | `exterior, outdoor, architecture, building, masterpiece`   | flat, wrong perspective, blurry |

Full lists: `config/prompt_domains.py` → `HARD_STYLE_RECOMMENDED_PROMPTS`, `HARD_STYLE_NEGATIVES`, `RECOMMENDED_PROMPTS_BY_DOMAIN`. **Style mixing and LoRAs:** see `STYLE_MIX_TIPS` and `LORA_MIX_TIPS` in the same file.

## Other hard areas

The same mechanism supports **hands**, **correct anatomy**, **perspective**, **symmetry**, **text/lettering**, **complex composition**, etc. — see `data/caption_utils.py` → `DOMAIN_TAGS["other_hard"]`, `DOMAIN_TAGS["anatomy"]`, `DOMAIN_TAGS["avoid_failures"]`. Use those tags in captions so they get boosted and the model learns them. For **what other models fail at** (hands, faces, double head, text, multiple subjects) and **concrete fixes**, see [MODEL_WEAKNESSES.md](MODEL_WEAKNESSES.md).
