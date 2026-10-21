# Generating 3D, realistic, interior/exterior and other hard domains

Many models struggle with **3D-rendered** looks, **photorealistic** images, **interior** and **exterior** scenes (architecture, rooms, landscapes), and **style mixes** (2.5D, semi-realistic, photorealistic anime). SDX is set up so these domains are learned well.

## How we handle them

1. **Hard-style boosting** — 3D, photorealistic, and style-mix tags are boosted **first** (stronger than general domains) so the model anchors on the right look. See `data/caption_utils.py`: `HARD_STYLE_TAGS`, `boost_hard_style_tags`. Training applies this before quality/domain boosts.
2. **Domain tag boosting** — When your training captions (or inference prompts) contain tags like `interior design`, `exterior`, the data pipeline moves them to the **front** of the caption (once, deduplicated) via `DOMAIN_TAGS`, `boost_domain_tags`.
3. **Recommended prefixes** — `config/defaults/prompt_domains.py` lists `HARD_STYLE_RECOMMENDED_PROMPTS`. At inference use `--hard-style 3d | realistic | 3d_realistic | style_mix` to prepend the right tags, with a matching `--negative-prompt` (table below).

## Training data

- **3D**: Include images with captions like `3d render, octane render, ...` or `blender 3d, isometric, ...`.
- **Realistic**: Use `photorealistic`, `realistic`, `raw photo`, `natural lighting`, etc.
- **Interior**: Use `interior design`, `indoor`, `room`, `living room`, `modern interior`, etc.
- **Exterior**: Use `exterior`, `outdoor`, `architecture`, `building`, `street view`, etc.

The pipeline automatically boosts these when present, so the model gets a strong signal for these domains.

## Inference

**Hard styles (3D, realistic, style mixes):** Use `--hard-style 3d | realistic | 3d_realistic | style_mix` to prepend recommended tags. Optionally add the matching `--negative-prompt` below.

| Hard style   | Example (prepended by --hard-style)                    | Example negative |
|--------------|--------------------------------------------------------|------------------|
| **3d**       | `3d render, octane render, masterpiece, best quality, 8k` | flat, 2d, blurry, bad proportions |
| **realistic**| `photorealistic, raw photo, natural lighting, masterpiece, 8k uhd` | unrealistic, cartoon, blurry |
| **3d_realistic** | `3d render, photorealistic, octane render, natural lighting, masterpiece` | flat, wrong lighting, plastic look |
| **style_mix** | `2.5d, semi-realistic, masterpiece, best quality, detailed` | blurry, incoherent style, messy |

**Other domains (interior, exterior):** Use the same tags in your prompt and negatives like these:

| Domain    | Example prompt prefix                                      | Example negative |
|-----------|------------------------------------------------------------|------------------|
| **Interior**  | `interior design, indoor, room, modern interior, masterpiece` | empty room, distorted perspective, blurry |
| **Exterior**  | `exterior, outdoor, architecture, building, masterpiece`   | flat, wrong perspective, blurry |

Full prefix lists: `config/defaults/prompt_domains.py` → `HARD_STYLE_RECOMMENDED_PROMPTS`. **Style mixing and LoRAs:** see [QUALITY.md](../QUALITY.md).

## Other hard areas

The same mechanism supports **hands**, **correct anatomy**, **perspective**, **symmetry**, **text/lettering**, **complex composition**, etc. — see `data/caption_utils.py` → `DOMAIN_TAGS["other_hard"]`, `DOMAIN_TAGS["anatomy"]`, `DOMAIN_TAGS["avoid_failures"]`. Use those tags in captions so they get boosted and the model learns them. For **what other models fail at** (hands, faces, double head, text, multiple subjects) and **concrete fixes**, see [QUALITY.md](../QUALITY.md) (Part 2 — model weaknesses).
