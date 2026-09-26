# VIDEOMAX — fix AI video failure modes

**Idea:** closed video APIs still morph faces, drop props, flip physics, and jump
cuts. SDX already had the repair primitives — they were underwired. VIDEOMAX
turns them all on in one flag, then **routes repairs per medium** so realistic,
anime, cartoon, VFX, stop-motion, and game looks each get the right temporal rules.

## What it fixes

| Failure | Repair |
|--------|--------|
| Identity / face drift | `identity_bind` + `identity_lock` + cross-keyframe identity |
| Objects vanishing | `permanence_repair` + **count_bind repair** (not score-only) |
| Hand / limb morph | `extremity_lock` |
| Logo / text melt | `glyph_lock` (auto on product cues) |
| Floating feet | `contact_ground` (off for anime/cartoon holds) |
| Impossible motion | `physics_gate` + **physics_repair** (off for cel / surreal) |
| Flicker / shimmer | `deflicker` + `hf_deshimmer` + `flow_consistency` |
| Prompt drift | `prompt_ground` + adherence gates + **invention** stack on keyframes |
| Occlusion pops | `occlusion_resolve` (on by default in VIDEOMAX) |
| Shot-to-shot jumps | **shot_chain** (last→first + cut harmonize) |
| Wrong medium feel | **style_router** + motion grammar + animation principles |
| Neural path skip | superiority stack now runs **after** VideoDiT decode |
| Style clobbering wave-5 | invention stacks **union** (`videowave` + style `artwave`/`maxwave`) |
| Occlusion gate ignored | `min_occlusion` passed into segment quality scoring |
| Physics teleport | `physics_repair` center-jump blend (not score-only) |

## Styles (auto from prompt, or `--motion-grammar`)

| Style | Temporal feel | VIDEOMAX dials |
|-------|---------------|----------------|
| `realistic` / `film` | 24fps shutter, grounded physics | permanent, contact, physics, strong deflicker |
| `anime_2d` | on-twos, holds, smear frames | soft permanence, no physics gate, weak deflicker |
| `cartoon` / `spider_verse` | squash/stretch, frame-rate chatter | secondary track, keep intentional pops |
| `pixar_3d` | appealing arcs, soft contact | identity + physics + permanence |
| `vfx` | plate continuity, FX birth/death | highest permanence / physics / occlusion |
| `product` | turntable, locked logo | glyph_lock, ruthless permanence |
| `stop_motion` / `lego` | thumb-jitter cadence | keep jitter, no hf_deshimmer |
| `pixel_art` / `low_poly` / `voxel` | sparse KF, game timing | gameassets inventions |
| `vector` | motion-graphics easing | glyph_lock, clean holds |
| `dream_logic` / `hybrid` | morph-tolerant / mixed media | soft or layered continuity |

## Quick start

```bash
# Full consistency stack (auto-detects medium from prompt)
python -m scripts.tools video_generate \
  --scene examples/scene.example.json \
  --ckpt results/run/best.pt \
  --videomax \
  --out runs/video/out.mp4

# Force anime timing + repairs
python -m scripts.tools video_generate \
  --prompt "sakuga chase, cel shaded anime" \
  --videomax --motion-grammar anime_2d --ckpt …

# Force VFX plate rules
python -m scripts.tools video_generate \
  --prompt "explosion compositing plate" \
  --video-quality max --motion-grammar vfx --ckpt …

# Scene JSON
# "edit": { "videomax": true, "motion_grammar": "cartoon" }
```

## Modes + uploads + timed control

| Mode | What it does |
|------|----------------|
| `t2v` | Text → retrieve/synth → keyframe edit → stitch |
| `i2v` | Still + optional motion clip → animate |
| `i2i` | Still drives keyframe inits (img2img video) |
| `v2v` | Your video plate + prompt restyle (motion preserved) |
| `cgi` | `--motion-grammar cgi` / style cues — feature CGI lookdev dials |

**HF foundations (scaffold):** VACE (`Wan2.1-VACE-*`) for edit/control, `SkyReels-V3-V2V` for dedicated V2V, `Wan2.1-FLF2V` for first–last-frame, `Wan2.2-Animate-*` for character animate, `LivePortrait` for face reenactment. Download with `python scripts/download/download_video_models.py --edit`. See [PRETRAINED_RECOMMENDED.md](../reference/PRETRAINED_RECOMMENDED.md).

**Invention Lab wave 5 (VIDEOWAVE):** 200 inventions (#406–605) covering temporal flicker, identity drift, physics, camera, edit/V2V, **adult NSFW↔SFW** (ADULTGATE), lip-sync, style grammar, and critics. Enable with `--invention-stack videowave` (VIDEOMAX defaults to `videowave,artwave`). Docs: [INVENTION_VIDEO.md](../research/INVENTION_VIDEO.md).

### Attach uploads (paths, not a cloud upload server)

```bash
python -m scripts.tools video_generate \
  --mode v2v --motion-clip plate.mp4 \
  --ref hero.png:identity --ref look.png:style \
  --prompt "cgi neon alley" --motion-grammar cgi --videomax --ckpt …
```

Scene JSON: `references` / `edit.multimodal_refs` with roles
`identity|style|motion|product|scene|voice|…`. See
[`examples/scene_director_media.example.json`](../../examples/scene_director_media.example.json).

### Timed concurrent events

```json
"events": [
  {"at_sec": 2.0, "until_sec": 3.5, "effect": "explosion", "concurrent": true},
  {"at_sec": 2.0, "camera": "whip pan", "concurrent": true},
  {"at_sec": 4.0, "prompt": "rain", "shot_id": "close"}
]
```

Events inject into keyframe prompts at the matching clock so multiple things
can happen at once; `shot_id` scopes to a shot.

## Code map

| Piece | Path |
|-------|------|
| Planner | [`pipelines/video/videomax.py`](../../pipelines/video/videomax.py) |
| Style router | [`pipelines/video/style_router.py`](../../pipelines/video/style_router.py) |
| Motion grammar | [`pipelines/video/motion_grammar.py`](../../pipelines/video/motion_grammar.py) |
| Animation principles | `animation_principles.py` |
| Style engines | `style_engines.py` |
| Physics repair | [`pipelines/video/physics_repair.py`](../../pipelines/video/physics_repair.py) |
| Superior orchestrator | `superior_pass.py` (count + physics repair) |
| Shot chain | `shot_chain.py` wired in `pipeline._run_assignments` |
| Options | `ProcessOptions.videomax` / `motion_grammar` / `*_repair` |

## Honest limits

VIDEOMAX is a **retrieve + keyframe + interpolate + repair** stack (plus optional
neural VideoDiT). It does not invent a pretrained Sora competitor overnight —
it makes the existing SDX path enforce consistency harder than typical APIs, and
routes those fixes so anime does not get live-action physics and VFX does not
get cartoon holds. True end-to-end temporal DiT quality still needs trained
video weights + VAE.
