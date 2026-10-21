# Video quality presets and VIDEOMAX

The video pipeline is retrieve → keyframe `sample.py` edit → interpolate → temporal filters → stitch. Its
quality knobs live in the scene JSON `edit` block. Presets fill those knobs for you. The code is in
[`pipelines/video/segment_processor.py`](../../pipelines/video/segment_processor.py) and
[`pipelines/video/motion_grammar.py`](../../pipelines/video/motion_grammar.py).

## Presets: `edit.video_quality`

| Value | What it sets |
|-------|--------------|
| `none` (default) | Nothing; your `edit` values are used as written. |
| `lite` | Segment retry on. |
| `standard` | Keyframes: 2 candidates + `--pick-best combo_vit`, `--boost-quality`, `--anti-perspective-drift`. Motion-beat keyframes, depth interpolation, flow consistency, deflicker, retry. |
| `strong` | Keyframes: 3 candidates, quality pack `top`, adherence pack, `--human-made standard`, holy-grail, dual-stage layout, `--less-ai`, `--naturalize`. Stronger deflicker and temporal smoothing, identity lock, drift repair, 3 retries, permanence / artifact gates. |
| `max` (or `"videomax": true`) | `strong`, plus camera stabilization, extremity gate, and the motion grammar for the prompt's medium. |

## Motion grammars: `edit.motion_grammar`

Each medium needs different temporal rules. Weak deflicker keeps anime line chatter and stop-motion jitter.
Pixel art must not blend across its grid. Product shots need strict permanence and glyph gates. A grammar
sets keyframe spacing, edit strength, temporal alpha and smoothing, deflicker strength, velocity easing
(eased or holds on twos), the default color grade, and the segment quality gates (permanence, artifact,
identity, physics, glyph).

Grammars: `realistic`, `film`, `product`, `vfx`, `cgi`, `pixar_3d`, `anime_2d`, `cartoon`, `spider_verse`,
`stop_motion`, `pixel_art`, `low_poly`, `vector`, `dream_logic`, `hybrid`.

- `motion_grammar: "auto"` (default): with `max`, the grammar is picked from cues in the scene prompt.
  Otherwise no grammar is applied.
- Any other value applies that grammar, with or without a preset.

```json
"edit": { "video_quality": "max", "motion_grammar": "anime_2d" }
```

## Segment gates and retry

`score_segment_quality` always measures temporal consistency and sharpness. Each extra axis (permanence,
artifact, identity, extremity, physics, glyph) is measured only when its `min_*` floor is above 0. A segment
that fails a gate is re-run, up to `max_retries` times.

## Per-shot overrides

Frontier blocks (tension curve, camera empathy, chromatic arc, parallax budget, stinger frames, …) can
override `edit` values for a single shot, for example `post_grade` or `camera_stabilize`. These per-shot
values win over the scene `edit` block and its preset.

## Modes

| `mode` | Input |
|--------|-------|
| `t2v` | Prompt; clips come from the local library, catalog or Pexels. |
| `i2v` | `anchor_image`, plus an optional motion clip. |
| `v2v` | `motion_clip` is your plate; each shot restyles the next window of it. |

## Limits

Keyframes are SDX image edits, and the motion between them comes from the reference footage. Temporal
quality is therefore bounded by that footage and by interpolation. No video model is trained here.
