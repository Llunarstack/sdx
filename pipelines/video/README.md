# Video pipeline (retrieve → transform → compose)

**Don't panic.** You only need to think about **3 layers**:

```
SCENE   → one JSON file (prompt, cast, props, effects, duration)
SHOTS   → camera beats inside that file (auto-generated if you skip this)
ENGINE  → retrieval → keyframe edit → interpolate → stitch (automatic)
```

Characters, objects, effects, and transforms are **libraries inside the scene file** —
you reference them by ID in shots. You never wire modules by hand unless you want to.

## The easy way (one file)

Edit `examples/scene.example.json`, then:

```bash
python -m scripts.tools video_generate --scene examples/scene.example.json --dry-run --out out.mp4
python -m scripts.tools video_generate --scene my_scene.json --validate-scene
python -m scripts.tools video_generate --scene my_scene.json --preflight --ckpt results/best.pt
python -m scripts.tools video_generate --scene my_scene.json --plan-only
```

### Image + text combined (i2v control)

Describe **what the image provides** vs **what text changes** per entity:

```json
"inputs": [
  {
    "id": "hero_sheet",
    "image": "hero.png",
    "provides": "face, armor, proportions",
    "text_changes": "walking forward, cape flowing",
    "control": "identity",
    "auto_rig": true
  }
],
"characters": {
  "hero": {
    "bind_input": "hero_sheet",
    "text_by_part": {
      "head": "preserve face from reference",
      "legs": "mid-stride walk"
    }
  }
}
```

**Control modes:** `lock` | `identity` | `transform` | `style` | `motion` | `inpaint` | `generate`

### Elements library (Kling / Runway-style)

```json
"elements": {
  "hero": {
    "images": ["hero.png"],
    "reference_sheet": true,
    "video_ref": "perf_clip.mp4",
    "bind_subject": true,
    "role": "character"
  }
},
"characters": {
  "knight": { "bind_element": "hero", "auto_rig": true }
}
```

### Storyboard cuts (multi-shot in one JSON)

```json
"storyboard": {
  "cuts": [
    { "duration_sec": 3, "prompt": "wide bridge", "camera": "establishing" },
    { "duration_sec": 2, "prompt": "tracking knight", "camera": "tracking", "transition": "dissolve" }
  ]
}
```

Use `shots` OR `storyboard.cuts` (cuts win when shots omitted).

### FLF2V + motion brush (per shot)

```json
{
  "flf2v": true,
  "start_image": "start.png",
  "end_image": "end.png",
  "motion_brush": { "box": [0.2, 0.1, 0.8, 0.9], "mode": "motion_only" }
}
```

See `examples/scene_tier1.example.json` and `examples/scene_i2v_control.example.json`.

### Scene file shape

| Section | What it is | Required? |
|---------|------------|-------------|
| `scene` | Global prompt, duration, fps, style | Yes |
| `characters` | Cast library (`knight`, `hero`, …) | No |
| `objects` | Props library (`sword`, `car`, …) | No |
| `effects` | Look presets (`fog`, `rain`, …) | No (built-in presets exist) |
| `transforms` | Color grade / global look tags | No |
| `shots` | Camera beats referencing the above | No (auto-split from scene prompt) |

## Flow (under the hood)

1. **Plan** — split prompt into shots (`shot_planner.py`, uses `frontier/cinema/shot_grammar`)
2. **Retrieve** — rank local clips, `data/video_catalog.json`, optional Pexels (`PEXELS_API_KEY`)
3. **Extract** — ffmpeg/OpenCV frame extraction
4. **Edit keyframes** — `sample.py` img2img on every Nth frame
5. **Interpolate** — optical flow + blend between keyframes
6. **Temporal pass** — flicker reduction harmonizer
7. **Identity lock** — re-composite anchor in face/body regions (`identity_lock.py`)
8. **Motion transfer** — warp edited keyframe along reference clip flow (I2V)
9. **Pose control** — auto-rig stick figure → ControlNet pose map
10. **Quality retry** — re-run weak segments with softer edit + stronger harmonize
11. **Post grade** — cinematic / teal_orange / muted / vibrant
12. **Depth interpolate** — edge-aware occlusion blend between keyframes
13. **Camera stabilize** — damp shaky reference clips before extract
14. **Deflicker** — rolling luminance harmonize after edits
15. **Motion-beat keyframes** — edit on motion peaks, not just every N frames
16. **Flow consistency** — forward-backward flow repair (less ghosting)
17. **Frame enhance** — unsharp clarity pass
18. **Propagated identity lock** — mask follows subject frame-to-frame
19. **Audio mux** — optional source-clip audio bed on final mp4
20. **Transitions** — `whip`, `flash`, `dip_to_black` between shots
21. **Region motion** — auto-rig limbs/body move independently from reference flow
22. **Semantic drift repair** — detect + blend frames that drift from anchor
23. **Velocity ease** — smoothstep retiming for cinematic acceleration
24. **Parallel segments** — process shots concurrently (`max_segment_workers`)
25. **Preflight** — `--preflight` validates paths before generation
26. **Stitch** — crossfade segments → final mp4 + `provenance.json`

### Edit block (scene JSON)

Tune the engine without touching Python:

```json
"edit": {
  "keyframe_interval": 6,
  "edit_strength": 0.58,
  "motion_transfer": true,
  "identity_lock": true,
  "identity_lock_strength": 0.85,
  "quality_retry": true,
  "max_retries": 2,
  "temporal_alpha": 0.12,
  "pose_control": true,
  "post_grade": "cinematic",
  "depth_interpolate": true,
  "deflicker": true,
  "motion_beat_keyframes": true,
  "flow_consistency": true,
  "propagate_masks": true,
  "region_motion": true,
  "semantic_drift_repair": true,
  "velocity_ease": true,
  "parallel_segments": false,
  "max_segment_workers": 2,
  "frame_enhance": false,
  "audio_from_source": false
}
```

Shot transitions (in scene `shots[].transition`): `cut` | `dissolve` | `match_action` | `whip` | `flash` | `dip_to_black`

## Quick start

```bash
# Plan only
python -m scripts.tools video_generate --prompt "city at dusk then portrait" --plan-only

# Dry run (no checkpoint — synthetic/motion pipeline only)
python -m scripts.tools video_generate --prompt "cinematic cat" --dry-run --duration 2 --out runs/v/cat.mp4

# Full run with keyframe edits
python -m scripts.tools video_generate --prompt "..." --ckpt results/best.pt --local-library data/video_refs --out out.mp4

# Image-to-video
python -m scripts.tools video_generate --mode i2v --prompt "slow zoom portrait" \
  --anchor-image photo.png --motion-clip refs/pan.mp4 --ckpt results/best.pt --out out.mp4
```

## Reference library

Put clips in `data/video_refs/` with optional sidecars:

```
my_clip.mp4
my_clip.mp4.json   # {"title": "...", "tags": ["city","wide"], "license": "user"}
```

Or catalog entries in `data/video_catalog.json`.

## Legal note

Use `--allow-download` / Pexels only with licensed sources. Provenance is written to `runs/.../provenance.json`.
