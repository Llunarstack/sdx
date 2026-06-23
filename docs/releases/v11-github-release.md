# SDX v11.0.0 — Layout Control & Research Expansion

**Release date:** June 2026  
**Tag:** [`v11.0.0`](https://github.com/Llunarstack/sdx/releases/tag/v11.0.0)

---

## Highlights

- **Regional box prompting** — Draw boxes, describe regions, add sketches and reference images (Ideogram-style layout control)
- **`frontier/` research layer** — Omost canvas, LAMIC schedules, dynamic CFG, layout metrics, idea registry
- **`innovations/` restructure** — Clean package split replacing `advanced_innovations/`
- **`diffusion/sampling/`** — Consolidated sampling extras with backward-compatible shims
- **648 tests passing** — Full suite green

---

## Install

```bash
git clone https://github.com/Llunarstack/sdx.git
cd sdx
git checkout v11.0.0
pip install -r requirements.txt
python -m pytest tests/ -q
```

---

## Try Box Layout Generation

```bash
python sample.py --ckpt results/best.pt \
  --box-layout examples/box_layout_sketch.example.json \
  --prompt "fantasy scene" \
  --out out.png
```

Build a layout programmatically:

```python
from frontier.layout import OmostCanvas, canvas_to_box_layout
import json

c = OmostCanvas()
c.set_global_description("coastal village at golden hour")
c.add_local_description("lighthouse", anchor="right", name="tower")
c.add_local_description("fishing boat", anchor="left", name="boat")
with open("layout.json", "w") as f:
    json.dump(canvas_to_box_layout(c), f, indent=2)
```

---

## Package Map

```
sdx/
├── innovations/     # Production research modules + agentic quality
├── frontier/        # Experimental layout, guidance, narrative ideas
├── diffusion/
│   └── sampling/    # Holy Grail presets, guidance fusion, runtime guards
└── utils/generation/
    └── regional_box_prompting.py
```

---

## Breaking Changes

| Before (v10) | After (v11) |
|---|---|
| `advanced_innovations/` | `innovations/` |
| `diffusion.sampling_extras` | `diffusion.sampling` (shim kept) |

---

## Documentation

- [Full release notes](v11.md)
- [Frontier README](../../frontier/README.md)
- [Innovations README](../../innovations/README.md)

---

## Tests

```
648 passed
python -m ruff check .  # clean
```
