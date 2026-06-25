<p align="center">
  <strong>SDX</strong> · Stable Diffusion Transformer eXtended<br/>
  <sub>Train, layout, direct, and deploy your own image &amp; video models — fully open.</sub>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="docs/releases/v12.md"><img src="https://img.shields.io/badge/release-v12.0.0-0ea5e9?style=flat-square" alt="v12"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-22c55e?style=flat-square" alt="License"/></a>
  <img src="https://img.shields.io/badge/tests-803%2B-22c55e?style=flat-square" alt="Tests"/>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-you-get">Features</a> ·
  <a href="#pipelines">Pipelines</a> ·
  <a href="#new-in-v12">v12</a> ·
  <a href="#v1-vs-v12">v1 → v12</a> ·
  <a href="#docs">Docs</a>
</p>

---

## What is SDX?

**SDX** is an open research framework for building **your own** text-to-image and text/image-to-video systems — not a wrapper around a closed API.

| | Closed APIs | Typical repos | **SDX** |
|---|-------------|---------------|---------|
| Fine-tune your data | ✗ | partial | **✓ end-to-end** |
| See the full pipeline | ✗ | scattered | **✓ readable entry points** |
| Layout + video control | vendor-locked | extensions | **✓ scene JSON + box layout** |
| Reproducibility | ✗ | varies | **✓ 802+ tests + metadata** |

---

## Quick start

```bash
git clone https://github.com/Llunarstack/sdx.git && cd sdx
pip install -r requirements.txt

# Image (demo checkpoint)
python demo.py

# Train on your folder
python train.py --data-path images/ --flow-matching-training --num-epochs 20

# Sample
python sample.py --ckpt outputs/best.pt --prompt "your prompt" --out result.png

# Video — plan from one scene file (v12)
python -m scripts.tools video_generate --scene examples/scene_frontier.example.json --plan-only
```

**Health check:** `python -m toolkit.training.env_health` · **Tests:** `pytest tests/ -q`

---

## What you get

<details open>
<summary><strong>Image generation</strong> — train.py + sample.py</summary>

- DiT + VAE latent diffusion, flow matching, DPO, GRPO (6 variants)
- Holy Grail adaptive CFG, TCIS committee scoring, Style Genome
- Regional box prompting (Ideogram-style layout JSON)
- Agentic quality: ELIQ, artifacts, drift repair, explainability

</details>

<details open>
<summary><strong>Video studio (v12)</strong> — pipelines/video/</summary>

One **scene JSON** → retrieve → keyframe edit → motion → polish → stitch.

- **Studio:** engine router (realistic / anime / voxel / …), director mode, character & world bibles
- **Controls:** elements, motion brush, FLF2V, storyboard cuts
- **Continuity:** eyeline, props, light motivation, thumbnail rehearsal
- **Frontier:** 25 filmmaker modules (tension curve, causal ripples, witness lens, …)

```bash
python -m scripts.tools video_generate --list-frontier
python -m scripts.tools video_generate --scene examples/scene_studio.example.json --preflight
```

</details>

<details>
<summary><strong>Frontier research</strong> — frontier/</summary>

80+ experimental modules: layout, guidance, narrative, realism, cinema, fusion. Browse the registry:

```python
from frontier.registry import list_ideas
implemented = list_ideas(status="implemented")
```

</details>

---

## Pipelines

Diagrams use **tables** (not Mermaid) so they render cleanly on GitHub mobile and dark mode.

### Training (`train.py`)

| Step | What happens |
|------|----------------|
| 1 | Load images + captions |
| 2 | VAE encode → latent space |
| 3 | Add noise @ timestep (flow or VP) |
| 4 | DiT predicts noise / velocity |
| 5 | Loss + backward (optional DPO / GRPO aux) |
| 6 | Checkpoint + metadata |

### Sampling (`sample.py`)

| Step | What happens |
|------|----------------|
| 1 | PromptStack cleans & expands prompt |
| 2 | Optional box layout → regional masks |
| 3 | T5 / CLIP encode conditioning |
| 4 | Denoise loop (Holy Grail CFG, regional blend) |
| 5 | VAE decode → image |

### Video (`pipelines/video/`)

| Step | What happens |
|------|----------------|
| 1 | `compile_scene_graph()` — studio + frontier layers |
| 2 | Shot plan + per-segment overrides |
| 3 | Retrieve reference clips |
| 4 | Keyframe img2img edits |
| 5 | Motion transfer, FLF2V, polish stack |
| 6 | Stitch + provenance |

---

## New in v12

| Area | Highlights |
|------|------------|
| **Video** | Scene-graph TI2V, 60+ modules, CLI tools |
| **Frontier** | 25 filmmaker modules + horizon expansion |
| **Quality** | Continuity validators, thumbnail-first rehearsal |
| **DX** | 802+ tests, ruff-clean CI, docs restructure |
| **README** | GitHub-native layout (this file) |

[Full v12 release notes →](docs/releases/v12.md)

---

## v1 vs v12

| | v1 (foundation) | **v12 (now)** |
|---|-----------------|---------------|
| Scope | Train + sample images | Image + **video studio** + frontier |
| Video | ✗ | Scene JSON director pipeline |
| Layout | ✗ | Regional boxes + storyboard |
| Tests | few | **802+** |
| Research | scripts | `innovations/` + `frontier/` |

[Full comparison →](docs/releases/VERSION_COMPARISON.md)

---

## How SDX compares (ecosystem)

SDX is a **framework you train** — not a hosted model scoreboard.

| Capability | SDXL | Flux | Ideogram | **SDX** |
|---|:---:|:---:|:---:|:---:|
| Full training pipeline | ◐ | ◐ | ✗ | **✓** |
| Flow / DPO / GRPO | ✗ | ◐ | ✗ | **✓** |
| Regional layout | ext | ext | ✓ | **✓** |
| Open scene-graph video | ✗ | ✗ | ✗ | **✓** |
| Self-host everything | ✓ | ◐ | ✗ | **✓** |

---

## Project structure

```
sdx/
├── train.py · sample.py · demo.py     # Image entry points
├── models/ · diffusion/               # DiT, schedulers, sampling
├── innovations/                       # Quality, agentic, control
├── frontier/                          # Experimental research
├── pipelines/video/                   # TI2V scene-graph studio (v12)
├── utils/generation/                  # Layout, CFG, sample features
└── tests/                             # 802+ tests
```

---

## System requirements

| | Minimum | Recommended |
|---|---------|-------------|
| Python | 3.10 | 3.11+ |
| PyTorch | 2.0 | 2.2+ |
| GPU VRAM | 16 GB | 24 GB+ |

---

## Documentation

| Topic | Link |
|-------|------|
| Getting started | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Codebase map | [docs/CODEBASE_GUIDE.md](docs/CODEBASE_GUIDE.md) |
| Video pipeline | [pipelines/video/README.md](pipelines/video/README.md) |
| Frontier | [frontier/README.md](frontier/README.md) |
| v12 release | [docs/releases/v12.md](docs/releases/v12.md) |
| v1 → v12 | [docs/releases/VERSION_COMPARISON.md](docs/releases/VERSION_COMPARISON.md) |

---

## Contributing

```bash
ruff check . --fix && ruff format .
pytest tests/ -q
```

### Are Cursor / Claude permanent contributors?

**No.** GitHub's contributor graph only counts **git commit author names**. AI assistants are not collaborators unless their name appears on commits.

To keep the graph human-only:

1. **Hook (recommended):** copy `scripts/tools/dev/prepare-commit-msg` to `.git/hooks/prepare-commit-msg` — strips `Co-authored-by: Cursor` / Claude trailers before each commit.
2. **Rewrite history (once):** `scripts/tools/dev/cursorfix.sh` reattributes old Cursor-authored commits to you.
3. **Policy:** do not add AI `Co-authored-by` lines (see README in dev scripts).

---

## Version history

| Version | Focus |
|---------|--------|
| **[v12](docs/releases/v12.md)** | Video studio, frontier horizon, 802+ tests | **← current** |
| [v11](docs/releases/v11.md) | Box layout, frontier, restructure |
| [v10](docs/releases/v10.md) | ELIQ, explainability |
| [v9](docs/releases/v9.md) | GRPO, agentic |
| [v1](docs/releases/v0.1.0.md) | Foundation framework |

---

## Citation

```bibtex
@software{sdx_2026,
  title={SDX: Advanced Text-to-Image and Video Generation Framework},
  author={Llunarstack},
  year={2026},
  version={12.0.0},
  url={https://github.com/Llunarstack/sdx}
}
```

---

<p align="center">
  <sub>Apache 2.0 · <a href="https://github.com/Llunarstack/sdx/issues">Issues</a> · <a href="https://github.com/Llunarstack/sdx/releases">Releases</a></sub>
</p>
