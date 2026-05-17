## SDX v8 — invent styles, unify prompts

**Source release v8.0.0** — creativity + prompt parity on top of the v7 operator baseline.

### Headline features

- **Style Genome** — structured invented aesthetics (palette, line, surface, camera, lighting, signature); novelty bank; explore manifests
- **PromptStack v2** — single staged pipeline for inference and training-caption guidance
- **Chaos modes** — `insane`, `apocalypse`, `chimera`, `glitch`, `eldritch`, `cyberpunk`; fusion, hypermutation, prompt clauses
- **Native style ops** — Rust Jaccard/FNV/merge, CUDA pick-best, Go manifest stats/dedupe, Mojo token experiments (Python fallbacks)

### Try it

```bash
pip install -r requirements.txt
python -m scripts.tools explore_styles --prompt "samurai at dusk" --mode chimera
python -m scripts.tools preview_prompt_stack --prompt "portrait" --json
python demo.py
```

### v7 → v8

| v7 | v8 |
|----|-----|
| CI, eval pack, security docs | + Style Genome + PromptStack v2 |
| Reproducibility artifacts | + training/inference prompt parity |
| Research playbooks | + explore_styles CLI + native style layer |

Full notes: [docs/releases/v8.md](https://github.com/Llunarstack/sdx/blob/v8.0.0/docs/releases/v8.md)
