# SDX v12.0.0 — AI Film Studio & Frontier Horizon

**Release date:** June 2026  
**Tag:** [`v12.0.0`](https://github.com/Llunarstack/sdx/releases/tag/v12.0.0)

---

## Highlights

- **Video pipeline** — scene-graph TI2V with studio compiler, 25 frontier filmmaking modules, continuity validators
- **800+ tests** — full CI green (ruff, pytest, smoke imports, doc links)
- **Frontier horizon** — 80+ experimental modules with registry
- **README v12** — GitHub-native layout (no broken Mermaid diagrams)
- **v1 → v12 comparison** — see [VERSION_COMPARISON.md](VERSION_COMPARISON.md)

---

## Quick start

```bash
git clone https://github.com/Llunarstack/sdx.git && cd sdx
pip install -r requirements.txt

# Still images (unchanged)
python sample.py --ckpt outputs/best.pt --prompt "sunset city" --out out.png

# Video (new)
python -m scripts.tools video_generate --scene examples/scene_frontier.example.json --plan-only
```

---

## Breaking changes

None for existing image training/sampling. Video is additive under `pipelines/video/`.

---

## Contributors & AI tools

GitHub's contributor graph reflects **commit author metadata only**. AI tools (Cursor, Claude) are **not permanent contributors** if you:

1. Install `scripts/tools/dev/prepare-commit-msg` as a git hook (strips `Co-authored-by` trailers)
2. Run `scripts/tools/dev/cursorfix.sh` once to rewrite old commits if needed

See README **Contributing** section.

---

## Full notes

[docs/releases/v12.md](v12.md)
