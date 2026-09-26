# SDX v13.0.0 — VIDEOMAX, RSI & Invention Waves

**Release date:** September 2026  
**Tag:** [`v13.0.0`](https://github.com/Llunarstack/sdx/releases/tag/v13.0.0)

## Highlights

- **VIDEOMAX** — one-flag video consistency (identity, permanence, physics, occlusion, shot chain)
- **RSI feedback** — like/dislike/pair → preference JSONL → Diffusion-DPO (`feedback` + `rsi_loop`)
- **Invention Lab** — artwave / maxwave / videowave + ALPHACUT / GAMEASSETS / PHOTOOPS / STYLECAST
- **Quality policy** — compete soft-defaults, CFG-Zero★, pick-best sync, perfect_gen multi-candidate
- **1700+ tests** — VIDEOMAX, RSI, waves, scorecard, scaffolds

## Quick start

```bash
git clone https://github.com/Llunarstack/sdx.git && cd sdx
pip install -r requirements.txt

python sample.py --ckpt outputs/best.pt --prompt "sunset city" --out out.png \
  --invention-stack auto --log-feedback

python -m scripts.tools feedback like out.png --prompt "sunset city"
python -m scripts.tools rsi_loop --base-ckpt outputs/best.pt --dry-run

python -m scripts.tools video_generate --scene examples/scene_frontier.example.json --plan-only
```

## Breaking changes

None required for basic train/sample. VIDEOMAX and RSI are additive.

## Full notes

[v13.md](v13.md) · [RSI](../guides/RSI_FEEDBACK.md) · [VIDEOMAX](../guides/VIDEOMAX.md)
