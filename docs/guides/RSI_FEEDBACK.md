# RSI — learn from your thumbs

SDX already had automated preference flywheels (benchmark → mine → Diffusion-DPO).
**RSI** closes the loop with **your** likes/dislikes on generated images and video frames.

## Quick path

```bash
# 1) Generate with provenance logging
python sample.py --ckpt results/.../best.pt --prompt "..." --out out.png --log-feedback

# 2) Rate
python -m scripts.tools feedback like out.png --prompt "..."
python -m scripts.tools feedback dislike out_bad.png --prompt "..."
# or pairwise:
python -m scripts.tools feedback pair --win out.png --lose out_bad.png --prompt "..."

# 3) Immediate inference bias (no train)
python -m scripts.tools feedback sync-taste
python sample.py ... --apply-user-taste

# 4) Train a better checkpoint from feedback
python -m scripts.tools rsi_loop --base-ckpt results/.../best.pt --from-feedback outputs/feedback/feedback.jsonl
# or merge with auto-mined pairs:
python -m scripts.tools preference_flywheel --from-feedback outputs/feedback/feedback.jsonl --out data/prefs.jsonl
python -m scripts.tools train_diffusion_dpo --ckpt results/.../best.pt --preference-jsonl data/prefs.jsonl --out rsi_dpo.pt
```

## Video

Enable `log_feedback: true` in video edit/process options (hero frame is logged).
Rate frames the same way with `--media-type video`.

## Commands

| Command | Role |
|---------|------|
| `python -m scripts.tools feedback …` | like / dislike / pair / pick / export-dpo / sync-taste / status |
| `python -m scripts.tools rsi_loop …` | merge feedback → DPO (optional `--run-auto-improve`) |
| `python -m scripts.tools preference_flywheel --from-feedback …` | mix human + benchmark pairs |
| `python -m scripts.tools taste_profile like …` | also appends to the feedback bus |

Default log: `outputs/feedback/feedback.jsonl`.
