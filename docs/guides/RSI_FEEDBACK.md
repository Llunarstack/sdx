# RSI: your model teaches itself to make better art

`rsi_loop` improves an SDX checkpoint with no human in the loop. Your thumbs-up/down are optional extra signal.
It only practises art: every prompt it trains on is a medium (oil painting, pixel art, linocut,
portrait photo, …, the 47 mediums in `config/defaults/art_mediums.py`) plus a subject, light and framing.

## One round

1. **Curriculum**: 16 art prompts, weighted toward the medium family (traditional / 2d / photo / generic)
   the champion currently scores worst on.
2. **Paint**: the champion makes 4 candidates per prompt through the normal `sample.py` pipeline.
   `--jobs-jsonl` loads the checkpoint and text encoders once per round.
3. **Judge**: one art score per image. It blends the HF reward panel (PickScore, CLIP-H, CAFE aesthetic,
   whichever are on disk), `art_critic` (composition, exposure, colour harmony) and `ai_look`
   (decoder grid, halos, oversmoothing and other generated-look tells).
4. **Learn**: best-vs-worst pairs per prompt (gap ≥ `--min-margin`) plus your feedback pairs train a
   candidate with `train_diffusion_dpo`.
5. **Gate**: candidate and champion paint the same held-out prompts with the same seeds. Held-out
   subjects never appear in training prompts. The candidate is promoted only if the mean score rises
   by `--min-gain` **and** it wins at least half the head-to-heads. Otherwise it is deleted and the
   learning rate halves.

## Run it

```bash
python -m scripts.tools rsi_loop --base-ckpt results/run/best.pt --rounds 5   # start
python -m scripts.tools rsi_loop --rounds 5                                   # resume, 5 more rounds
python -m scripts.tools rsi_loop --status                                     # history
python -m scripts.tools rsi_loop --base-ckpt results/run/best.pt --dry-run    # preview prompts + commands
```

Output goes to `outputs/rsi/`: `champion.pt` (current best), `champion.prev.pt` (the one before it),
`state.json`, and `round_NNN/` with images, `prefs.jsonl` and `report.json`.

Before the first round:

- **Train first.** RSI sharpens a model that already draws. It cannot rescue a checkpoint that still outputs noise.
- **The checkpoint must sample.** `python sample.py --ckpt <ckpt> --prompt test` has to work with its text
  encoder (T5-XXL etc.) and VAE on disk.
- **Offline by default.** The judge sets `HF_HUB_OFFLINE=1` unless you set it, so a long unattended run
  never starts a surprise multi-GB download.

## Add your own taste

```bash
python sample.py --ckpt ... --prompt "..." --out out.png --log-feedback
python -m scripts.tools feedback like out.png --prompt "..."
python -m scripts.tools feedback pair --win a.png --lose b.png --prompt "..."
python -m scripts.tools feedback sync-taste    # immediate inference bias: sample.py --apply-user-taste
```

Every round mixes the pairs from `outputs/feedback/feedback.jsonl` into training (`--from-feedback ''` turns that off).
Video frames can be rated with `--media-type video`, but RSI trains on images only.
