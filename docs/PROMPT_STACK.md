# Prompt stack — how text becomes conditioning

This doc ties together the **Python modules** and **CLI flags** that shape the positive and negative strings before T5 (or triple-encoder) runs. For **copy-paste recipes**, see [PROMPT_COOKBOOK.md](PROMPT_COOKBOOK.md).

---

## End-to-end path (inference)

1. **User input** — `--prompt`, optional `--negative-prompt`, `--tags` / `--tags-file`, `--lora-trigger`, character sheet / scene blueprint JSON, emphasis `(word)` / `[word]` in `sample.py`.
2. **Defaults** — If negative is empty, `config.reference.prompt_domains` supplies `DEFAULT_NEGATIVE_PROMPT` (or text-in-image variants when `--text-in-image` or detected phrases).
3. **Optional prefixes** — `--hard-style`, `--naturalize` / `--naturalize-deep`, `--anti-bleed`, `--diversity`, originality tokens, etc. (see `sample.py` after the prompt is assembled).
4. **`utils.prompt.content_controls.apply_content_controls`** — SFW/NSFW scaffolding, quality packs (including **`micro_detail`** texture/material), **`adherence_pack`** (standard/strict literalism), pose/view/domain, Civitai trigger banks, **one-shot** scaffolding, **anti-AI** / **human-media** / **LoRA scaffold** packs, composition guards, etc. (`--safety-mode`, `--less-ai`, `--auto-content-fix`, …). With **`--auto-content-fix`** (default on), long prompts can auto-pick **`standard`** / **`strict`** adherence from length + keywords.
5. **Extra negatives** — Scale distortion, character/scene negatives, **multi-LoRA** `LORA_STACK_NEGATIVE` when two+ `--lora` paths.
6. **Conflict filter** — `utils.prompt.neg_filter.filter_negative_by_positive` removes negative tokens that duplicate positive tokens (unless `--no-neg-filter`).
7. **Encode** — T5 (or triple bundle) → `encoder_hidden_states` / CFG uncond.

**Debug:** if `apply_content_controls` fails, `sample.py` prints a warning; set **`SDX_DEBUG=1`** for a full traceback.

**Training:** emphasis in JSONL captions is only applied to the DiT forward if you pass **`--train-prompt-emphasis`** (strips brackets for T5 and passes `token_weights`; see [TRAINING_TEXT_TO_PIXELS.md](TRAINING_TEXT_TO_PIXELS.md)).

---

## Key files

| Piece | Path | Role |
|-------|------|------|
| Content controls | [`utils/prompt/content_controls.py`](../utils/prompt/content_controls.py) | `apply_content_controls`, `infer_content_controls_from_prompt` |
| Pos/neg conflict filter | [`utils/prompt/neg_filter.py`](../utils/prompt/neg_filter.py) | `filter_negative_by_positive`, `positive_token_set` |
| Domain / anti-AI strings | [`config/defaults/prompt_domains.py`](../config/defaults/prompt_domains.py) | Defaults, `ANTI_AI_*`, `NATURAL_LOOK_*`, `LORA_STACK_NEGATIVE`, tips |
| Shims | [`config/prompt_domains.py`](../config/prompt_domains.py) | Re-exports reference module |
| Sampling CLI | [`sample.py`](../sample.py) | Orchestrates the chain above |
| Emphasis → token weights | [`utils/prompt/prompt_emphasis.py`](../utils/prompt/prompt_emphasis.py) | `( )` / `[ ]` parsing + T5 `offset_mapping` weights; **`train.py --train-prompt-emphasis`** for training–inference parity |
| Originality injection | [`utils/prompt/originality_augment.py`](../utils/prompt/originality_augment.py) | **`sample.py --originality`** / **`train.py --train-originality-prob`** — composition tokens after subject tags |
| Refinement smoke | [`inference.py`](../inference.py) | Load ckpt + diffusion; optional `--verify` |

---

## Preview without a GPU

```bash
python scripts/tools/preview_generation_prompt.py --prompt "1girl, red dress" --safety-mode nsfw --less-ai
```

Prints **effective positive / negative** after content controls + the same conflict filter as `sample.py` (subset of flags; no checkpoint). See [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md).

---

## Flag cheat sheet (sampling)

| Intent | Flags (examples) |
|--------|------------------|
| SFW vs NSFW intent | `--safety-mode sfw\|nsfw\|none` |
| Stronger first-try composition | `--one-shot-boost` (default on) / `--no-one-shot-boost` |
| Keyword inference | `--auto-content-fix` (default on) / `--no-auto-content-fix` |
| Less “AI render” look | `--less-ai`, `--anti-ai-pack lite\|strong`, `--human-media photographic\|dslr\|film`, `--naturalize` |
| LoRA fusion | `--lora-scaffold blend\|…`, `--lora-scaffold-auto` |
| Quality ladder | `--quality-pack top\|one_shot\|micro_detail\|…`, `--boost-quality` |
| Prompt literalism | `--adherence-pack standard\|strict` (optional auto via `--auto-content-fix`) |
| Presets | `--preset`, `--op-mode`, `--hard-style` |

Full list: `python sample.py --help`.

---

## Training vs inference

- **Training** uses captions from folders / JSONL (emphasis, regional captions, optional negative fields). Enable **`--boost-adherence-caption`** on `train.py` to prepend adherence tags from [`data/caption_utils.py`](../data/caption_utils.py) (`prepend_adherence_boost`) for stronger literal conditioning. The **same** quality/domain ideas are documented in [MODEL_WEAKNESSES.md](MODEL_WEAKNESSES.md) and [prompt_domains.py](../config/defaults/prompt_domains.py) for caption writing.
- **Inference** adds the dynamic stack above so one checkpoint can be steered without retraining.

---

## See also

- [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) — diffusion loop + T5 + VAE
- [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) — config ↔ checkpoint ↔ sample (§13)
- [QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md) — Civitai tips + community issue matrix
- [scripts/tools/README.md](../scripts/tools/README.md) — `preview_generation_prompt.py`
- [config/defaults/prompt_domains.py](../config/defaults/prompt_domains.py) — default negatives and tips
