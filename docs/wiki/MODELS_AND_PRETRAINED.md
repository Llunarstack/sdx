# Models and Pretrained Wiring

This page documents how SDX selects local model folders versus Hugging Face IDs.

## Resolver behavior

`utils/modeling/model_paths.py` is the canonical resolver.

- If `pretrained/<ModelName>` exists and is non-empty, SDX uses that local path.
- Otherwise SDX falls back to the configured Hugging Face repo ID.
- This allows portable runs in both offline (local cache) and online (hub fallback) setups.

## Canonical pretrained catalog

The catalog is exposed by `pretrained_catalog()` and includes:

- `T5-XXL`
- `CLIP-ViT-L-14`
- `CLIP-ViT-bigG-14`
- `DINOv2-Large`
- `DINOv2-Giant`
- `SigLIP-SO400M`
- `Qwen2.5-14B-Instruct`
- `StableCascade-Prior`
- `StableCascade-Decoder`
- `GenSearcher-8B`

## Health/report command

Use the tooling command to inspect active resolution and local sizes:

```bash
python -m scripts.tools pretrained_status --out-json pretrained_status.json
```

The report includes:

- resolved source (local path vs HF fallback)
- local/remote state per model
- local folder size
- GenSearcher local shard validation status

## GenSearcher verification

For `GenSearcher-8B`, `verify_gen_searcher_8b_local()` checks required shard/tokenizer/config files for a complete local install.

This avoids silent failures where a partially downloaded local folder exists but misses required model files.

## Suggested practice

- Keep one canonical folder per model under `pretrained/`.
- Prefer `safetensors` where available.
- Run `pretrained_status` after cleanup or migration.
- Pair with `startup_readiness` before long training jobs.
