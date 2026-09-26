# SDX RunPod — five scripts

Clone to `/workspace/sdx`, place credentials in `runpod/secret.txt` (gitignored; copy from `D:\Development\secret.txt` locally). `runpod/setup.sh` installs them to `/workspace/secret.txt` automatically.

| Script | What it does |
|--------|----------------|
| **`setup.sh`** | Install apt packages, Python/CUDA deps, create dirs, link `pretrained/` |
| **`download.sh`** | HF models + scrape all booru sites + enrich/RAG/control prep |
| **`test.sh`** | Integration smoke (scrape + pipeline validation) |
| **`train.sh`** | Train full model / LoRA / ControlNet |
| **`sample.sh`** | Generate images from `best.pt` |
| **`run.sh`** | All of the above in order (`python scripts/run_pipeline.py`) |

## Quick start

```bash
cd /workspace/sdx
bash runpod/setup.sh
bash runpod/test.sh --skip-train          # ~2 min sanity check
bash runpod/download.sh                   # models + data (resumable)
bash runpod/train.sh
bash runpod/sample.sh
```

One shot: `bash runpod/run.sh --skip-train`

## Download options

```bash
bash runpod/download.sh --models-only     # HF weights only
bash runpod/download.sh --data-only       # scrape only
bash runpod/download.sh --skip-preprocess # skip enrich/RAG/control on resume
```

## Train modes

```bash
bash runpod/train.sh                                    # full model
SDX_TRAIN_MODE=lora SDX_INIT_CKPT=... bash runpod/train.sh
SDX_TRAIN_MODE=control bash runpod/train.sh
```

## Paths

| Variable | Default |
|----------|---------|
| `SDX_SECRETS_FILE` | `/workspace/secret.txt` |
| `SDX_DATA` | `/workspace/data` |
| `SDX_PRETRAINED` | `/workspace/pretrained` |
| `SDX_PROMPT_RESEARCH` | `1` — VLM + RAG + Qwen captions (set `0` for fast booru-only) |
| `SDX_MODEL_PROFILE` | `train` — `train` (~65GB), `enrich`, `inference`, or `full` (~199GB) |
| `HF_TOKEN` | HuggingFace token for gated models (`huggingface-cli login`) |

See `env.defaults` for full list. Internal install helpers live in `runpod/lib/` (not user-facing).

Details: [IMAGE_GEN_PIPELINE.md](IMAGE_GEN_PIPELINE.md)

## Windows

```powershell
.\runpod\setup.ps1
.\runpod\test.ps1 --skip-train
python scripts\run_pipeline.py --help
```
