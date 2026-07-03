#!/usr/bin/env bash
# H100-max training: latent cache + large batch + compile + feed GPU constantly.
#
#   cd /workspace/sdx && bash runpod/train_h100.sh
#   nohup bash runpod/train_h100.sh > /workspace/train.log 2>&1 &
#
# Aggressive (more VRAM, faster steps — OOM? lower SDX_GLOBAL_BATCH_SIZE):
#   SDX_H100_AGGRESSIVE=1 bash runpod/train_h100.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
cd "$ROOT"

# H100 throughput env
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

MANIFEST="${SDX_MANIFEST:-$SDX_DATA/combined/manifest.jsonl}"
LATENTS="${SDX_LATENT_CACHE:-$SDX_DATA/latent_cache}"
T5="${SDX_TEXT_ENCODER:-$SDX_PRETRAINED/T5-XXL-safetensors}"
IMAGE_SIZE="${SDX_IMAGE_SIZE:-512}"
EPOCHS="${SDX_EPOCHS:-20}"
MAX_STEPS="${SDX_MAX_STEPS:-}"

if [ "${SDX_H100_AGGRESSIVE:-1}" = "1" ]; then
  BATCH="${SDX_GLOBAL_BATCH_SIZE:-32}"
  WORKERS="${SDX_NUM_WORKERS:-48}"
  PREFETCH="${SDX_PREFETCH_FACTOR:-8}"
  PRECOMPUTE_BS="${SDX_PRECOMPUTE_BATCH:-64}"
  GRAD_CKPT=(--no-grad-checkpoint)
else
  BATCH="${SDX_GLOBAL_BATCH_SIZE:-16}"
  WORKERS="${SDX_NUM_WORKERS:-24}"
  PREFETCH="${SDX_PREFETCH_FACTOR:-4}"
  PRECOMPUTE_BS="${SDX_PRECOMPUTE_BATCH:-32}"
  GRAD_CKPT=()
fi

if [ "${SDX_USE_GRAD_CHECKPOINT:-0}" = "1" ]; then
  GRAD_CKPT=()
fi

echo "==> H100 train profile: batch=$BATCH workers=$WORKERS prefetch=$PREFETCH aggressive=${SDX_H100_AGGRESSIVE:-0}"

echo "==> Merge manifests"
python setup/merge_manifests.py --data-root "$SDX_DATA" --out "$MANIFEST"

echo "==> Sanitize manifest (drop missing / non-image rows)"
python setup/sanitize_manifest.py --manifest "$MANIFEST" --data-root "$SDX_DATA" --backup

if [ ! -f "$T5/model.safetensors" ]; then
  echo "Missing T5 safetensors at $T5"
  echo "  hf download mcmonkey/google_t5-v1_1-xxl_encoderonly --local-dir $T5"
  exit 1
fi

ROWS=$(wc -l <"$MANIFEST")
CACHED=$(find "$LATENTS" -name '*.pt' 2>/dev/null | wc -l || echo 0)
NEED=$((ROWS * 90 / 100))
if [ "$CACHED" -lt "$NEED" ]; then
  echo "==> Precompute latents ($CACHED / $ROWS) -> $LATENTS"
  python scripts/training/precompute_latents.py \
    --manifest-jsonl "$MANIFEST" \
    --data-root "$SDX_DATA" \
    --out-dir "$LATENTS" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$PRECOMPUTE_BS" \
    --num-workers "$WORKERS"
else
  echo "==> Latent cache OK ($CACHED files)"
fi

EXTRA=()
[ -n "$MAX_STEPS" ] && EXTRA+=(--max-steps "$MAX_STEPS")

echo "==> Train DiT-XL on $ROWS rows"
python train.py \
  --manifest-jsonl "$MANIFEST" \
  --data-path "$SDX_DATA" \
  --results-dir "$SDX_RESULTS" \
  --text-encoder "$T5" \
  --latent-cache-dir "$LATENTS" \
  --flow-matching-training \
  --live-dashboard \
  --train-style-guidance-mode auto \
  --region-caption-mode append \
  --epochs "$EPOCHS" \
  --global-batch-size "$BATCH" \
  --image-size "$IMAGE_SIZE" \
  --num-workers "$WORKERS" \
  --prefetch-factor "$PREFETCH" \
  --compile-mode max-autotune \
  "${GRAD_CKPT[@]}" \
  "${EXTRA[@]}" \
  "$@"

echo "Done. Sample: bash runpod/sample.sh"
