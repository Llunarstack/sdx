#!/usr/bin/env bash
# H100-max training: full DiT feature stack + latent cache + multi-GPU.
#
#   cd /workspace/sdx && bash runpod/train_h100.sh
#   nohup bash runpod/train_h100.sh > /workspace/train.log 2>&1 &
#
# 3× H100: SDX_NPROC_PER_NODE=3 SDX_GLOBAL_BATCH_SIZE=36 bash runpod/train_h100.sh
# Baseline DiT only: SDX_FULL_TRAIN_FEATURES=0 bash runpod/train_h100.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
# shellcheck source=/dev/null
source "$ROOT/runpod/lib/train_features.sh"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

MANIFEST="${SDX_MANIFEST:-$SDX_DATA/combined/manifest.jsonl}"
ENRICHED="${SDX_ENRICHED_MANIFEST:-$SDX_DATA/enriched/manifest.jsonl}"
if [ -z "${SDX_MANIFEST:-}" ] && [ -f "$ENRICHED" ] && [ -s "$ENRICHED" ]; then
  MANIFEST="$ENRICHED"
fi
LATENTS="${SDX_LATENT_CACHE:-$SDX_DATA/latent_cache}"
IMAGE_SIZE="${SDX_IMAGE_SIZE:-512}"
EPOCHS="${SDX_EPOCHS:-20}"
MAX_STEPS="${SDX_MAX_STEPS:-}"
NPROC="${SDX_NPROC_PER_NODE:-1}"

if [ "${SDX_H100_AGGRESSIVE:-1}" = "1" ]; then
  BATCH="${SDX_GLOBAL_BATCH_SIZE:-24}"
  WORKERS="${SDX_NUM_WORKERS:-48}"
  PREFETCH="${SDX_PREFETCH_FACTOR:-8}"
  PRECOMPUTE_BS="${SDX_PRECOMPUTE_BATCH:-64}"
  GRAD_CKPT=(--no-grad-checkpoint)
else
  BATCH="${SDX_GLOBAL_BATCH_SIZE:-12}"
  WORKERS="${SDX_NUM_WORKERS:-24}"
  PREFETCH="${SDX_PREFETCH_FACTOR:-4}"
  PRECOMPUTE_BS="${SDX_PRECOMPUTE_BATCH:-32}"
  GRAD_CKPT=()
fi

if [ "${SDX_USE_GRAD_CHECKPOINT:-1}" = "1" ]; then
  GRAD_CKPT=()
fi

FEATURE_ARGS=()
sdx_build_train_feature_args FEATURE_ARGS

echo "==> H100 full train: batch=$BATCH gpus=$NPROC workers=$WORKERS features=${SDX_FULL_TRAIN_FEATURES:-1}"

python setup/ensure_t5_safetensors.py
python setup/ensure_repa_encoder.py

echo "==> Merge + sanitize manifest"
python setup/merge_manifests.py --data-root "$SDX_DATA" --out "$MANIFEST"
python setup/sanitize_manifest.py --manifest "$MANIFEST" --data-root "$SDX_DATA" --backup --verify-images

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
[ -n "${SDX_RESUME:-}" ] && EXTRA+=(--resume "$SDX_RESUME")

TRAIN_ARGS=(
  train.py
  --manifest-jsonl "$MANIFEST"
  --data-path "$SDX_DATA"
  --results-dir "$SDX_RESULTS"
  --latent-cache-dir "$LATENTS"
  --flow-matching-training
  --live-dashboard
  --epochs "$EPOCHS"
  --global-batch-size "$BATCH"
  --image-size "$IMAGE_SIZE"
  --num-workers "$WORKERS"
  --prefetch-factor "$PREFETCH"
  --compile-mode "${SDX_COMPILE_MODE:-max-autotune}"
  "${GRAD_CKPT[@]}"
  "${FEATURE_ARGS[@]}"
  "${EXTRA[@]}"
  "$@"
)

echo "==> Train DiT-XL on $ROWS rows"
if [ "$NPROC" -gt 1 ]; then
  python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "${TRAIN_ARGS[@]}"
else
  python "${TRAIN_ARGS[@]}"
fi

echo "Done. Sample: bash runpod/sample.sh"
