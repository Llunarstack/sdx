#!/usr/bin/env bash
# Train image model — full DiT feature stack (no video/frontier).
#
#   bash runpod/train.sh
#   SDX_TRAIN_MODE=lora SDX_INIT_CKPT=/path/to/base.pt bash runpod/train.sh
#
# Modes: full | lora | control | lora_control
# Set SDX_FULL_TRAIN_FEATURES=0 for baseline DiT only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"
# shellcheck source=/dev/null
source "$ROOT/runpod/lib/train_features.sh"
cd "$ROOT"

MODE="${SDX_TRAIN_MODE:-full}"
ENRICHED="${SDX_ENRICHED_MANIFEST:-$SDX_DATA/enriched/manifest.jsonl}"
MANIFEST="${SDX_MANIFEST:-$SDX_DATA/combined/manifest.jsonl}"
EPOCHS="${SDX_EPOCHS:-20}"
BATCH="${SDX_GLOBAL_BATCH_SIZE:-4}"
IMAGE_SIZE="${SDX_IMAGE_SIZE:-512}"
MAX_STEPS="${SDX_MAX_STEPS:-}"
INIT="${SDX_INIT_CKPT:-}"
NPROC="${SDX_NPROC_PER_NODE:-1}"

EXTRA=()
FEATURE_ARGS=()
sdx_build_train_feature_args FEATURE_ARGS

if [ -n "$MAX_STEPS" ]; then EXTRA+=(--max-steps "$MAX_STEPS"); fi
if [ -n "$INIT" ]; then EXTRA+=(--init-from "$INIT"); fi

case "$MODE" in
  full|lora)
    if [ -z "${SDX_MANIFEST:-}" ] && [ -f "$ENRICHED" ] && [ -s "$ENRICHED" ]; then
      MANIFEST="$ENRICHED"
    fi
    if [ "$MODE" = "lora" ]; then
      EXTRA+=(--lora-train --lora-rank "${SDX_LORA_RANK:-32}" --lora-alpha "${SDX_LORA_ALPHA:-32}")
      if [ -z "$INIT" ]; then
        echo "WARN: SDX_INIT_CKPT not set — LoRA trains from random init." >&2
      fi
    fi
    ;;
  control)
    MANIFEST="${SDX_CONTROL_MANIFEST:-$SDX_DATA/control/manifest.jsonl}"
  ;;
  lora_control)
    MANIFEST="${SDX_CONTROL_MANIFEST:-$SDX_DATA/control/manifest.jsonl}"
    EXTRA+=(
      --lora-train --lora-rank "${SDX_LORA_RANK:-32}" --lora-alpha "${SDX_LORA_ALPHA:-32}"
    )
  ;;
  *)
    echo "Unknown SDX_TRAIN_MODE=$MODE (use full|lora|control|lora_control)" >&2
    exit 2
    ;;
esac

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST — run bash runpod/download.sh first." >&2
  exit 1
fi

echo "Training mode=$MODE manifest=$MANIFEST features=${SDX_FULL_TRAIN_FEATURES:-1} gpus=$NPROC"

python setup/ensure_t5_safetensors.py
python setup/ensure_repa_encoder.py

python setup/sanitize_manifest.py --manifest "$MANIFEST" --data-root "$SDX_DATA" --backup --verify-images

TRAIN_ARGS=(
  train.py
  --manifest-jsonl "$MANIFEST"
  --data-path "$SDX_DATA"
  --results-dir "$SDX_RESULTS"
  --flow-matching-training
  --live-dashboard
  --epochs "$EPOCHS"
  --global-batch-size "$BATCH"
  --image-size "$IMAGE_SIZE"
  "${FEATURE_ARGS[@]}"
  "${EXTRA[@]}"
  "$@"
)

if [ "$NPROC" -gt 1 ]; then
  python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "${TRAIN_ARGS[@]}"
else
  python "${TRAIN_ARGS[@]}"
fi

echo
echo "Sample: bash runpod/sample.sh"
