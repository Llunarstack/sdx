#!/usr/bin/env bash
# Low-cost pipeline presets. Always clones/updates repo via start.sh.
#
#   bash runpod/budget.sh smoke     # setup + verify only
#   bash runpod/budget.sh data      # models + scrape + WD tags (no VLM, no train)
#   bash runpod/budget.sh train     # train on existing data (~$400-800 tier)
#   bash runpod/budget.sh full      # everything (expensive)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TIER="${1:-train}"
shift || true

case "$TIER" in
  smoke)
    export SDX_MODEL_PROFILE=train
    exec bash "$HERE/start.sh" --skip-models --skip-scrape --skip-tag --skip-enrich --skip-train --skip-lora-bank "$@"
    ;;
  data)
    export SDX_MODEL_PROFILE=ultimate
    export SDX_PROMPT_RESEARCH=0
    export SDX_MAX_POSTS="${SDX_MAX_POSTS:-200000}"
    exec bash "$HERE/start.sh" --skip-train --skip-lora-bank "$@"
    ;;
  train)
    export SDX_PROMPT_RESEARCH=0
    export SDX_USE_WD_TAGGER=1
    export SDX_FULL_TRAIN_FEATURES="${SDX_FULL_TRAIN_FEATURES:-0}"
    export SDX_EPOCHS="${SDX_EPOCHS:-8}"
    export SDX_TRAIN_LORA_BANK=0
    export SDX_NPROC_PER_NODE="${SDX_NPROC_PER_NODE:-3}"
    export SDX_GLOBAL_BATCH_SIZE="${SDX_GLOBAL_BATCH_SIZE:-36}"
    exec bash "$HERE/start.sh" --train-only --skip-lora-bank "$@"
    ;;
  full)
    exec bash "$HERE/start.sh" "$@"
    ;;
  -h|--help|help)
    sed -n '2,12p' "$0"
    exit 0
    ;;
  *)
    echo "Unknown tier: $TIER (use smoke|data|train|full)" >&2
    exit 2
    ;;
esac
