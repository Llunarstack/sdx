#!/usr/bin/env bash
# SDX on RunPod — ONE script. Everything else is internal.
#
#   bash /workspace/sdx/runpod/sdx.sh              # help
#   bash /workspace/sdx/runpod/sdx.sh train        # train (default, cheapest useful path)
#   bash /workspace/sdx/runpod/sdx.sh data         # models + scrape + tags
#   bash /workspace/sdx/runpod/sdx.sh full          # everything (days, $$$)
#   bash /workspace/sdx/runpod/sdx.sh sample        # generate an image
#
# Fresh pod (paste once):
#   apt-get update -qq && apt-get install -y -qq git && \
#   git clone --depth 1 -b feat/runpod-readiness-scraper-lora \
#     https://github.com/Llunarstack/sdx.git /workspace/sdx && \
#   bash /workspace/sdx/runpod/sdx.sh train
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib/ensure_repo.sh"
sdx_ensure_repo || exit 1

# shellcheck source=/dev/null
source "$SDX_ROOT/runpod/env.defaults"
cd "$SDX_ROOT"

CMD="${1:-help}"
shift || true

_sdx_help() {
  cat <<'EOF'
SDX RunPod — one script, five commands:

  train    Train on data you already have (cheap default)
  data     Download models + scrape danbooru/rule34 + tag images
  full     data + train + LoRA bank (expensive, days)
  sample   Generate an image from your checkpoint
  setup    Install deps only (first-time pod prep)

Examples:
  bash /workspace/sdx/runpod/sdx.sh train
  SDX_PROMPT="@wlop 1girl" bash /workspace/sdx/runpod/sdx.sh sample

Fresh pod (run once):
  git clone --depth 1 -b feat/runpod-readiness-scraper-lora \
    https://github.com/Llunarstack/sdx.git /workspace/sdx

Ignore every other runpod/*.sh — they call this internally.
EOF
}

case "$CMD" in
  help|-h|--help)
    _sdx_help
    ;;
  setup)
    exec bash "$SDX_ROOT/runpod/setup.sh" "$@"
    ;;
  data)
    export SDX_MODEL_PROFILE="${SDX_MODEL_PROFILE:-ultimate}"
    export SDX_PROMPT_RESEARCH="${SDX_PROMPT_RESEARCH:-0}"
    export SDX_MAX_POSTS="${SDX_MAX_POSTS:-200000}"
    exec bash "$SDX_ROOT/runpod/ultimate.sh" --data-only "$@"
    ;;
  train)
    export SDX_PROMPT_RESEARCH="${SDX_PROMPT_RESEARCH:-0}"
    export SDX_USE_WD_TAGGER="${SDX_USE_WD_TAGGER:-1}"
    export SDX_FULL_TRAIN_FEATURES="${SDX_FULL_TRAIN_FEATURES:-0}"
    export SDX_EPOCHS="${SDX_EPOCHS:-8}"
    export SDX_TRAIN_LORA_BANK=0
    export SDX_NPROC_PER_NODE="${SDX_NPROC_PER_NODE:-3}"
    export SDX_GLOBAL_BATCH_SIZE="${SDX_GLOBAL_BATCH_SIZE:-36}"
    exec bash "$SDX_ROOT/runpod/ultimate.sh" --train-only --skip-lora-bank "$@"
    ;;
  full)
    exec bash "$SDX_ROOT/runpod/ultimate.sh" "$@"
    ;;
  sample)
    exec bash "$SDX_ROOT/runpod/sample.sh" "$@"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    echo >&2
    _sdx_help >&2
    exit 2
    ;;
esac
