#!/usr/bin/env bash
# SDX on RunPod — one script does everything:
#   models → scrape danbooru+rule34 → WD tagger → VLM captions → train base → LoRA bank
#
#   bash /workspace/sdx/runpod/sdx.sh          # runs the full pipeline
#   bash /workspace/sdx/runpod/sdx.sh sample     # generate after training
#
# Fresh pod (paste once):
#   apt-get update -qq && apt-get install -y -qq git && \
#   git clone --depth 1 -b feat/runpod-readiness-scraper-lora \
#     https://github.com/Llunarstack/sdx.git /workspace/sdx && \
#   bash /workspace/sdx/runpod/sdx.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib/ensure_repo.sh"
sdx_ensure_repo || exit 1

# shellcheck source=/dev/null
source "$HERE/lib/load_secrets.sh"
sdx_load_hf_token || echo "WARN: no HF auth — run: huggingface-cli login" >&2

# shellcheck source=/dev/null
source "$SDX_ROOT/runpod/env.defaults"
cd "$SDX_ROOT"

CMD="${1:-run}"
shift || true

_sdx_help() {
  cat <<'EOF'
SDX — one command runs the whole pipeline:

  (no args) / run     Full pipeline (what you want):
                        1. Download pretrained models (T5, CLIP, VLM, WD tagger, …)
                        2. Scrape ALL danbooru + rule34xxx
                        3. WD EVA02 tagger enriches every image's tags
                        4. VLM + RAG rewrites captions for training quality
                        5. Train base DiT checkpoint
                        6. Train artist/style LoRA bank (mix weights at sample time)
  sample              Generate: @artist and @style:anime control LoRA weights
  setup               Install deps only (first pod boot)
  train               Re-train only (skip scrape/download; for retries)
  data                Steps 1-4 only (no training)

Inference after training:
  SDX_PROMPT="@wlop @style:anime 1girl, cherry blossoms" bash /workspace/sdx/runpod/sdx.sh sample
  --artist-strength 1.2  scales the artist LoRA

Fresh pod:
  git clone --depth 1 -b feat/runpod-readiness-scraper-lora \
    https://github.com/Llunarstack/sdx.git /workspace/sdx
EOF
}

_sdx_check_scrape_secrets() {
  python3 - <<'PY' 2>/dev/null || return 0
import os, sys
sys.path.insert(0, os.environ.get("SDX_ROOT", "/workspace/sdx"))
from scripts.scrape.secrets_config import get_secrets_path, parse_secrets_file
path = get_secrets_path(os.environ.get("SDX_SECRETS_FILE"))
if not path.is_file():
    sys.exit(1)
need = {"danbooru", "rule34xxx"}
have = set(parse_secrets_file(path).keys())
sys.exit(0 if need.issubset(have) else 1)
PY
  echo "ERROR: booru credentials missing in ${SDX_SECRETS_FILE:-/workspace/secret.txt}" >&2
  echo "  HF login does NOT cover danbooru/rule34 scraping." >&2
  echo "  runpod/secret.txt is gitignored — create /workspace/secret.txt on the pod:" >&2
  echo "    • RunPod web UI: open /workspace/secret.txt in the file browser and paste" >&2
  echo "    • Or shell:  cat > /workspace/secret.txt <<'EOF'" >&2
  echo "                 danbooru" >&2
  echo "                 user: ..." >&2
  echo "                 api: ..." >&2
  echo "                 EOF" >&2
  echo "  Template: runpod/secrets.example.txt" >&2
  exit 1
}

_sdx_run_full() {
  # Force pipeline defaults (env.defaults uses train/1-GPU — wrong for full run).
  _NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [ "${_NGPU:-0}" -lt 1 ] && _NGPU=1
  export SDX_MODEL_PROFILE=ultimate
  export SDX_SCRAPE_SITES="${SDX_SCRAPE_SITES:-danbooru rule34xxx}"
  export SDX_MAX_POSTS=0
  export SDX_USE_WD_TAGGER=1
  export SDX_PROMPT_RESEARCH=1
  export SDX_TRAIN_LORA_BANK=1
  export SDX_FULL_TRAIN_FEATURES=1
  export SDX_EPOCHS=20
  export SDX_NPROC_PER_NODE="${SDX_NPROC_PER_NODE:-$_NGPU}"
  export SDX_GLOBAL_BATCH_SIZE="${SDX_GLOBAL_BATCH_SIZE:-$(( SDX_NPROC_PER_NODE * 12 ))}"

  cat <<EOF
╔══════════════════════════════════════════════════════════════╗
║  SDX full pipeline                                           ║
║  1. Pretrained models (profile=$SDX_MODEL_PROFILE)             ║
║  2. Scrape: $SDX_SCRAPE_SITES (max_posts=$SDX_MAX_POSTS=unlimited)  ║
║  3. WD EVA02 tagger → richer per-image tags                   ║
║  4. VLM + RAG caption enrichment                             ║
║  5. Train base DiT (${SDX_EPOCHS} epochs, ${SDX_NPROC_PER_NODE}× GPU)       ║
║  6. Train LoRA bank → @artist / @style weights at sample     ║
╚══════════════════════════════════════════════════════════════╝
EOF

  _sdx_check_scrape_secrets
  exec bash "$SDX_ROOT/runpod/ultimate.sh" "$@"
}

case "$CMD" in
  help|-h|--help)
    _sdx_help
    ;;
  run|full)
    _sdx_run_full "$@"
    ;;
  setup)
    exec bash "$SDX_ROOT/runpod/setup.sh" "$@"
    ;;
  data)
    export SDX_MODEL_PROFILE="${SDX_MODEL_PROFILE:-ultimate}"
    export SDX_SCRAPE_SITES="${SDX_SCRAPE_SITES:-danbooru rule34xxx}"
    export SDX_MAX_POSTS="${SDX_MAX_POSTS:-0}"
    export SDX_USE_WD_TAGGER=1
    export SDX_PROMPT_RESEARCH="${SDX_PROMPT_RESEARCH:-1}"
    exec bash "$SDX_ROOT/runpod/ultimate.sh" --data-only "$@"
    ;;
  train)
    export SDX_NPROC_PER_NODE="${SDX_NPROC_PER_NODE:-3}"
    export SDX_GLOBAL_BATCH_SIZE="${SDX_GLOBAL_BATCH_SIZE:-36}"
    export SDX_FULL_TRAIN_FEATURES="${SDX_FULL_TRAIN_FEATURES:-1}"
    export SDX_EPOCHS="${SDX_EPOCHS:-20}"
    export SDX_TRAIN_LORA_BANK="${SDX_TRAIN_LORA_BANK:-1}"
    exec bash "$SDX_ROOT/runpod/ultimate.sh" --train-only "$@"
    ;;
  sample)
    exec bash "$SDX_ROOT/runpod/sample.sh" "$@"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    _sdx_help >&2
    exit 2
    ;;
esac
