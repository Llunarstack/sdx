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
# shellcheck source=/dev/null
source "$HERE/lib/install_scrape_secrets.sh"
# shellcheck source=/dev/null
source "$HERE/lib/turbo_scrape.sh"
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
                        2. HF datasets (turbo): danbooru + rule34 + e621 + gelbooru packs
                        3. WD EVA02 tagger enriches every image's tags
                        4. VLM + RAG rewrites captions for training quality
                        5. Train base DiT checkpoint
                        6. Train artist/style LoRA bank (mix weights at sample time)
  sample              Generate: @artist and @style:anime control LoRA weights
  setup               Install deps only (first pod boot)
  train               Re-train only (skip scrape/download; for retries)
  data                Steps 1-4 only (no training)
  models              Download pretrained weights only (no scrape/train)
  secrets             Install /workspace/secret.txt from runpod/secret.txt

Booru API scrape (slow): set SDX_DATA_SOURCE=booru + secret.txt.
Default is HF bulk datasets — needs hf auth login only.
Upload your local runpod/secret.txt to /workspace/sdx/runpod/secret.txt via RunPod
file browser, then:  bash runpod/sdx.sh secrets

Fresh pod:
  git clone --depth 1 -b feat/runpod-readiness-scraper-lora \
    https://github.com/Llunarstack/sdx.git /workspace/sdx
EOF
}

_sdx_check_scrape_secrets() {
  if [ "${SDX_DATA_SOURCE:-hf}" = "hf" ]; then
    return 0
  fi
  if sdx_ensure_scrape_secrets 2>/dev/null; then
    return 0
  fi
  echo "ERROR: booru credentials missing in ${SDX_SECRETS_FILE:-/workspace/secret.txt}" >&2
  echo "  HF / hf auth login does NOT cover danbooru or rule34 scraping." >&2
  echo "" >&2
  echo "  Fix (pick one):" >&2
  echo "    1. RunPod file browser → upload your secret.txt to:" >&2
  echo "         /workspace/sdx/runpod/secret.txt" >&2
  echo "       then run:  bash runpod/sdx.sh secrets" >&2
  echo "    2. Or paste directly into /workspace/secret.txt (file browser)" >&2
  echo "    3. Or:  cat > /workspace/secret.txt <<'EOF'  (paste, then EOF on its own line)" >&2
  echo "" >&2
  echo "  Template: runpod/secrets.example.txt" >&2
  exit 1
}

_sdx_apply_turbo_scrape() { sdx_apply_turbo_scrape; }

_sdx_run_full() {
  # Force pipeline defaults (env.defaults uses train/1-GPU — wrong for full run).
  _NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [ "${_NGPU:-0}" -lt 1 ] && _NGPU=1
  _sdx_apply_turbo_scrape
  # Full pipeline always uses all four HF packs (ignore stale RunPod template env).
  export SDX_DATA_SOURCE="${SDX_DATA_SOURCE:-hf}"
  export SDX_DATA_SITES="danbooru rule34xxx e621 rule34xyz"
  export SDX_SCRAPE_SITES="$SDX_DATA_SITES"
  export SDX_MODEL_PROFILE=ultimate
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
║  2. Data: ${SDX_DATA_SOURCE:-hf} (${SDX_DATA_SITES:-danbooru rule34xxx e621 rule34xyz})     ║
║  3. WD EVA02 tagger → richer per-image tags                   ║
║  4. VLM + RAG caption enrichment                             ║
║  5. Train base DiT (${SDX_EPOCHS} epochs, ${SDX_NPROC_PER_NODE}× GPU)       ║
║  6. Train LoRA bank → @artist / @style weights at sample     ║
╚══════════════════════════════════════════════════════════════╝
EOF

  _sdx_check_scrape_secrets
  exec bash "$SDX_ROOT/runpod/ultimate.sh" "$@"
}

_sdx_run_models() {
  export SDX_MODEL_PROFILE="${SDX_MODEL_PROFILE:-ultimate}"
  echo "==> Pretrained models only (profile=$SDX_MODEL_PROFILE)"
  python setup/download_pretrained.py \
    --dest "${SDX_PRETRAINED:-/workspace/pretrained}" \
    --profile "$SDX_MODEL_PROFILE" \
    --workers "${SDX_DL_WORKERS:-4}"
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
    _sdx_apply_turbo_scrape
    export SDX_DATA_SOURCE="${SDX_DATA_SOURCE:-hf}"
    export SDX_DATA_SITES="danbooru rule34xxx e621 rule34xyz"
    export SDX_SCRAPE_SITES="$SDX_DATA_SITES"
    export SDX_MODEL_PROFILE="${SDX_MODEL_PROFILE:-ultimate}"
    export SDX_MAX_POSTS="${SDX_MAX_POSTS:-0}"
    export SDX_USE_WD_TAGGER=1
    export SDX_PROMPT_RESEARCH="${SDX_PROMPT_RESEARCH:-1}"
    _sdx_check_scrape_secrets
    exec bash "$SDX_ROOT/runpod/ultimate.sh" --data-only "$@"
    ;;
  models)
    _sdx_run_models "$@"
    ;;
  secrets)
    if sdx_ensure_scrape_secrets; then
      echo "Scrape secrets OK: ${SDX_SECRETS_FILE:-/workspace/secret.txt}"
      python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ.get("SDX_ROOT", "/workspace/sdx"))
from scripts.scrape.secrets_config import get_secrets_path, parse_secrets_file
print("Sites:", sorted(parse_secrets_file(get_secrets_path()).keys()))
PY
    else
      echo "No valid scrape secrets yet." >&2
      echo "Upload secret.txt to $SDX_ROOT/runpod/secret.txt via RunPod file browser, then re-run." >&2
      exit 1
    fi
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
