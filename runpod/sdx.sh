#!/usr/bin/env bash
# SDX on RunPod — one script:
#   models → HF datasets (danbooru, rule34, e621, gelbooru) → WD tags → enrich → train
#
#   bash /workspace/sdx/runpod/sdx.sh
#
# Fresh pod:
#   apt-get update -qq && apt-get install -y -qq git && \
#   git clone --depth 1 -b feat/runpod-readiness-scraper-lora \
#     https://github.com/Llunarstack/sdx.git /workspace/sdx && \
#   hf auth login && \
#   bash /workspace/sdx/runpod/sdx.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib/ensure_repo.sh"
sdx_ensure_repo || exit 1

# shellcheck source=/dev/null
source "$HERE/lib/load_secrets.sh"
# shellcheck source=/dev/null
source "$HERE/lib/hf_sites.sh"
sdx_load_hf_token || echo "WARN: run hf auth login for gated HF datasets" >&2

# shellcheck source=/dev/null
source "$SDX_ROOT/runpod/env.defaults"
cd "$SDX_ROOT"

CMD="${1:-run}"
shift || true

_sdx_help() {
  cat <<'EOF'
SDX — full pipeline (Hugging Face datasets only, no live booru API):

  (no args) / run     Full pipeline:
                        1. Pretrained models (T5, CLIP, VLM, WD tagger, …)
                        2. HF datasets: danbooru, rule34, e621, gelbooru
                        3. WD EVA02 tagger
                        4. VLM + RAG captions
                        5. Train base DiT
                        6. Train LoRA bank (@artist / @style at sample)
  datasets            Download HF datasets only (same 4 packs)
  data                Models + datasets + tag + enrich (no training)
  models              Pretrained weights only
  train               Train only (data must already exist)
  sample              Generate images
  setup               Install deps

Auth: hf auth login  (required for some HF dataset packs)

HF packs (see setup/hf_dataset_packs.json):
  danbooru   -> vikhyatoolkit/danbooru2023
  rule34xxx  -> deepghs/rule34_full
  e621       -> NebulaeWis/e621-2024-webp-4Mpixel
  rule34xyz  -> deepghs/gelbooru-webp-4Mpixel

Cap download size:  export SDX_HF_MAX_SAMPLES=100000
Re-download pack:   export SDX_HF_FORCE=1
EOF
}

_sdx_run_full() {
  _NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [ "${_NGPU:-0}" -lt 1 ] && _NGPU=1
  sdx_export_hf_sites
  export SDX_MODEL_PROFILE=ultimate
  export SDX_USE_WD_TAGGER=1
  export SDX_PROMPT_RESEARCH=1
  export SDX_TRAIN_LORA_BANK=1
  export SDX_FULL_TRAIN_FEATURES=1
  export SDX_EPOCHS=20
  export SDX_NPROC_PER_NODE="${SDX_NPROC_PER_NODE:-$_NGPU}"
  export SDX_GLOBAL_BATCH_SIZE="${SDX_GLOBAL_BATCH_SIZE:-$(( SDX_NPROC_PER_NODE * 12 ))}"

  cat <<EOF
╔══════════════════════════════════════════════════════════════╗
║  SDX full pipeline (Hugging Face datasets)                   ║
║  1. Pretrained models (profile=$SDX_MODEL_PROFILE)             ║
║  2. HF datasets: $SDX_HF_SITES
║  3. WD EVA02 tagger                                            ║
║  4. VLM + RAG captions                                         ║
║  5. Train base DiT (${SDX_EPOCHS} epochs, ${SDX_NPROC_PER_NODE}× GPU)       ║
║  6. LoRA bank (@artist / @style)                               ║
╚══════════════════════════════════════════════════════════════╝
EOF
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
  datasets)
    exec bash "$SDX_ROOT/runpod/datasets.sh" "$@"
    ;;
  data)
    sdx_export_hf_sites
    export SDX_MODEL_PROFILE="${SDX_MODEL_PROFILE:-ultimate}"
    export SDX_USE_WD_TAGGER=1
    export SDX_PROMPT_RESEARCH="${SDX_PROMPT_RESEARCH:-1}"
    exec bash "$SDX_ROOT/runpod/ultimate.sh" --data-only "$@"
    ;;
  models)
    _sdx_run_models "$@"
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
  secrets|scrape)
    echo "This pipeline uses Hugging Face datasets only — no booru API credentials needed." >&2
    echo "Run: hf auth login" >&2
    echo "Then: bash runpod/sdx.sh datasets   (or bash runpod/sdx.sh for full pipeline)" >&2
    exit 0
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    _sdx_help >&2
    exit 2
    ;;
esac
