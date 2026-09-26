#!/usr/bin/env bash
# Download training images from Hugging Face (danbooru, rule34, e621, gelbooru).
# This is NOT a live booru API scrape — only HF dataset exports.
#
#   bash runpod/datasets.sh           # background
#   bash runpod/datasets.sh --fg      # foreground
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib/ensure_repo.sh"
sdx_ensure_repo || exit 1
# shellcheck source=/dev/null
source "$HERE/lib/load_secrets.sh"
# shellcheck source=/dev/null
source "$HERE/lib/hf_sites.sh"
# shellcheck source=/dev/null
source "$HERE/lib/turbo_hf.sh"
sdx_load_hf_token || echo "WARN: run hf auth login for gated datasets" >&2
# shellcheck source=/dev/null
source "$SDX_ROOT/runpod/env.defaults"
sdx_export_hf_sites
sdx_apply_turbo_hf
cd "$SDX_ROOT"

LOCK="${SDX_DATA_LOCK:-$SDX_DATA/.data_download.lock}"
LOG="${SDX_DATA_LOG:-/workspace/datasets.log}"
FG=0
for arg in "$@"; do
  case "$arg" in
    --fg) FG=1 ;;
    *) echo "Unknown flag: $arg (use --fg)" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOCK")" "$(dirname "$LOG")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Dataset download already running. Lock: $LOCK" >&2
  exit 1
fi

echo "HF datasets (turbo): $SDX_HF_SITES"
echo "  workers=$SDX_HF_EXPORT_WORKERS parallel_packs=$SDX_HF_PARALLEL_PACKS"
echo "  dest: $SDX_DATA"
echo "  log:  $LOG"

if [ "$FG" = 1 ]; then
  bash "$SDX_ROOT/runpod/download.sh" --data-only --skip-preprocess 2>&1 | tee -a "$LOG"
else
  nohup bash "$SDX_ROOT/runpod/download.sh" --data-only --skip-preprocess >>"$LOG" 2>&1 &
  echo "PID: $!"
  echo "  tail -f $LOG"
fi
