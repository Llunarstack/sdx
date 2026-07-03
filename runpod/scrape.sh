#!/usr/bin/env bash
# Start booru scrape in background (one instance only).
#
#   bash runpod/scrape.sh          # default unlimited crawl
#   bash runpod/scrape.sh --turbo  # H100: max download threads + faster frame sample
#   bash runpod/scrape.sh --fg     # foreground (for debugging)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/runpod/env.defaults"

SCRAPE_LOCK="${SDX_SCRAPE_LOCK:-$SDX_DATA/.scrape.lock}"
LOG="${SDX_SCRAPE_LOG:-/workspace/scrape.log}"
FG=0
TURBO=0
for arg in "$@"; do
  case "$arg" in
    --fg) FG=1 ;;
    --turbo) TURBO=1 ;;
    *) echo "Unknown flag: $arg (use --fg | --turbo)" >&2; exit 2 ;;
  esac
done

if [ "$TURBO" = 1 ]; then
  export SDX_SCRAPE_WORKERS="${SDX_SCRAPE_WORKERS:-96}"
  export SDX_FRAME_FPS="${SDX_FRAME_FPS:-2}"
  export SDX_MAX_FRAMES_PER_POST="${SDX_MAX_FRAMES_PER_POST:-120}"
  export SDX_SPLIT_FRAMES="${SDX_SPLIT_FRAMES:-1}"
  export SDX_MAX_POSTS="${SDX_MAX_POSTS:-0}"
  # API limits unchanged — only download parallelism goes up.
fi

mkdir -p "$(dirname "$SCRAPE_LOCK")" "$(dirname "$LOG")"
exec 9>"$SCRAPE_LOCK"
if ! flock -n 9; then
  echo "Scrape already running. Lock: $SCRAPE_LOCK" >&2
  echo "  pgrep -af download_datasets" >&2
  exit 1
fi

CMD=(
  env
  SDX_MAX_POSTS="${SDX_MAX_POSTS:-0}"
  SDX_SCRAPE_WORKERS="${SDX_SCRAPE_WORKERS:-48}"
  SDX_SPLIT_FRAMES="${SDX_SPLIT_FRAMES:-1}"
  SDX_MAX_FRAMES_PER_POST="${SDX_MAX_FRAMES_PER_POST:-0}"
  SDX_FRAME_FPS="${SDX_FRAME_FPS:-2}"
  bash "$ROOT/runpod/download.sh" --data-only
)

if [ "$FG" = 1 ]; then
  echo "Scrape (foreground) -> log also at $LOG"
  "${CMD[@]}" 2>&1 | tee -a "$LOG"
else
  echo "Starting scrape in background -> $LOG"
  nohup "${CMD[@]}" >>"$LOG" 2>&1 &
  echo "PID: $!"
  echo "  tail -f $LOG"
  echo "  bash runpod/scrape_stats.sh"
fi
