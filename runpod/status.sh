#!/usr/bin/env bash
# Live SDX pipeline monitor: HF export, tagging, enrich, training, LoRA bank.
#
#   bash runpod/status.sh              # refresh every 5s
#   bash runpod/status.sh --interval 2
#   bash runpod/status.sh --once
#   bash runpod/sdx.sh status
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib/hf_sites.sh"
# shellcheck source=/dev/null
source "$HERE/env.defaults"
sdx_export_hf_sites

INTERVAL=5
ONCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --once) ONCE=1; shift ;;
    --interval)
      INTERVAL="${2:-5}"
      shift 2
      ;;
    --interval=*) INTERVAL="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) shift ;;
  esac
done

DATASETS_LOG="${SDX_DATA_LOG:-/workspace/datasets.log}"
TRAIN_LOG="${SDX_TRAIN_LOG:-/workspace/train.log}"
ULTIMATE_LOG="${SDX_ULTIMATE_LOG:-/workspace/ultimate.log}"
DATA_LOCK="${SDX_DATA_LOCK:-$SDX_DATA/.data_download.lock}"

_lines() {
  local f="$1"
  if [ -f "$f" ] && [ -s "$f" ]; then
    wc -l <"$f" | tr -d ' '
  else
    echo 0
  fi
}

_human_bytes() {
  local n="${1:-0}"
  if [ "$n" -lt 1024 ] 2>/dev/null; then
    echo "${n}B"
  elif [ "$n" -lt 1048576 ] 2>/dev/null; then
    echo "$(( n / 1024 ))K"
  elif [ "$n" -lt 1073741824 ] 2>/dev/null; then
    echo "$(( n / 1048576 ))M"
  else
    echo "$(( n / 1073741824 ))G"
  fi
}

_procs() {
  local pat="$1"
  pgrep -af "$pat" 2>/dev/null | grep -v 'pgrep -af' | grep -v 'runpod/status.sh' || true
}

_phase() {
  if _procs '[h]f_export_to_sdx_manifest' | grep -q .; then echo "HF export (writing images)"
  elif _procs '[d]ownload_hf_datasets' | grep -q .; then echo "HF export (orchestrator)"
  elif _procs '[t]ag_manifest_wd' | grep -q .; then echo "WD tagger"
  elif _procs '[e]nrich_manifest_captions' | grep -q .; then echo "Caption enrich (VLM+RAG)"
  elif _procs '[p]reprocess_control_maps' | grep -q .; then echo "Control maps"
  elif _procs '[p]recompute_latents' | grep -q .; then echo "Latent precompute"
  elif _procs '[t]rain.py' | grep -q .; then echo "DiT training"
  elif _procs '[t]rain_lora_bank' | grep -q .; then echo "LoRA bank training"
  elif _procs '[u]ltimate.sh' | grep -q .; then echo "Ultimate pipeline"
  elif _procs '[d]ownload.sh' | grep -q .; then echo "Download / data prep"
  elif _procs '[d]atasets.sh' | grep -q .; then echo "HF datasets"
  elif [ -f "$DATA_LOCK" ] && _procs '[d]ownload_hf_datasets|[h]f_export' | grep -q .; then echo "HF export"
  else echo "idle"
  fi
}

_log_tail() {
  local file="$1" n="${2:-4}"
  if [ -f "$file" ]; then
    tail -n "$n" "$file" 2>/dev/null | sed 's/^/    /'
  else
    echo "    (no log)"
  fi
}

_render() {
  local now prev_t prev_total total site rows sz_b rate elapsed
  now="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  prev_t="${SDX_STATUS_PREV_T:-0}"
  prev_total="${SDX_STATUS_PREV_TOTAL:-0}"
  total=0

  clear 2>/dev/null || printf '\033[H\033[2J'

  printf '╔══════════════════════════════════════════════════════════════════════╗\n'
  printf '║  SDX live status — %s                          ║\n' "$now"
  printf '╚══════════════════════════════════════════════════════════════════════╝\n'
  printf '\n'

  local rev git_ok
  rev="?"
  if [ -d "${SDX_ROOT:-/workspace/sdx}/.git" ]; then
    git_ok="$(cd "${SDX_ROOT:-/workspace/sdx}" && git rev-parse --short HEAD 2>/dev/null || echo "?")"
    rev="$git_ok"
  fi
  printf '  repo: %-12s  phase: %s\n' "$rev" "$(_phase)"
  printf '  sites: %s\n' "$SDX_HF_SITES"
  if [ -n "${SDX_HF_MAX_SAMPLES:-}" ] && [ "${SDX_HF_MAX_SAMPLES:-0}" -gt 0 ] 2>/dev/null; then
    printf '  cap:   SDX_HF_MAX_SAMPLES=%s\n' "$SDX_HF_MAX_SAMPLES"
  fi
  printf '\n'

  printf '── Active processes ──────────────────────────────────────────────────\n'
  local any=0
  for pat in download_hf_datasets hf_export_to_sdx_manifest tag_manifest_wd enrich_manifest_captions precompute_latents 'train.py' train_lora_bank ultimate.sh download.sh; do
    local hits
    hits="$(_procs "$pat")"
    if [ -n "$hits" ]; then
      any=1
      printf '  [%s]\n' "$pat"
      echo "$hits" | sed 's/^/    /'
    fi
  done
  if [ "$any" = 0 ]; then
    printf '  (none)\n'
  fi
  if [ -f "$DATA_LOCK" ]; then
    printf '  lock: %s\n' "$DATA_LOCK"
  fi
  printf '\n'

  if command -v nvidia-smi >/dev/null 2>&1; then
    printf '── GPU ───────────────────────────────────────────────────────────────\n'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | while IFS=, read -r idx util memu memt temp; do
          idx="${idx// /}"; util="${util// /}"; memu="${memu// /}"; memt="${memt// /}"; temp="${temp// /}"
          printf '  GPU %s  util=%3s%%  mem=%s/%s MiB  %s°C\n' "$idx" "$util" "$memu" "$memt" "$temp"
        done || printf '  (nvidia-smi failed)\n'
    printf '\n'
  fi

  printf '── HF dataset export (images = manifest rows) ────────────────────────\n'
  printf '  %-14s %10s %8s %10s\n' "SITE" "ROWS" "DISK" "MANIFEST"
  for site in $SDX_HF_SITES; do
    local d m rows du_out
    d="$SDX_DATA/$site"
    m="$d/manifest.jsonl"
    rows="$(_lines "$m")"
    total=$((total + rows))
    du_out="$(du -sh "$d" 2>/dev/null | cut -f1 || echo '-')"
    if [ -f "$m" ]; then
      sz_b="$(stat -c %s "$m" 2>/dev/null || echo 0)"
      printf '  %-14s %10s %8s %10s\n' "$site" "$rows" "$du_out" "$(_human_bytes "$sz_b")"
    else
      printf '  %-14s %10s %8s %10s\n' "$site" "0" "$du_out" "-"
    fi
  done

  elapsed=1
  if [ "$prev_t" -gt 0 ]; then
    elapsed=$(( $(date +%s) - prev_t ))
    [ "$elapsed" -lt 1 ] && elapsed=1
  fi
  if [ "$prev_total" -gt 0 ] && [ "$total" -ge "$prev_total" ]; then
    rate=$(( (total - prev_total) / elapsed ))
    printf '  %-14s %10s %8s %10s\n' "TOTAL" "$total" "" "${rate}/s"
  else
    printf '  %-14s %10s\n' "TOTAL" "$total"
  fi
  SDX_STATUS_PREV_T="$(date +%s)"
  SDX_STATUS_PREV_TOTAL="$total"
  export SDX_STATUS_PREV_T SDX_STATUS_PREV_TOTAL

  local data_du
  data_du="$(du -sh "$SDX_DATA" 2>/dev/null | cut -f1 || echo '-')"
  printf '  data dir total: %s\n' "$data_du"
  printf '\n'

  printf '── Pipeline manifests ────────────────────────────────────────────────\n'
  _manifest_line() { printf '  %-10s %s rows\n' "$1:" "$(_lines "$2")"; }
  _manifest_line "combined" "$SDX_DATA/combined/manifest.jsonl"
  _manifest_line "tagged" "${SDX_TAGGED_MANIFEST:-$SDX_DATA/tagged/manifest.jsonl}"
  _manifest_line "enriched" "${SDX_ENRICHED_MANIFEST:-$SDX_DATA/enriched/manifest.jsonl}"
  _manifest_line "control" "${SDX_CONTROL_MANIFEST:-$SDX_DATA/control/manifest.jsonl}"
  printf '\n'

  local latents cached
  latents="${SDX_LATENT_CACHE:-$SDX_DATA/latent_cache}"
  cached=0
  if [ -d "$latents" ]; then
    cached="$(find "$latents" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  fi
  printf '── Training ──────────────────────────────────────────────────────────\n'
  printf '  latent cache: %s / %s .pt files\n' "$cached" "$(_lines "${SDX_DATA}/combined/manifest.jsonl")"
  local best
  best=""
  if [ -d "${SDX_RESULTS:-/workspace/results}" ]; then
    best="$(find "${SDX_RESULTS}" -name 'best.pt' 2>/dev/null | while read -r f; do
      [ -f "$f" ] || continue
      printf '%s %s\n' "$(stat -c %Y "$f" 2>/dev/null || echo 0)" "$f"
    done | sort -rn | head -1 | cut -d' ' -f2-)"
  fi
  if [ -n "$best" ]; then
    printf '  base ckpt:  %s\n' "$best"
  else
    printf '  base ckpt:  (none yet)\n'
  fi
  local lora_n=0
  if [ -d "${SDX_LORA_BANK:-$SDX_DATA/lora_bank}" ]; then
    lora_n="$(find "${SDX_LORA_BANK:-$SDX_DATA/lora_bank}" -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
  fi
  printf '  lora bank:  %s adapters\n' "$lora_n"
  if [ -f "$TRAIN_LOG" ]; then
    printf '  train log (last metrics):\n'
    grep -E 'step|epoch|loss|Train DiT' "$TRAIN_LOG" 2>/dev/null | tail -3 | sed 's/^/    /' || _log_tail "$TRAIN_LOG" 3
  fi
  printf '\n'

  printf '── Recent logs ───────────────────────────────────────────────────────\n'
  printf '  datasets (%s):\n' "$DATASETS_LOG"
  _log_tail "$DATASETS_LOG" 5
  if [ -f "$ULTIMATE_LOG" ]; then
    printf '  ultimate (%s):\n' "$ULTIMATE_LOG"
    _log_tail "$ULTIMATE_LOG" 3
  fi
  printf '\n'
  if [ "$ONCE" = 0 ]; then
    printf '  refreshing every %ss — Ctrl+C to stop\n' "$INTERVAL"
  fi
}

while true; do
  _render
  [ "$ONCE" = 1 ] && break
  sleep "$INTERVAL"
done
