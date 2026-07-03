#!/usr/bin/env bash
# One-liner bootstrap for a **fresh** RunPod (no repo yet). Paste in Web Terminal:
#
#   apt-get update -qq && apt-get install -y -qq git
#   git clone --depth 1 -b feat/runpod-readiness-scraper-lora \
#     https://github.com/Llunarstack/sdx.git /workspace/sdx
#   bash /workspace/sdx/runpod/bootstrap.sh train
#
# Or after clone:
#   bash /workspace/sdx/runpod/bootstrap.sh [smoke|data|train|full]
set -euo pipefail

SDX_ROOT="${SDX_ROOT:-/workspace/sdx}"
export SDX_REPO_URL="${SDX_REPO_URL:-https://github.com/Llunarstack/sdx.git}"
export SDX_REPO_REF="${SDX_REPO_REF:-feat/runpod-readiness-scraper-lora}"

if ! command -v git >/dev/null 2>&1; then
  echo "Installing git..."
  apt-get update -qq && apt-get install -y -qq git
fi

if [ ! -f "$SDX_ROOT/runpod/start.sh" ]; then
  echo "==> Cloning SDX -> $SDX_ROOT"
  rm -rf "$SDX_ROOT"
  git clone --depth 1 --branch "$SDX_REPO_REF" "$SDX_REPO_URL" "$SDX_ROOT" 2>/dev/null || {
    git clone --depth 1 "$SDX_REPO_URL" "$SDX_ROOT"
    git -C "$SDX_ROOT" checkout "$SDX_REPO_REF" 2>/dev/null || true
  }
fi

exec bash "$SDX_ROOT/runpod/budget.sh" "${@:-train}"
