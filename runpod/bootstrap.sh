#!/usr/bin/env bash
# First-time pod only — clones repo, then runs sdx.sh. After clone, use sdx.sh directly.
set -euo pipefail

SDX_ROOT="${SDX_ROOT:-/workspace/sdx}"
export SDX_REPO_URL="${SDX_REPO_URL:-https://github.com/Llunarstack/sdx.git}"
export SDX_REPO_REF="${SDX_REPO_REF:-feat/runpod-readiness-scraper-lora}"

if [ ! -f "$SDX_ROOT/runpod/sdx.sh" ]; then
  command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
  echo "==> Cloning SDX -> $SDX_ROOT"
  rm -rf "$SDX_ROOT"
  git clone --depth 1 --branch "$SDX_REPO_REF" "$SDX_REPO_URL" "$SDX_ROOT" 2>/dev/null || {
    git clone --depth 1 "$SDX_REPO_URL" "$SDX_ROOT"
    git -C "$SDX_ROOT" checkout "$SDX_REPO_REF" 2>/dev/null || true
  }
fi

exec bash "$SDX_ROOT/runpod/sdx.sh" "${@:-train}"
