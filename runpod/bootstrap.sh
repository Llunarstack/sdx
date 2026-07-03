#!/usr/bin/env bash
# Run from anywhere on the pod. Handles clone, update, then starts the full pipeline.
#
#   bash /workspace/sdx/runpod/bootstrap.sh
#
# Background:
#   nohup bash /workspace/sdx/runpod/bootstrap.sh > /workspace/sdx.log 2>&1 &
#   tail -f /workspace/sdx.log
set -euo pipefail

SDX="${SDX_ROOT:-/workspace/sdx}"
REPO="${SDX_REPO_URL:-https://github.com/Llunarstack/sdx.git}"
BRANCH="${SDX_REPO_REF:-feat/runpod-readiness-scraper-lora}"

command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }

if [ -d "$SDX/.git" ]; then
  echo "==> Updating $SDX"
  git -C "$SDX" fetch origin "$BRANCH" 2>/dev/null || git -C "$SDX" fetch origin 2>/dev/null || true
  git -C "$SDX" checkout "$BRANCH" 2>/dev/null || true
  git -C "$SDX" pull --ff-only origin "$BRANCH" 2>/dev/null || git -C "$SDX" pull --ff-only 2>/dev/null || true
elif [ -d "$SDX" ]; then
  echo "==> Removing broken $SDX (not a git repo)"
  rm -rf "$SDX"
fi

if [ ! -f "$SDX/runpod/sdx.sh" ]; then
  echo "==> Cloning SDX -> $SDX"
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$SDX" 2>/dev/null || {
    git clone --depth 1 "$REPO" "$SDX"
    git -C "$SDX" checkout "$BRANCH" 2>/dev/null || true
  }
fi

if [ ! -f "$SDX/runpod/sdx.sh" ]; then
  echo "ERROR: sdx.sh still missing after clone. Check branch $BRANCH on GitHub." >&2
  exit 1
fi

echo "==> Starting full SDX pipeline (models → scrape → tags → train → LoRA bank)"
exec bash "$SDX/runpod/sdx.sh" run "$@"
