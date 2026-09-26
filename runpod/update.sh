#!/usr/bin/env bash
# Force-sync repo on a pod (discards local edits) and fix CRLF on shell scripts.
#
#   bash runpod/update.sh
#   bash runpod/update.sh datasets --fg
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/lib/ensure_repo.sh"
# shellcheck source=/dev/null
source "$HERE/lib/fix_shell.sh"

export SDX_ROOT="${SDX_ROOT:-/workspace/sdx}"
export SDX_REPO_REF="${SDX_REPO_REF:-feat/runpod-readiness-scraper-lora}"

cd "$SDX_ROOT"
echo "==> Fetch origin"
git fetch origin
echo "==> Reset to origin/$SDX_REPO_REF (local pod edits discarded)"
git reset --hard "origin/$SDX_REPO_REF"
sdx_fix_shell_scripts "$SDX_ROOT"
echo "==> At $(git rev-parse --short HEAD)"
exec bash "$SDX_ROOT/runpod/sdx.sh" "$@"
