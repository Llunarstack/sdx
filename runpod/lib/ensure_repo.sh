#!/usr/bin/env bash
# Ensure SDX repo exists at SDX_ROOT. Source from runpod/*.sh — do not run directly.
#
#   source "$(dirname "$0")/lib/ensure_repo.sh"
#   sdx_ensure_repo || exit 1

sdx_ensure_repo() {
  export SDX_ROOT="${SDX_ROOT:-/workspace/sdx}"
  export SDX_REPO_URL="${SDX_REPO_URL:-https://github.com/Llunarstack/sdx.git}"
  export SDX_REPO_REF="${SDX_REPO_REF:-feat/runpod-readiness-scraper-lora}"

  if [ -f "$SDX_ROOT/runpod/sdx.sh" ]; then
    cd "$SDX_ROOT" || return 1
    git fetch origin 2>/dev/null || true
    git checkout "$SDX_REPO_REF" 2>/dev/null || true
    if ! git pull --ff-only 2>/dev/null; then
      echo "WARN: git pull failed — local edits may block updates." >&2
      echo "  Fix: cd $SDX_ROOT && git stash -u && git pull --ff-only" >&2
    fi
    return 0
  fi

  # Legacy checkout without sdx.sh — force refresh
  if [ -d "$SDX_ROOT/.git" ]; then
    echo "WARN: old SDX checkout (no sdx.sh) — run: bash $SDX_ROOT/runpod/bootstrap.sh" >&2
    cd "$SDX_ROOT" || return 1
    git pull --ff-only 2>/dev/null || true
    [ -f "$SDX_ROOT/runpod/sdx.sh" ] && return 0
  fi

  echo "SDX not found at $SDX_ROOT" >&2
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: install git first: apt-get update && apt-get install -y git" >&2
    return 1
  fi

  if [ -e "$SDX_ROOT" ] && [ ! -d "$SDX_ROOT/.git" ]; then
    echo "ERROR: $SDX_ROOT exists but is not an SDX git checkout." >&2
    echo "  mv $SDX_ROOT ${SDX_ROOT}.bak   # or set SDX_ROOT to another path" >&2
    return 1
  fi

  echo "==> Cloning $SDX_REPO_URL (branch $SDX_REPO_REF) -> $SDX_ROOT"
  mkdir -p "$(dirname "$SDX_ROOT")"
  if ! git clone --depth 1 --branch "$SDX_REPO_REF" "$SDX_REPO_URL" "$SDX_ROOT" 2>/dev/null; then
    echo "WARN: branch $SDX_REPO_REF unavailable — cloning default branch" >&2
    git clone --depth 1 "$SDX_REPO_URL" "$SDX_ROOT"
    cd "$SDX_ROOT" || return 1
    git checkout "$SDX_REPO_REF" 2>/dev/null || true
  fi
  cd "$SDX_ROOT" || return 1
  echo "==> SDX ready at $SDX_ROOT"
  return 0
}

sdx_die_if_missing() {
  local path="$1"
  local hint="$2"
  if [ ! -e "$path" ]; then
    echo "ERROR: missing $path" >&2
    echo "  $hint" >&2
    return 1
  fi
  return 0
}
