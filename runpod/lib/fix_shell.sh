#!/usr/bin/env bash
# Strip Windows CRLF from shell scripts (RunPod file browser / git autocrlf).
sdx_fix_shell_scripts() {
  local root="${1:-${SDX_ROOT:-/workspace/sdx}}"
  local f
  [ -d "$root/runpod" ] || return 0
  while IFS= read -r -d '' f; do
    sed -i 's/\r$//' "$f" 2>/dev/null || true
  done < <(find "$root/runpod" -name '*.sh' -print0 2>/dev/null)
}
