#!/usr/bin/env bash
# Ultimate pipeline entry — same as start.sh (kept as alias).
SCRIPT="$(cd "$(dirname "$0")" && pwd)/start.sh"
exec bash "$SCRIPT" "$@"
