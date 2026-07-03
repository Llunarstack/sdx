#!/usr/bin/env bash
# Ultimate H100 training — use runpod/train_h100.sh (kept as alias).
exec "$(cd "$(dirname "$0")" && pwd)/train_h100.sh" "$@"