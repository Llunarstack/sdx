#!/usr/bin/env bash
# Back-compat alias — use runpod/status.sh for live monitoring.
exec bash "$(cd "$(dirname "$0")" && pwd)/status.sh" --once "$@"
