#!/usr/bin/env bash
# Ultimate H100 training alias — full pipeline: bash runpod/ultimate.sh
exec "$(cd "$(dirname "$0")" && pwd)/ultimate.sh" "$@"