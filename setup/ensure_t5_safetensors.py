#!/usr/bin/env python3
"""Ensure T5-XXL has ``model.safetensors`` (avoids torch>=2.6 requirement for ``.bin``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _t5_dir() -> Path:
    base = Path(os.environ.get("SDX_PRETRAINED", REPO_ROOT / "pretrained"))
    local = REPO_ROOT / "pretrained"
    if local.is_symlink():
        base = local.resolve()
    enc = os.environ.get("SDX_TEXT_ENCODER", "").strip()
    if enc:
        return Path(enc)
    safetensors_dir = base / "T5-XXL-safetensors"
    if (safetensors_dir / "model.safetensors").is_file():
        return safetensors_dir
    return base / "T5-XXL"


def main() -> int:
    dest = _t5_dir()
    safetensors = dest / "model.safetensors"
    if safetensors.is_file() and safetensors.stat().st_size > 1_000_000_000:
        print(f"T5 safetensors OK: {safetensors}")
        return 0

    repo = "mcmonkey/google_t5-v1_1-xxl_encoderonly"
    print(f"Downloading T5 encoder safetensors from {repo} -> {dest} (~9 GB)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 2

    dest.mkdir(parents=True, exist_ok=True)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from utils.hf_secrets import apply_hf_token_to_env, get_hf_token

    apply_hf_token_to_env()
    hf_tok = get_hf_token()
    for attempt in range(1, 6):
        try:
            snapshot_download(
                repo,
                local_dir=str(dest),
                token=hf_tok,
                max_workers=2 if hf_tok else 1,
                allow_patterns=[
                    "model.safetensors",
                    "config.json",
                    "spiece.model",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
            )
            break
        except Exception as e:
            import re
            import time

            m = re.search(r"Retry after (\d+)", str(e), re.I)
            wait = int(m.group(1)) + 5 if m else min(120, 15 * attempt)
            print(f"  attempt {attempt}/5 failed; retry in {wait}s (set HF_TOKEN in secret.txt)", file=sys.stderr)
            time.sleep(wait)
    else:
        return 1
    if safetensors.is_file():
        print(f"Downloaded: {safetensors}")
        return 0
    print("Download finished but model.safetensors not found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
