#!/usr/bin/env bash
# Shared DiT image-training feature flags (everything except video/frontier).
#
# Source from runpod/train.sh / train_h100.sh after env.defaults.
# Set SDX_FULL_TRAIN_FEATURES=0 to disable the advanced stack (baseline DiT only).
#
# WARNING: architecture flags (REPA, SSM, RoPE, register tokens, creativity, qk-norm)
# change checkpoint shape — do not --resume a checkpoint trained without them.
set -euo pipefail

sdx_build_train_feature_args() {
  local -n _out=$1
  if [ "${SDX_FULL_TRAIN_FEATURES:-1}" = "0" ]; then
    return 0
  fi

  # --- Text conditioning (triple = T5 + CLIP-L + bigG fusion trained with DiT) ---
  local enc_mode="${SDX_TEXT_ENCODER_MODE:-triple}"
  _out+=(--text-encoder-mode "$enc_mode")
  if [ -n "${SDX_TEXT_ENCODER:-}" ]; then
    _out+=(--text-encoder "$SDX_TEXT_ENCODER")
  fi
  if [ -n "${SDX_CLIP_TEXT_ENCODER_L:-}" ]; then
    _out+=(--clip-text-encoder-l "$SDX_CLIP_TEXT_ENCODER_L")
  fi
  if [ -n "${SDX_CLIP_TEXT_ENCODER_BIGG:-}" ]; then
    _out+=(--clip-text-encoder-bigg "$SDX_CLIP_TEXT_ENCODER_BIGG")
  fi
  if [ "$enc_mode" = "penta" ]; then
    [ -n "${SDX_CLIP_TEXT_ENCODER_H:-}" ] && _out+=(--clip-text-encoder-h "$SDX_CLIP_TEXT_ENCODER_H")
    [ -n "${SDX_CLIP_TEXT_ENCODER_LONG:-}" ] && _out+=(--clip-text-encoder-long "$SDX_CLIP_TEXT_ENCODER_LONG")
  fi

  # --- DiT architecture / loss auxiliaries ---
  _out+=(
    --repa-weight "${SDX_REPA_WEIGHT:-0.5}"
    --repa-encoder-model "${SDX_REPA_ENCODER:-facebook/dinov2-base}"
    --repa-out-dim "${SDX_REPA_OUT_DIM:-768}"
    --ssm-every-n "${SDX_SSM_EVERY_N:-4}"
    --ssm-kernel-size "${SDX_SSM_KERNEL_SIZE:-7}"
    --num-register-tokens "${SDX_NUM_REGISTER_TOKENS:-4}"
    --use-rope
    --rope-base "${SDX_ROPE_BASE:-10000}"
    --layerscale-init "${SDX_LAYERSCALE_INIT:-1e-5}"
    --qk-norm
    --drop-path-rate "${SDX_DROP_PATH_RATE:-0.05}"
    --patch-se
    --prompt-reinject-every-n "${SDX_PROMPT_REINJECT_EVERY_N:-4}"
    --prompt-reinject-alpha "${SDX_PROMPT_REINJECT_ALPHA:-0.1}"
    --prompt-timestep-schedule-enabled
    --creativity-embed-dim "${SDX_CREATIVITY_EMBED_DIM:-64}"
    --creativity-jitter-std "${SDX_CREATIVITY_JITTER_STD:-0.1}"
    --token-keep-ratio "${SDX_TOKEN_KEEP_RATIO:-0.95}"
  )

  if [ "${SDX_MOE_NUM_EXPERTS:-0}" -gt 0 ] 2>/dev/null; then
    _out+=(--moe-num-experts "${SDX_MOE_NUM_EXPERTS}" --moe-top-k "${SDX_MOE_TOP_K:-2}")
  fi

  # --- Caption / dataset training packs (not separate VLMs — prompt shaping) ---
  _out+=(
    --train-style-guidance-mode "${SDX_TRAIN_STYLE_GUIDANCE:-auto}"
    --region-caption-mode "${SDX_REGION_CAPTION_MODE:-append}"
    --train-art-guidance-mode "${SDX_TRAIN_ART_GUIDANCE:-auto}"
    --train-anatomy-guidance "${SDX_TRAIN_ANATOMY_GUIDANCE:-auto}"
    --boost-adherence-caption
    --train-shortcomings-mitigation "${SDX_TRAIN_SHORTCOMINGS:-auto}"
    --caption-unicode-normalize
    --foveated-train-prob "${SDX_FOVEATED_TRAIN_PROB:-0.15}"
    --train-originality-prob "${SDX_TRAIN_ORIGINALITY_PROB:-0.1}"
  )

  if [ "${SDX_USE_HIERARCHICAL_CAPTIONS:-1}" = "1" ]; then
    _out+=(--use-hierarchical-captions)
  fi

  # --- ControlNet (image) when control maps were preprocessed (optional per batch) ---
  if [ "${SDX_ENABLE_CONTROL_TRAIN:-1}" = "1" ]; then
    local ctrl="${SDX_CONTROL_MANIFEST:-}"
    if [ -n "$ctrl" ] && [ -s "$ctrl" ]; then
      _out+=(
        --control-cond-dim 1
        --control-num-types "${SDX_CONTROL_NUM_TYPES:-9}"
        --control-scale "${SDX_CONTROL_SCALE:-0.85}"
      )
    fi
  fi
}
