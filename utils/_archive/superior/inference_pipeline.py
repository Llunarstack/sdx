"""
Build ``sample.py`` argument lists for **Superior** multi-candidate + ranking flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(slots=True)
class SuperiorInferenceConfig:
    """Knobs for a high-quality default sampling run."""

    num_candidates: int = 4
    pick_metric: str = "combo"
    use_composite_rank: bool = True
    cfg_scale: float = 7.0
    cfg_rescale: float = 0.7
    compile_inference: bool = False
    local_rag_jsonl: Optional[str] = None
    rag_top_k: int = 8
    clip_guard_threshold: float = 0.0
    self_correct_clip: bool = False
    steps: int = 28
    fdg_cfg_strength: float = 0.0
    zeresfdg_strength: float = 1.0
    feature_cache_delta: float = 0.0
    block_cache_threshold: float = 0.18
    taylor_cache: bool = True
    rcfgpp_tangent: float = 0.0
    apg_parallel_eta: float = -1.0
    cfg_zero_star: bool = True
    qsilk_micrograin: float = 0.1


def build_superior_sample_argv(
    *,
    ckpt: str,
    prompt: str,
    out: str,
    config: Optional[SuperiorInferenceConfig] = None,
) -> List[str]:
    """
    Return argv fragment for subprocess ``sample.py`` invocation (no executable name).

    Example::

        import subprocess, sys
        from utils.superior import build_superior_sample_argv, SuperiorInferenceConfig

        cfg = SuperiorInferenceConfig()
        argv = [
            sys.executable,
            "sample.py",
            *build_superior_sample_argv(ckpt="results/run/best.pt", prompt="a cat", out="out.png", config=cfg),
        ]
        subprocess.run(argv, check=True)
    """
    c = config or SuperiorInferenceConfig()
    args: List[str] = [
        "--ckpt",
        ckpt,
        "--prompt",
        prompt,
        "--out",
        out,
        "--num",
        str(max(1, c.num_candidates)),
        "--cfg-scale",
        str(c.cfg_scale),
        "--cfg-rescale",
        str(c.cfg_rescale),
        "--steps",
        str(c.steps),
    ]
    if c.num_candidates >= 2:
        metric = "superior_composite" if c.use_composite_rank else c.pick_metric
        args.extend(["--pick-best", metric])
    if c.compile_inference:
        args.append("--compile-inference")
    if c.local_rag_jsonl:
        args.extend(["--local-rag-jsonl", c.local_rag_jsonl, "--local-rag-top-k", str(c.rag_top_k)])
    if c.clip_guard_threshold > 0:
        args.extend(["--clip-guard-threshold", str(c.clip_guard_threshold)])
    if c.self_correct_clip:
        args.append("--superior-self-correct")
    if c.fdg_cfg_strength > 0:
        args.extend(["--fdg-cfg-strength", str(c.fdg_cfg_strength)])
    if c.zeresfdg_strength > 0:
        args.extend(["--zeresfdg-strength", str(c.zeresfdg_strength)])
    if c.cfg_zero_star:
        args.append("--cfg-zero-star")
    if c.qsilk_micrograin > 0:
        args.extend(["--qsilk-micrograin", str(c.qsilk_micrograin)])
    if c.feature_cache_delta > 0:
        args.extend(["--feature-cache-delta", str(c.feature_cache_delta)])
    if c.block_cache_threshold > 0:
        args.extend(["--block-cache-thresh", str(c.block_cache_threshold)])
    if c.taylor_cache and c.block_cache_threshold > 0:
        args.append("--taylor-cache")
    if c.rcfgpp_tangent > 0:
        args.extend(["--rcfgpp-tangent", str(c.rcfgpp_tangent)])
    if c.apg_parallel_eta >= 0:
        args.extend(["--apg-parallel-eta", str(c.apg_parallel_eta)])
    return args


__all__ = ["SuperiorInferenceConfig", "build_superior_sample_argv"]
