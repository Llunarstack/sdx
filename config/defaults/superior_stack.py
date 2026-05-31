"""
Superior Stack **defaults** — single import for flywheel + inference tooling.

Not applied automatically by ``train.py``; used by ``utils/superior/flywheel.py`` and CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class SuperiorStackDefaults:
    """Recommended knobs for high-quality inference + alignment loops."""

    preset: str = "sdxl"
    num_candidates: int = 4
    pick_best: str = "superior_composite"
    steps: int = 28
    cfg_scale: float = 7.0
    cfg_rescale: float = 0.7
    expand_prompt: bool = True
    self_correct: bool = True
    compile_inference: bool = False
    local_rag_top_k: int = 8
    # Alignment loop
    dpo_steps: int = 500
    dpo_beta: float = 300.0
    dpo_timestep_weight: str = "high_noise"
    dpo_safeguard: float = 0.85
    dpo_ref_ema_alpha: float = 0.01
    fdg_cfg_strength: float = 0.0
    zeresfdg_strength: float = 1.0
    apg_parallel_eta: float = -1.0
    cfg_zero_star: bool = True
    qsilk_micrograin: float = 0.1
    taylor_cache: bool = False
    taylor_cache_order: int = 1
    block_cache_threshold: float = 0.18
    auto_loop_iterations: int = 1
    model_soup: bool = True
    soup_weight_base: float = 0.35
    soup_weight_dpo: float = 0.65
    benchmark_suite_pack: str = "top_contender_proxy_v1"
    preference_min_margin: float = 0.08
    # Curation
    dedup: str = "phash"
    min_caption_len: int = 8
    max_caption_len: int = 512


@dataclass(slots=True)
class FlywheelPlan:
    """End-to-end quality flywheel plan (curate → align → promote)."""

    base_ckpt: str
    work_dir: str = "flywheel_run"
    manifest_in: Optional[str] = None
    manifest_out: Optional[str] = None
    local_rag_jsonl: Optional[str] = None
    vit_ckpt: Optional[str] = None
    promote_path: str = "flywheel_run/best.pt"
    skip_curate: bool = False
    skip_align: bool = False
    defaults: SuperiorStackDefaults = field(default_factory=SuperiorStackDefaults)
    extra_auto_loop_args: List[str] = field(default_factory=list)


DEFAULTS = SuperiorStackDefaults()

__all__ = ["DEFAULTS", "FlywheelPlan", "SuperiorStackDefaults"]
