"""
SDX **Superior Stack** — composable quality and grounding modules on top of DiT + diffusion.

These are real, testable building blocks (retrieval, multi-metric ranking, self-correction,
distillation helpers). They do not replace training; they raise the ceiling at inference and
data-prep time when wired through ``sample.py`` or ``python -m scripts.tools superior_generate``.
"""

from __future__ import annotations

from config.defaults.superior_stack import FlywheelPlan, SuperiorStackDefaults

from .auto_loop import AutoImproveConfig, build_auto_improve_argv, run_auto_improve
from .auto_stack import SuperiorPromptStack, apply_superior_prompt_stack
from .composite_ranker import CompositeRanker, RankWeights
from .distill import DistillStepPlan, LADDConfig, ladd_config_from_train
from .dpo_pipeline import DPOStageConfig, MinePairsConfig, run_alignment_loop, run_dpo_training
from .ensemble import EnsembleConfig, generate_ensemble
from .eval_report import build_markdown_report, write_report
from .flywheel import curate_manifest, run_flywheel
from .glyph_encoder import ByteHashGlyphEncoder
from .hard_negative import (
    HardNegativeBundle,
    load_hard_negatives_from_results,
    merge_negative_prompt,
    mine_hard_negatives,
)
from .inference_pipeline import SuperiorInferenceConfig, build_superior_sample_argv
from .model_soup import average_state_dicts, save_soup_checkpoint, soup_checkpoints
from .prompt_expand import ExpandConfig, expand_prompt, expand_prompt_heuristic
from .quality_gates import GateThresholds, QualityGateRunner
from .retrieval import TfidfFactIndex, build_tfidf_index_from_jsonl
from .reward_scorer import RewardWeights, UnifiedRewardScorer
from .self_correct import SelfCorrectConfig, SelfCorrectPolicy
from .vit_mining import ViTMineConfig, mine_vit_from_results_json, mine_vit_preference_pairs, score_image_vit

__all__ = [
    "apply_superior_prompt_stack",
    "AutoImproveConfig",
    "average_state_dicts",
    "build_auto_improve_argv",
    "build_superior_sample_argv",
    "build_tfidf_index_from_jsonl",
    "build_markdown_report",
    "ByteHashGlyphEncoder",
    "CompositeRanker",
    "curate_manifest",
    "DistillStepPlan",
    "DPOStageConfig",
    "EnsembleConfig",
    "ExpandConfig",
    "expand_prompt",
    "expand_prompt_heuristic",
    "FlywheelPlan",
    "generate_ensemble",
    "GateThresholds",
    "HardNegativeBundle",
    "LADDConfig",
    "ladd_config_from_train",
    "load_hard_negatives_from_results",
    "merge_negative_prompt",
    "mine_hard_negatives",
    "MinePairsConfig",
    "QualityGateRunner",
    "RankWeights",
    "RewardWeights",
    "run_alignment_loop",
    "run_auto_improve",
    "run_dpo_training",
    "run_flywheel",
    "save_soup_checkpoint",
    "SelfCorrectConfig",
    "SelfCorrectPolicy",
    "soup_checkpoints",
    "SuperiorInferenceConfig",
    "SuperiorPromptStack",
    "SuperiorStackDefaults",
    "TfidfFactIndex",
    "UnifiedRewardScorer",
    "ViTMineConfig",
    "mine_vit_from_results_json",
    "mine_vit_preference_pairs",
    "score_image_vit",
    "write_report",
]
