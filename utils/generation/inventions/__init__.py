"""
SDX Invention Lab — novel inference/architecture modules aimed at unsolved T2I failures.

These are *original SDX designs* (names + mechanics), inspired by known gaps
(composition, counting, negation, anatomy, plastic look) but not copies of a
single paper's code.
"""

from __future__ import annotations

from utils.generation.inventions.anatomon import AnatomonPlan, apply_anatomon_prompts, plan_anatomon
from utils.generation.inventions.bindlock import BindLockPlan, apply_bindlock_to_prompts, plan_bindlock
from utils.generation.inventions.consensus_denoise import ConsensusConfig, blend_consensus_latents, plan_consensus_seeds
from utils.generation.inventions.countgate import CountGatePlan, apply_countgate_prompts, plan_countgate
from utils.generation.inventions.failure_oracle import FailureOracleReport, diagnose_failures, repair_plan_from_failures
from utils.generation.inventions.friction_texture import (
    FrictionSchedule,
    apply_friction_to_prompts,
    friction_noise_scale,
)
from utils.generation.inventions.hydra_slots import HydraSlotPlan, plan_hydra_slots
from utils.generation.inventions.negatron import NegatronPlan, apply_negatron, plan_negatron
from utils.generation.inventions.spectra_routing import SpectraState, spectra_cfg_scale, spectra_hf_boost
from utils.generation.inventions.stack import InventionStackResult, apply_invention_stack
from utils.generation.inventions.wire import (
    InventionRuntime,
    apply_invention_to_namespace,
    enrich_prompts_with_inventions,
    prepare_invention_runtime,
)

__all__ = [
    "AnatomonPlan",
    "BindLockPlan",
    "ConsensusConfig",
    "CountGatePlan",
    "FailureOracleReport",
    "FrictionSchedule",
    "HydraSlotPlan",
    "InventionRuntime",
    "InventionStackResult",
    "NegatronPlan",
    "SpectraState",
    "apply_anatomon_prompts",
    "apply_bindlock_to_prompts",
    "apply_countgate_prompts",
    "apply_friction_to_prompts",
    "apply_invention_stack",
    "apply_invention_to_namespace",
    "apply_negatron",
    "blend_consensus_latents",
    "diagnose_failures",
    "enrich_prompts_with_inventions",
    "friction_noise_scale",
    "plan_anatomon",
    "plan_bindlock",
    "plan_consensus_seeds",
    "plan_countgate",
    "plan_hydra_slots",
    "plan_negatron",
    "prepare_invention_runtime",
    "repair_plan_from_failures",
    "spectra_cfg_scale",
    "spectra_hf_boost",
]
