"""
SDX **Agentic Stack** — autonomous plan → tool → verify → reflect → evolve loops.

Inspired by GenEvolve (tool-orchestrated trajectories), VisionCreator UTPC
(Understand / Think / Plan / Create), and VisionCreator-R1 Act–Reflect–Think–Act.

Operational layer on top of ``utils/superior/`` and ``sample.py``. Research schemas
live in ``research/agi_image/``.
"""

from __future__ import annotations

from config.defaults.agentic_stack import AgenticStackDefaults

from .agent import AgentRunResult, ImageGenerationAgent
from .experience import (
    TrajectoryExperience,
    apply_memory_to_prompt,
    distill_trajectory_experience,
    load_experience_memory,
)
from .planner import build_default_plan, default_stop_conditions, plan_from_prompt, plan_visual_brain
from .reflector import ReflectionOutcome, reflect_on_result
from .roles import RolePipeline, RolePipelineResult, RoleStageResult
from .state import AgentContext, AgentTrace, TrajectoryRecord
from .tools import AgentTool, ToolRegistry, ToolResult

__all__ = [
    "AgentContext",
    "AgentRunResult",
    "AgentTool",
    "AgentTrace",
    "AgenticStackDefaults",
    "ImageGenerationAgent",
    "ReflectionOutcome",
    "RolePipeline",
    "RolePipelineResult",
    "RoleStageResult",
    "ToolRegistry",
    "ToolResult",
    "TrajectoryExperience",
    "TrajectoryRecord",
    "apply_memory_to_prompt",
    "build_default_plan",
    "default_stop_conditions",
    "distill_trajectory_experience",
    "load_experience_memory",
    "plan_from_prompt",
    "plan_visual_brain",
    "reflect_on_result",
]
