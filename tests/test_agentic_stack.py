"""Agentic stack: planner, reflector, experience, agent dry-run."""

from __future__ import annotations

import json
from pathlib import Path

from config.defaults.agentic_stack import AgenticStackDefaults
from research.agi_image.planning.generation_plan import GenerationStepKind
from utils.agentic import (
    AgentContext,
    ImageGenerationAgent,
    TrajectoryRecord,
    apply_memory_to_prompt,
    build_default_plan,
    default_stop_conditions,
    distill_trajectory_experience,
    load_experience_memory,
    plan_from_prompt,
    reflect_on_result,
)
from utils.agentic.experience import TrajectoryExperience, append_experience_memory
from utils.agentic.tools import AgentTool, ToolRegistry


def test_plan_from_prompt_full_utpc() -> None:
    plan = plan_from_prompt("a cat", use_rag=True, expand=True, verify=True, self_correct=True)
    kinds = [s.kind for s in plan.steps]
    assert GenerationStepKind.knowledge_retrieval in kinds
    assert GenerationStepKind.refine_prompt in kinds
    assert GenerationStepKind.diffusion_sample in kinds
    assert GenerationStepKind.verify_visual in kinds
    assert GenerationStepKind.critique_revise in kinds
    assert len(plan.steps) == 5


def test_plan_from_prompt_minimal() -> None:
    plan = plan_from_prompt("x", use_rag=False, expand=False, verify=False, self_correct=False)
    assert len(plan.steps) == 1
    assert plan.steps[0].kind == GenerationStepKind.diffusion_sample


def test_build_default_plan_respects_defaults() -> None:
    d = AgenticStackDefaults(expand_prompt=False, self_correct=False)
    plan = build_default_plan("dog", d, use_rag=False)
    kinds = [s.kind for s in plan.steps]
    assert GenerationStepKind.refine_prompt not in kinds
    assert GenerationStepKind.critique_revise not in kinds


def test_reflect_accepts_good_metrics() -> None:
    out = reflect_on_result({"composite": 0.8, "clip": 0.3, "sharpness": 200.0}, iteration=0)
    assert out.accepted
    assert out.prompt_patch == ""


def test_reflect_patches_low_metrics() -> None:
    out = reflect_on_result({"composite": 0.4, "clip": 0.1, "sharpness": 50.0}, iteration=1)
    assert not out.accepted
    assert "sharp" in out.prompt_patch or "subject" in out.prompt_patch
    assert "blur" in out.negative_patch.lower() or "quality" in out.negative_patch.lower()


def test_distill_trajectory_experience() -> None:
    trajectories = [
        TrajectoryRecord(
            trajectory_id="a",
            prompt_final="cat, studio lighting, tack sharp",
            negative_prompt="blur",
            tool_sequence=["rag_retrieve", "expand_prompt", "generate", "verify"],
            out_path="a.png",
            composite=0.75,
            clip_score=0.3,
            metrics={"composite": 0.75, "clip": 0.3},
            accepted=True,
        ),
        TrajectoryRecord(
            trajectory_id="b",
            prompt_final="cat",
            negative_prompt="",
            tool_sequence=["generate", "verify"],
            out_path="b.png",
            composite=0.4,
            clip_score=0.1,
            metrics={"composite": 0.4, "clip": 0.1},
            accepted=False,
        ),
    ]
    exp = distill_trajectory_experience("cat", trajectories)
    assert exp is not None
    assert exp.best_id == "a"
    assert exp.worst_id == "b"
    assert "studio" in exp.prompt_delta.lower() or "sharp" in exp.prompt_delta.lower()
    assert "rag_retrieve" in exp.tool_delta


def test_experience_memory_roundtrip(tmp_path: Path) -> None:
    mem = tmp_path / "experience_memory.jsonl"
    exp = TrajectoryExperience(
        goal_prompt="dog",
        best_id="x",
        worst_id="y",
        prompt_delta="cinematic lighting",
        negative_delta="blur",
    )
    append_experience_memory(mem, exp)
    loaded = load_experience_memory(mem)
    assert len(loaded) == 1
    assert loaded[0].prompt_delta == "cinematic lighting"
    merged = apply_memory_to_prompt("a dog in park", loaded)
    assert "cinematic lighting" in merged


def test_tool_registry_dry_run(tmp_path: Path) -> None:
    ctx = AgentContext(
        ckpt="fake.pt",
        prompt="sunset",
        work_dir=str(tmp_path),
        out="out.png",
        dry_run=True,
    )
    reg = ToolRegistry(ctx, repo_root=tmp_path, expand_prompt=True)
    reg.execute(AgentTool.expand_prompt)
    gen = reg.execute(AgentTool.generate)
    assert gen.ok
    verify = reg.execute(AgentTool.verify)
    assert verify.ok
    assert verify.data["composite"] > 0.0


def test_agent_run_dry_run(tmp_path: Path) -> None:
    ctx = AgentContext(
        ckpt="fake.pt",
        prompt="mountain lake",
        work_dir=str(tmp_path / "run"),
        dry_run=True,
    )
    d = AgenticStackDefaults(max_reflect_loops=1, min_composite_accept=0.8)
    res = ImageGenerationAgent(d, repo_root=tmp_path).run(ctx)
    assert res.accepted
    trace_path = tmp_path / "run" / "agent_trace.json"
    assert trace_path.is_file()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["goal_prompt"] == "mountain lake"
    assert len(trace["trajectories"]) == 1


def test_agent_evolve_dry_run(tmp_path: Path) -> None:
    ctx = AgentContext(
        ckpt="fake.pt",
        prompt="red car",
        work_dir=str(tmp_path / "evolve"),
        dry_run=True,
    )
    res = ImageGenerationAgent(AgenticStackDefaults(), repo_root=tmp_path).run_evolve(ctx, variants=2)
    assert len(res.trace.trajectories) == 2
    assert (tmp_path / "evolve" / "evolve_trace.json").is_file()


def test_default_stop_conditions() -> None:
    stops = default_stop_conditions(AgenticStackDefaults(max_reflect_loops=5))
    assert stops.max_outer_loops == 5


def test_role_pipeline_dry_run(tmp_path: Path) -> None:
    from utils.agentic.roles import RolePipeline

    ctx = AgentContext(
        ckpt="fake.pt",
        prompt="forest path",
        work_dir=str(tmp_path),
        out="roles.png",
        dry_run=True,
    )
    pipe = RolePipeline.from_context(ctx, repo_root=tmp_path)
    res = pipe.run()
    assert res.ok
    assert len(res.stages) == 3
    assert res.stages[0].role == "reasoner"
    assert res.stages[1].role == "designer"
    assert res.stages[2].role == "verifier"
    assert res.metrics["composite"] > 0.0
