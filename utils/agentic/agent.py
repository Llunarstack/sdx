"""
**ImageGenerationAgent** — main Act–Reflect–Think–Act orchestrator.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config.defaults.agentic_stack import AgenticStackDefaults

from .experience import (
    TrajectoryExperience,
    append_experience_memory,
    apply_memory_to_prompt,
    distill_trajectory_experience,
    load_experience_memory,
)
from .planner import default_stop_conditions, plan_from_prompt
from .reflector import reflect_on_result, reflect_on_result_llm
from .state import AgentContext, AgentTrace, TrajectoryRecord
from .tools import AgentTool, ToolRegistry


@dataclass(slots=True)
class AgentRunResult:
    accepted: bool
    out_path: str
    trace: AgentTrace
    experience: Optional[TrajectoryExperience] = None


class ImageGenerationAgent:
    """
    Autonomous agent: plan → tools → verify → reflect → retry.

    Supports optional experience memory (GenEvolve-style trajectory distillation).
    """

    def __init__(
        self,
        defaults: Optional[AgenticStackDefaults] = None,
        *,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.defaults = defaults or AgenticStackDefaults()
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])

    def run(self, ctx: AgentContext) -> AgentRunResult:
        d = self.defaults
        work = Path(ctx.work_dir)
        if not ctx.dry_run:
            work.mkdir(parents=True, exist_ok=True)

        memory_path = work / "experience_memory.jsonl"
        memory = load_experience_memory(memory_path)
        seeded_prompt = apply_memory_to_prompt(ctx.prompt, memory)

        trace = AgentTrace(goal_prompt=ctx.prompt)
        stops = default_stop_conditions(d)
        trajectories: List[TrajectoryRecord] = []

        registry = ToolRegistry(
            ctx,
            repo_root=self.repo_root,
            num_candidates=d.num_candidates,
            pick_metric=d.pick_metric,
            expand_prompt=d.expand_prompt,
            self_correct=d.self_correct,
            rag_top_k=d.rag_top_k,
            extra_sample_args=d.extra_sample_args,
        )
        registry.session["prompt"] = seeded_prompt

        use_rag = bool(str(ctx.local_rag_jsonl or "").strip())
        plan = plan_from_prompt(
            seeded_prompt,
            use_rag=use_rag,
            expand=d.expand_prompt,
            verify=True,
            self_correct=d.self_correct,
        )

        accepted = False
        tid = uuid.uuid4().hex[:8]
        tool_seq: List[str] = []

        for loop in range(max(1, stops.max_outer_loops)):
            trace.iterations = loop + 1
            for step in plan.steps:
                tool_name = str(step.kwargs.get("tool", ""))
                tool = AgentTool(tool_name) if tool_name in AgentTool._value2member_map_ else None
                if tool is None:
                    continue
                tool_seq.append(tool.value)
                registry.execute(tool)

            v = registry.execute(AgentTool.verify)
            metrics = dict(v.data) if v.ok else {}
            _qwen = str(getattr(ctx, "qwen_path", "") or getattr(d, "qwen_path", "") or "").strip()
            if _qwen:
                ref = reflect_on_result_llm(
                    metrics,
                    iteration=loop,
                    prompt=str(registry.session.get("prompt", ctx.prompt)),
                    qwen_path=_qwen,
                    device=str(ctx.device),
                    min_composite=d.min_composite_accept,
                    min_clip=d.min_clip_accept,
                    expected_text=str(ctx.expected_text or ""),
                )
            else:
                ref = reflect_on_result(
                    metrics,
                    iteration=loop,
                    min_composite=d.min_composite_accept,
                    min_clip=d.min_clip_accept,
                    expected_text=str(ctx.expected_text or ""),
                )
            trace.reflections.append(
                {
                    "iteration": loop,
                    "accepted": ref.accepted,
                    "summary": ref.verdict.summary,
                    "suggestion": ref.verdict.suggestion,
                }
            )
            if ref.accepted:
                accepted = True
                break
            if ref.prompt_patch:
                registry.session["prompt"] = (registry.session["prompt"] + ", " + ref.prompt_patch).strip(", ")
                trace.experience_patches.append(ref.prompt_patch)
            if ref.negative_patch:
                neg = registry.session.get("negative_prompt", "")
                registry.session["negative_prompt"] = (neg + ", " + ref.negative_patch).strip(", ")
            registry.execute(AgentTool.generate, out_suffix=f"_retry{loop}")

        out_path = registry.session.get("last_out", ctx.out)
        metrics = dict(registry.session.get("metrics", {}))
        traj = TrajectoryRecord(
            trajectory_id=tid,
            prompt_final=str(registry.session.get("prompt", ctx.prompt)),
            negative_prompt=str(registry.session.get("negative_prompt", "")),
            tool_sequence=tool_seq,
            out_path=str(out_path),
            composite=float(metrics.get("composite", 0.0)),
            clip_score=float(metrics.get("clip", 0.0)),
            metrics=metrics,
            accepted=accepted,
        )
        trajectories.append(traj)
        trace.trajectories = trajectories
        trace.best_trajectory_id = tid if accepted else (trajectories[0].trajectory_id if trajectories else "")
        trace.final_out = str(out_path)

        exp = distill_trajectory_experience(ctx.prompt, trajectories) if len(trajectories) >= 2 else None
        if exp and not ctx.dry_run:
            append_experience_memory(memory_path, exp)

        trace.save(work / "agent_trace.json")
        return AgentRunResult(accepted=accepted, out_path=str(out_path), trace=trace, experience=exp)

    def run_visual_brain(self, ctx: AgentContext) -> AgentRunResult:
        """
        Full visual understanding pipeline: search → understand → dissect → generate → edit.
        """
        from utils.brain.visual_brain import VisualBrain, VisualBrainConfig

        cfg = VisualBrainConfig(
            web_search=bool(getattr(ctx, "web_search", True)),
            max_search_images=4,
            num_candidates=self.defaults.num_candidates,
            pick_metric=self.defaults.pick_metric,
            self_correct=self.defaults.self_correct,
            max_edit_loops=self.defaults.max_reflect_loops,
            min_coverage=self.defaults.min_composite_accept,
            min_clip=self.defaults.min_clip_accept,
            creative_rag=True,
        )
        brain = VisualBrain(config=cfg, defaults=self.defaults, repo_root=self.repo_root)
        res = brain.run(
            ckpt=ctx.ckpt,
            prompt=ctx.prompt,
            work_dir=ctx.work_dir,
            out=ctx.out,
            device=ctx.device,
            negative_prompt=ctx.negative_prompt,
            reference_images=list(getattr(ctx, "reference_images", []) or []),
            local_rag_jsonl=str(ctx.local_rag_jsonl or ""),
            expected_text=str(ctx.expected_text or ""),
            vit_ckpt=str(ctx.vit_ckpt or ""),
            dry_run=bool(ctx.dry_run),
        )
        trace = AgentTrace(goal_prompt=ctx.prompt)
        trace.iterations = res.iterations
        trace.final_out = res.out_path
        trace.reflections = [{"accepted": res.accepted, "metrics": res.metrics}]
        if res.trace_path:
            trace.save(Path(ctx.work_dir) / "visual_brain_trace.json")
        traj = TrajectoryRecord(
            trajectory_id=uuid.uuid4().hex[:8],
            prompt_final=res.brief.build_generation_prompt(),
            negative_prompt=res.brief.negative_prompt,
            tool_sequence=["visual_brain"],
            out_path=res.out_path,
            composite=float(res.metrics.get("composite", 0.0)),
            clip_score=float(res.metrics.get("clip", 0.0)),
            metrics=res.metrics,
            accepted=res.accepted,
        )
        trace.trajectories = [traj]
        return AgentRunResult(accepted=res.accepted, out_path=res.out_path, trace=trace)

    def run_evolve(self, ctx: AgentContext, *, variants: Optional[int] = None) -> AgentRunResult:
        """
        Run multiple tool trajectories (different expand/rag toggles), distill experience.
        """
        n = max(2, int(variants or self.defaults.trajectory_variants))
        work = Path(ctx.work_dir)
        all_traj: List[TrajectoryRecord] = []
        trace = AgentTrace(goal_prompt=ctx.prompt)

        configs = [
            {"expand": True, "rag": True},
            {"expand": True, "rag": False},
            {"expand": False, "rag": True},
        ][:n]

        for i, cfg in enumerate(configs):
            sub = AgentContext(
                ckpt=ctx.ckpt,
                prompt=ctx.prompt,
                work_dir=str(work / f"variant_{i}"),
                out=f"variant_{i}.png",
                device=ctx.device,
                negative_prompt=ctx.negative_prompt,
                local_rag_jsonl=ctx.local_rag_jsonl if cfg["rag"] else "",
                vit_ckpt=ctx.vit_ckpt,
                expected_text=ctx.expected_text,
                dry_run=ctx.dry_run,
            )
            d = AgenticStackDefaults(
                expand_prompt=bool(cfg["expand"]),
                self_correct=self.defaults.self_correct,
                num_candidates=self.defaults.num_candidates,
                pick_metric=self.defaults.pick_metric,
            )
            agent = ImageGenerationAgent(d, repo_root=self.repo_root)
            res = agent.run(sub)
            if res.trace.trajectories:
                t = res.trace.trajectories[0]
                t.trajectory_id = f"v{i}"
                all_traj.append(t)

        exp = distill_trajectory_experience(ctx.prompt, all_traj)
        if exp and not ctx.dry_run:
            append_experience_memory(work / "experience_memory.jsonl", exp)
            if exp.prompt_delta:
                ctx.prompt = apply_memory_to_prompt(ctx.prompt, [exp])

        best = max(all_traj, key=lambda t: t.composite, default=None)
        trace.trajectories = all_traj
        trace.experience_patches = [exp.prompt_delta] if exp and exp.prompt_delta else []
        trace.best_trajectory_id = best.trajectory_id if best else ""
        trace.final_out = best.out_path if best else ""
        trace.save(work / "evolve_trace.json")

        return AgentRunResult(
            accepted=bool(best and best.accepted),
            out_path=best.out_path if best else "",
            trace=trace,
            experience=exp,
        )


__all__ = ["AgentRunResult", "ImageGenerationAgent"]
