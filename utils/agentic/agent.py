"""Agentic image generation: plan -> tool -> verify -> reflect -> evolve (GenEvolve trajectories, VisionCreator UTPC planning, Act-Reflect-Think-Act) with Designer / Verifier / Reasoner roles and trajectory memory, plus Visual Brain (understand references, search, scene brief, generate / edit / verify) and its session memory box."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config.presets import AgenticStackDefaults
from PIL import Image

from utils.captioning import ImageUnderstanding, download_search_hits, search_reference_images, understand_images
from utils.modeling.model_paths import repo_root as _sdx_repo_root
from utils.prompt.rag import merge_facts_into_prompt


@dataclass(slots=True)
class AgentContext:
    """Runtime context for one agent session."""

    ckpt: str
    prompt: str
    work_dir: str = "agentic_run"
    out: str = "agent_out.png"
    device: str = "cuda"
    negative_prompt: str = ""
    local_rag_jsonl: str = ""
    vit_ckpt: str = ""
    expected_text: str = ""
    qwen_path: str = ""
    reference_images: list[str] = field(default_factory=list)
    web_search: bool = True
    repo_root: str | None = None
    dry_run: bool = False


@dataclass(slots=True)
class TrajectoryRecord:
    """One full tool-orchestrated attempt (GenEvolve-style trajectory)."""

    trajectory_id: str
    prompt_final: str
    negative_prompt: str
    tool_sequence: list[str]
    out_path: str
    composite: float = 0.0
    clip_score: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    notes: str = ""


@dataclass(slots=True)
class AgentTrace:
    """Full session trace for logs / experience distillation."""

    goal_prompt: str
    iterations: int = 0
    trajectories: list[TrajectoryRecord] = field(default_factory=list)
    reflections: list[dict[str, Any]] = field(default_factory=list)
    experience_patches: list[str] = field(default_factory=list)
    best_trajectory_id: str = ""
    final_out: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_prompt": self.goal_prompt,
            "iterations": self.iterations,
            "trajectories": [asdict(t) for t in self.trajectories],
            "reflections": list(self.reflections),
            "experience_patches": list(self.experience_patches),
            "best_trajectory_id": self.best_trajectory_id,
            "final_out": self.final_out,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(slots=True)
class TrajectoryExperience:
    """Structured lesson from comparing trajectories on the same goal."""

    goal_prompt: str
    best_id: str
    worst_id: str
    prompt_delta: str
    negative_delta: str
    metric_delta: dict[str, float] = field(default_factory=dict)
    tool_delta: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def distill_trajectory_experience(
    goal_prompt: str,
    trajectories: Sequence[TrajectoryRecord],
) -> TrajectoryExperience | None:
    """
    Compare best vs worst trajectory; abstract prompt/tool differences.
    """
    if len(trajectories) < 2:
        return None
    ranked = sorted(trajectories, key=lambda t: float(t.composite), reverse=True)
    best, worst = ranked[0], ranked[-1]
    if best.trajectory_id == worst.trajectory_id:
        return None

    def _unique_suffix(a: str, b: str) -> str:
        al = set(x.strip().lower() for x in a.split(",") if x.strip())
        bl = set(x.strip().lower() for x in b.split(",") if x.strip())
        extra = [x for x in al - bl if x]
        return ", ".join(sorted(extra)[:8])

    prompt_delta = _unique_suffix(best.prompt_final, worst.prompt_final)
    neg_delta = _unique_suffix(worst.negative_prompt, best.negative_prompt)
    tool_delta = [t for t in best.tool_sequence if t not in worst.tool_sequence]
    metric_delta = {
        k: float(getattr(best, "metrics", {}).get(k, 0.0)) - float(getattr(worst, "metrics", {}).get(k, 0.0))
        for k in set(list(getattr(best, "metrics", {}).keys()) + list(getattr(worst, "metrics", {}).keys()))
    }
    return TrajectoryExperience(
        goal_prompt=goal_prompt,
        best_id=best.trajectory_id,
        worst_id=worst.trajectory_id,
        prompt_delta=prompt_delta,
        negative_delta=neg_delta,
        metric_delta=metric_delta,
        tool_delta=tool_delta,
    )


def append_experience_memory(path: Path, exp: TrajectoryExperience) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(exp.to_json() + "\n")


def load_experience_memory(path: Path, *, limit: int = 32) -> list[TrajectoryExperience]:
    if not path.is_file():
        return []
    out: list[TrajectoryExperience] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(
                TrajectoryExperience(
                    goal_prompt=str(d.get("goal_prompt", "")),
                    best_id=str(d.get("best_id", "")),
                    worst_id=str(d.get("worst_id", "")),
                    prompt_delta=str(d.get("prompt_delta", "")),
                    negative_delta=str(d.get("negative_delta", "")),
                    metric_delta=dict(d.get("metric_delta", {})),
                    tool_delta=list(d.get("tool_delta", [])),
                )
            )
        except json.JSONDecodeError:
            continue
    return out[-limit:]


def apply_memory_to_prompt(prompt: str, memory: Sequence[TrajectoryExperience]) -> str:
    """Merge recurring prompt deltas from experience memory."""
    if not memory:
        return prompt
    bits: list[str] = []
    seen: set[str] = set()
    pl = prompt.lower()
    for exp in memory:
        for chunk in exp.prompt_delta.split(","):
            c = chunk.strip()
            if c and c.lower() not in seen and c.lower() not in pl:
                seen.add(c.lower())
                bits.append(c)
    if not bits:
        return prompt
    return prompt + ", " + ", ".join(bits[:6])


class GenerationStepKind(str, Enum):
    refine_prompt = "refine_prompt"
    diffusion_sample = "diffusion_sample"
    inpaint_region = "inpaint_region"
    verify_visual = "verify_visual"
    verify_text_ocr = "verify_text_ocr"
    critique_revise = "critique_revise"
    semantic_segment = "semantic_segment"
    knowledge_retrieval = "knowledge_retrieval"


@dataclass(slots=True)
class GenerationStep:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: GenerationStepKind = GenerationStepKind.diffusion_sample
    description: str = ""
    deps: list[str] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationPlan:
    """DAG of steps (``deps`` hold step ids); the agent walks ``steps`` in order."""

    goal_id: str
    steps: list[GenerationStep] = field(default_factory=list)

    def add(self, step: GenerationStep) -> GenerationStep:
        self.steps.append(step)
        return step


@dataclass(slots=True)
class StopConditions:
    max_outer_loops: int = 4


def plan_from_prompt(
    prompt: str,
    *,
    use_rag: bool = True,
    expand: bool = True,
    verify: bool = True,
    self_correct: bool = True,
    goal_id: str | None = None,
) -> GenerationPlan:
    """Build a linear UTPC plan for a single-image request."""
    gid = goal_id or uuid.uuid4().hex[:12]
    plan = GenerationPlan(goal_id=gid)
    prev: list[str] = []

    if use_rag:
        s = GenerationStep(
            kind=GenerationStepKind.knowledge_retrieval,
            description="Retrieve local facts (TF-IDF RAG)",
            kwargs={"tool": "rag_retrieve"},
        )
        plan.add(s)
        prev = [s.id]

    if expand:
        s = GenerationStep(
            kind=GenerationStepKind.refine_prompt,
            description="Expand prompt with quality hints",
            deps=list(prev),
            kwargs={"tool": "expand_prompt"},
        )
        plan.add(s)
        prev = [s.id]

    gen = GenerationStep(
        kind=GenerationStepKind.diffusion_sample,
        description="Generate N candidates + composite pick",
        deps=list(prev),
        kwargs={"tool": "generate", "num": 4},
    )
    plan.add(gen)
    prev = [gen.id]

    if verify:
        v = GenerationStep(
            kind=GenerationStepKind.verify_visual,
            description="Score composite / CLIP / sharpness",
            deps=list(prev),
            kwargs={"tool": "verify"},
        )
        plan.add(v)
        prev = [v.id]

    if self_correct:
        plan.add(
            GenerationStep(
                kind=GenerationStepKind.critique_revise,
                description="CLIP-gated refine if verify fails",
                deps=list(prev),
                kwargs={"tool": "self_correct"},
            )
        )

    return plan


def plan_visual_brain(
    prompt: str,
    *,
    has_references: bool = False,
    web_search: bool = True,
    expected_text: str = "",
    use_rag: bool = True,
    expand: bool = True,
    goal_id: str | None = None,
) -> GenerationPlan:
    """
    Understand → dissect → brief → generate → verify → OCR/edit plan.

    Keeps user prompt in scene brief; never drops original intent.
    """
    gid = goal_id or uuid.uuid4().hex[:12]
    plan = GenerationPlan(goal_id=gid)
    prev: list[str] = []

    if use_rag:
        s = GenerationStep(
            kind=GenerationStepKind.knowledge_retrieval,
            description="Local TF-IDF facts",
            kwargs={"tool": "rag_retrieve"},
        )
        plan.add(s)
        prev = [s.id]

    if web_search:
        s = GenerationStep(
            kind=GenerationStepKind.knowledge_retrieval,
            description="Web image search (DuckDuckGo / Wikimedia)",
            deps=list(prev),
            kwargs={"tool": "web_search"},
        )
        plan.add(s)
        prev = [s.id]

    if has_references or web_search:
        s = GenerationStep(
            kind=GenerationStepKind.semantic_segment,
            description="OCR + VLM understand references",
            deps=list(prev),
            kwargs={"tool": "understand_refs"},
        )
        plan.add(s)
        prev = [s.id]

        s = GenerationStep(
            kind=GenerationStepKind.semantic_segment,
            description="Dissect parts for ControlNet / inpaint init",
            deps=list(prev),
            kwargs={"tool": "dissect_refs"},
        )
        plan.add(s)
        prev = [s.id]

        s = GenerationStep(
            kind=GenerationStepKind.refine_prompt,
            description="Synthesize scene brief from user + references",
            deps=list(prev),
            kwargs={"tool": "build_scene_brief"},
        )
        plan.add(s)
        prev = [s.id]
    elif expand:
        s = GenerationStep(
            kind=GenerationStepKind.refine_prompt,
            description="Expand prompt",
            deps=list(prev),
            kwargs={"tool": "expand_prompt"},
        )
        plan.add(s)
        prev = [s.id]

    gen = GenerationStep(
        kind=GenerationStepKind.diffusion_sample,
        description="Generate with understanding-informed args",
        deps=list(prev),
        kwargs={"tool": "generate"},
    )
    plan.add(gen)
    prev = [gen.id]

    v = GenerationStep(
        kind=GenerationStepKind.verify_visual,
        description="Composite / CLIP / sharpness",
        deps=list(prev),
        kwargs={"tool": "verify"},
    )
    plan.add(v)
    prev = [v.id]

    if expected_text:
        plan.add(
            GenerationStep(
                kind=GenerationStepKind.verify_text_ocr,
                description="OCR match vs expected text",
                deps=list(prev),
                kwargs={"tool": "verify_ocr"},
            )
        )
        prev = [plan.steps[-1].id]

    plan.add(
        GenerationStep(
            kind=GenerationStepKind.inpaint_region,
            description="Inpaint / OCR-fix if verify fails",
            deps=list(prev),
            kwargs={"tool": "inpaint_edit"},
        )
    )
    plan.add(
        GenerationStep(
            kind=GenerationStepKind.critique_revise,
            description="Self-correct pass",
            deps=[plan.steps[-1].id],
            kwargs={"tool": "self_correct"},
        )
    )
    return plan


def build_default_plan(
    prompt: str,
    defaults: AgenticStackDefaults | None = None,
    *,
    use_rag: bool = True,
) -> GenerationPlan:
    d = defaults or AgenticStackDefaults()
    return plan_from_prompt(
        prompt,
        use_rag=use_rag,
        expand=d.expand_prompt,
        verify=True,
        self_correct=d.self_correct,
    )


def default_stop_conditions(defaults: AgenticStackDefaults | None = None) -> StopConditions:
    d = defaults or AgenticStackDefaults()
    return StopConditions(max_outer_loops=d.max_reflect_loops)


@dataclass(slots=True)
class VerificationVerdict:
    source: str
    iteration: int
    acceptance: bool
    summary: str = ""
    suggestion: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ReflectionOutcome:
    accepted: bool
    verdict: VerificationVerdict
    prompt_patch: str = ""
    negative_patch: str = ""


def _suggest_from_metrics(metrics: dict[str, float], *, expected_text: str = "") -> tuple[str, str]:
    """Heuristic Think step: map failed metrics → prompt/negative patches."""
    prompt_bits: list[str] = []
    neg_bits: list[str] = []
    comp = float(metrics.get("composite", 0.0) or 0.0)
    clip = float(metrics.get("clip", 0.0) or 0.0)
    sharp = float(metrics.get("sharpness", 0.0) or 0.0)

    if clip < 0.22:
        prompt_bits.append("exact subject match to description, clear focal subject")
    if sharp < 120.0:
        prompt_bits.append(" tack sharp, crisp micro-detail")
        neg_bits.append("blurry, soft focus, motion blur, plastic skin, waxy face")
    if comp < 0.55:
        prompt_bits.append("professional photography, balanced exposure, natural skin texture")
        neg_bits.append("low quality, artifacts, oversaturated, ai generated look, oversmoothed")
    if expected_text and float(metrics.get("ocr_match", 1.0) or 1.0) < 0.7:
        prompt_bits.append(f'legible text reading "{expected_text}"')
        neg_bits.append("misspelled text, garbled typography")

    return ", ".join(prompt_bits), ", ".join(neg_bits)


def reflect_on_result(
    metrics: dict[str, float],
    *,
    iteration: int,
    min_composite: float = 0.62,
    min_clip: float = 0.22,
    expected_text: str = "",
) -> ReflectionOutcome:
    """
    Reflect on verify metrics; return acceptance + corrective patches.
    """
    comp = float(metrics.get("composite", 0.0) or 0.0)
    clip = float(metrics.get("clip", 0.0) or 0.0)
    accepted = comp >= float(min_composite) and clip >= float(min_clip)
    prompt_patch, negative_patch = ("", "")
    suggestion = ""
    if not accepted:
        prompt_patch, negative_patch = _suggest_from_metrics(metrics, expected_text=expected_text)
        suggestion = prompt_patch or "improve adherence and sharpness"

    verdict = VerificationVerdict(
        source="heuristic",
        iteration=int(iteration),
        acceptance=accepted,
        summary=f"composite={comp:.3f} clip={clip:.3f}",
        suggestion=suggestion or None,
        metrics={k: float(v) for k, v in metrics.items()},
    )
    return ReflectionOutcome(
        accepted=accepted,
        verdict=verdict,
        prompt_patch=prompt_patch,
        negative_patch=negative_patch,
    )


def reflect_on_result_llm(
    metrics: dict[str, float],
    *,
    iteration: int,
    prompt: str,
    qwen_path: str,
    device: str = "cuda",
    min_composite: float = 0.62,
    min_clip: float = 0.22,
    expected_text: str = "",
) -> ReflectionOutcome:
    """
    LLM-backed reflect step (Qwen or other HF causal LM). Falls back to heuristics on failure.
    """
    base = reflect_on_result(
        metrics,
        iteration=iteration,
        min_composite=min_composite,
        min_clip=min_clip,
        expected_text=expected_text,
    )
    if base.accepted or not str(qwen_path or "").strip():
        return base
    try:
        tok, model = load_qwen_causal_lm(str(qwen_path), device=device)
        messages = [
            {
                "role": "system",
                "content": (
                    "You improve image-generation prompts after a failed verification. "
                    "Reply with two lines only: PROMPT_PATCH: ... then NEGATIVE_PATCH: ..."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original prompt: {prompt}\n"
                    f"Metrics: {metrics}\n"
                    f"Iteration: {iteration}\n"
                    "Suggest concise prompt and negative patches."
                ),
            },
        ]
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"Prompt: {prompt}\nMetrics: {metrics}"
        inputs = tok(text, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        gen = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        prompt_patch = ""
        negative_patch = ""
        for line in gen.splitlines():
            low = line.strip().lower()
            if low.startswith("prompt_patch:"):
                prompt_patch = line.split(":", 1)[-1].strip()
            elif low.startswith("negative_patch:"):
                negative_patch = line.split(":", 1)[-1].strip()
        if not prompt_patch and not negative_patch:
            return base
        verdict = VerificationVerdict(
            source="llm",
            iteration=int(iteration),
            acceptance=False,
            summary=base.verdict.summary,
            suggestion=prompt_patch or base.verdict.suggestion,
            metrics={k: float(v) for k, v in metrics.items()},
        )
        return ReflectionOutcome(
            accepted=False,
            verdict=verdict,
            prompt_patch=prompt_patch or base.prompt_patch,
            negative_patch=negative_patch or base.negative_patch,
        )
    except Exception:
        return base


def load_qwen_causal_lm(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
):
    """Load a Hugging Face causal LM from local path or hub id."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = torch_dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dt,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda" or getattr(model, "hf_device_map", None) is None:
        model = model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def expand_prompt_qwen(
    tokenizer,
    model,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> str:
    """Return an expanded prompt string (best-effort)."""
    messages = [
        {
            "role": "system",
            "content": "You expand image generation prompts with concrete visual detail. Output only the prompt, no quotes.",
        },
        {"role": "user", "content": f"Expand this prompt for a high-quality image: {prompt}"},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def sample_cmd(
    repo_root: Path,
    *,
    ckpt: str,
    prompt: str,
    out: Path,
    device: str,
    num: int,
    pick_metric: str,
    negative_prompt: str = "",
    expected_text: str = "",
    vit_ckpt: str = "",
    local_rag_jsonl: str = "",
) -> list[str]:
    """``sample.py`` argv shared by the agent tools and the Visual Brain (superior preset, best-of-N)."""
    cmd = [sys.executable, str(repo_root / "sample.py"), "--ckpt", str(ckpt), "--prompt", prompt, "--out", str(out)]
    cmd += ["--device", device, "--num", str(num), "--pick-best", pick_metric, "--preset", "superior"]
    for flag, value in (
        ("--negative-prompt", negative_prompt),
        ("--expected-text", expected_text),
        ("--pick-vit-ckpt", vit_ckpt),
        ("--local-rag-jsonl", local_rag_jsonl),
    ):
        if str(value or "").strip():
            cmd += [flag, str(value)]
    return cmd


class AgentTool(str, Enum):
    rag_retrieve = "rag_retrieve"
    web_search = "web_search"
    understand_refs = "understand_refs"
    dissect_refs = "dissect_refs"
    build_scene_brief = "build_scene_brief"
    expand_prompt = "expand_prompt"
    generate = "generate"
    verify = "verify"
    verify_ocr = "verify_ocr"
    inpaint_edit = "inpaint_edit"
    self_correct = "self_correct"
    reflect = "reflect"
    benchmark = "benchmark"
    flywheel = "flywheel"


@dataclass(slots=True)
class ToolResult:
    tool: AgentTool
    ok: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Execute agent tools against ``AgentContext`` + mutable session state."""

    def __init__(
        self,
        ctx: Any,
        *,
        repo_root: Path,
        num_candidates: int = 4,
        pick_metric: str = "superior_composite",
        expand_prompt: bool = True,
        self_correct: bool = True,
        rag_top_k: int = 8,
        extra_sample_args: list[str] | None = None,
    ) -> None:
        self.ctx = ctx
        self.repo_root = repo_root
        self.num_candidates = max(1, int(num_candidates))
        self.pick_metric = str(pick_metric)
        self.expand_prompt = bool(expand_prompt)
        self.self_correct = bool(self_correct)
        self.rag_top_k = int(rag_top_k)
        self.extra_sample_args = list(extra_sample_args or [])
        self.session: dict[str, Any] = {
            "prompt": ctx.prompt,
            "negative_prompt": ctx.negative_prompt,
            "rag_facts": [],
            "reference_paths": list(getattr(ctx, "reference_images", []) or []),
            "understandings": [],
            "scene_brief": None,
            "last_out": "",
            "metrics": {},
        }

    def execute(self, tool: AgentTool, **kwargs: Any) -> ToolResult:
        fn = {
            AgentTool.rag_retrieve: self._rag_retrieve,
            AgentTool.web_search: self._web_search,
            AgentTool.understand_refs: self._understand_refs,
            AgentTool.dissect_refs: self._dissect_refs,
            AgentTool.build_scene_brief: self._build_scene_brief,
            AgentTool.expand_prompt: self._expand_prompt,
            AgentTool.generate: self._generate,
            AgentTool.verify: self._verify,
            AgentTool.verify_ocr: self._verify_ocr,
            AgentTool.inpaint_edit: self._inpaint_edit,
            AgentTool.self_correct: self._self_correct,
            AgentTool.reflect: self._reflect_stub,
            AgentTool.benchmark: self._benchmark,
            AgentTool.flywheel: self._flywheel,
        }.get(tool)
        if fn is None:
            return ToolResult(tool=tool, ok=False, message=f"unknown tool {tool}")
        return fn(**kwargs)

    def _rag_retrieve(self, **_: Any) -> ToolResult:
        path = str(getattr(self.ctx, "local_rag_jsonl", "") or "").strip()
        if not path:
            return ToolResult(AgentTool.rag_retrieve, ok=True, message="no rag corpus")
        from utils.agentic.quality_loop import build_tfidf_index_from_jsonl

        idx = build_tfidf_index_from_jsonl(path)
        facts = idx.query(self.session["prompt"], top_k=self.rag_top_k)
        self.session["rag_facts"] = facts
        if facts:
            merged = self.session["prompt"] + ". " + " ".join(facts[:3])
            self.session["prompt"] = merged.strip()
        return ToolResult(AgentTool.rag_retrieve, ok=True, data={"facts": facts, "count": len(facts)})

    def _web_search(self, **_: Any) -> ToolResult:
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.web_search, ok=True, message="dry_run")
        allow = bool(getattr(self.ctx, "web_search", True))
        if not allow:
            return ToolResult(AgentTool.web_search, ok=True, message="web search disabled")
        from utils.captioning import download_search_hits, search_reference_images

        work = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "search"
        sr = search_reference_images(self.session["prompt"], max_results=4, allow_web=True)
        hits = download_search_hits(sr.hits, work, max_download=4)
        for h in hits:
            if h.local_path:
                self.session["reference_paths"].append(h.local_path)
        return ToolResult(
            AgentTool.web_search,
            ok=True,
            data={"hits": len(hits), "paths": [h.local_path for h in hits if h.local_path], "notes": sr.notes},
        )

    def _understand_refs(self, **_: Any) -> ToolResult:
        paths = list(self.session.get("reference_paths") or [])
        if not paths:
            return ToolResult(AgentTool.understand_refs, ok=True, message="no references")
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.understand_refs, ok=True, message="dry_run")
        from utils.captioning import understand_images

        work = Path(getattr(self.ctx, "work_dir", "agentic_run"))
        unders = understand_images(
            paths,
            user_prompt=self.session["prompt"],
            work_dir=work,
            device=str(getattr(self.ctx, "device", "cuda")),
        )
        self.session["understandings"] = [u.__dict__ for u in unders]
        facts = []
        for i, u in enumerate(unders):
            if u.caption:
                facts.append(f"Reference {i + 1}: {u.caption[:250]}")
            if u.ocr_text:
                facts.append(f"Reference {i + 1} text: {u.ocr_text[:120]}")
        self.session["rag_facts"].extend(facts)
        return ToolResult(AgentTool.understand_refs, ok=True, data={"count": len(unders), "facts": facts})

    def _dissect_refs(self, **_: Any) -> ToolResult:
        paths = list(self.session.get("reference_paths") or [])
        if not paths:
            return ToolResult(AgentTool.dissect_refs, ok=True, message="no references")
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.dissect_refs, ok=True, message="dry_run")
        from utils.generation.editing import dissect_images_to_parts

        work = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "dissection"
        _reqs, parts, facts = dissect_images_to_parts(
            self.session["prompt"],
            paths,
            output_dir=work,
            enable_heavy_models=True,
        )
        self.session["rag_facts"].extend(facts)
        self.session["dissect_parts"] = len(parts)
        if parts:
            try:
                from utils.generation.editing import build_init_and_inpaint_mask

                spec = build_init_and_inpaint_mask(
                    reference_images=paths,
                    parts=parts,
                    output_dir=work / "composite",
                    target_size=(512, 512),
                )
                self.session["init_image"] = spec.init_image_path
                self.session["inpaint_mask"] = spec.mask_path
            except Exception:
                pass
        return ToolResult(AgentTool.dissect_refs, ok=True, data={"parts": len(parts), "facts": facts})

    def _build_scene_brief(self, **_: Any) -> ToolResult:
        from utils.captioning import ImageUnderstanding

        unders_raw = self.session.get("understandings") or []
        unders = [ImageUnderstanding(**u) if isinstance(u, dict) else u for u in unders_raw]
        brief = synthesize_scene_brief(
            self.session["prompt"],
            understandings=unders,
            dissection_facts=self.session.get("rag_facts") or [],
            expected_text=str(getattr(self.ctx, "expected_text", "") or ""),
            negative_prompt=str(self.session.get("negative_prompt") or ""),
            init_image=str(self.session.get("init_image") or ""),
            inpaint_mask=str(self.session.get("inpaint_mask") or ""),
        )
        self.session["scene_brief"] = brief
        self.session["prompt"] = brief.build_generation_prompt()
        return ToolResult(AgentTool.build_scene_brief, ok=True, data={"elements": len(brief.elements)})

    def _expand_prompt(self, **_: Any) -> ToolResult:
        if not self.expand_prompt:
            return ToolResult(AgentTool.expand_prompt, ok=True, message="skipped")
        from utils.agentic.quality_loop import expand_prompt_heuristic

        before = self.session["prompt"]
        self.session["prompt"] = expand_prompt_heuristic(before)
        return ToolResult(AgentTool.expand_prompt, ok=True, data={"before": before, "after": self.session["prompt"]})

    def _generate(self, *, out_suffix: str = "", extra_args: list[str] | None = None) -> ToolResult:
        out = Path(getattr(self.ctx, "out", "agent_out.png"))
        if out_suffix:
            out = out.with_name(out.stem + out_suffix + out.suffix)
        work = Path(getattr(self.ctx, "work_dir", "agentic_run"))
        if not out.is_absolute():
            out = work / out
        if bool(getattr(self.ctx, "dry_run", False)):
            self.session["last_out"] = str(out)
            return ToolResult(AgentTool.generate, ok=True, message="dry_run", data={"out": str(out)})
        cmd = sample_cmd(
            self.repo_root,
            ckpt=str(self.ctx.ckpt),
            prompt=self.session["prompt"],
            out=out,
            device=str(getattr(self.ctx, "device", "cuda")),
            num=self.num_candidates,
            pick_metric=self.pick_metric,
            negative_prompt=str(self.session.get("negative_prompt") or ""),
            expected_text=str(getattr(self.ctx, "expected_text", "") or ""),
            vit_ckpt=str(getattr(self.ctx, "vit_ckpt", "") or ""),
            local_rag_jsonl=str(getattr(self.ctx, "local_rag_jsonl", "") or ""),
        )
        if self.self_correct:
            cmd.append("--superior-self-correct")
        if self.expand_prompt:
            cmd.append("--expand-prompt")
        refs = self.session.get("reference_paths") or []
        if refs:
            cmd.extend(["--dissect-refs", ",".join(str(r) for r in refs)])
            cmd.append("--creative-rag")
        init_i = str(self.session.get("init_image") or "").strip()
        mask_i = str(self.session.get("inpaint_mask") or "").strip()
        if init_i and mask_i:
            cmd.extend(["--init-image", init_i, "--mask", mask_i, "--auto-init-from-dissection"])
        cmd.extend(self.extra_sample_args)
        if extra_args:
            cmd.extend(extra_args)
        rc = subprocess.run(cmd, cwd=str(self.repo_root), check=False).returncode
        ok = rc == 0 and out.is_file()
        self.session["last_out"] = str(out)
        return ToolResult(AgentTool.generate, ok=ok, message=f"exit={rc}", data={"out": str(out), "cmd": cmd})

    def _verify(self, **_: Any) -> ToolResult:
        if bool(getattr(self.ctx, "dry_run", False)):
            metrics = {"composite": 0.85, "clip": 0.35, "sharpness": 200.0}
            self.session["metrics"] = metrics
            return ToolResult(AgentTool.verify, ok=True, data=metrics)
        out = self.session.get("last_out", "")
        if not out or not Path(out).is_file():
            return ToolResult(AgentTool.verify, ok=False, message="no image to verify")
        from utils.agentic.quality_loop import verify_metrics

        metrics = verify_metrics(
            np.array(Image.open(out).convert("RGB"), dtype=np.uint8),
            self.session["prompt"],
            device=str(self.ctx.device),
            vit_ckpt=str(getattr(self.ctx, "vit_ckpt", "") or "").strip(),
            expected_text=str(getattr(self.ctx, "expected_text", "") or "").strip(),
        )
        self.session["metrics"] = metrics
        return ToolResult(AgentTool.verify, ok=True, data=metrics)

    def _verify_ocr(self, **_: Any) -> ToolResult:
        v = self._verify()
        exp = str(getattr(self.ctx, "expected_text", "") or "").strip()
        if not exp:
            return ToolResult(AgentTool.verify_ocr, ok=True, message="no expected text", data=v.data)
        ok = float(v.data.get("ocr_match", 0.0) or 0.0) >= 0.65
        return ToolResult(AgentTool.verify_ocr, ok=ok, data=v.data)

    def _inpaint_edit(self, **kwargs: Any) -> ToolResult:
        """Regional edit via sample.py img2img/inpaint."""
        out = self.session.get("last_out", "")
        if not out or not Path(out).is_file():
            return ToolResult(AgentTool.inpaint_edit, ok=False, message="no image")
        region = str(kwargs.get("region", "full") or "full")
        extra = ["--init-image", str(out), "--strength", str(kwargs.get("strength", 0.4))]
        if region != "full":
            try:
                from utils.generation.editing import build_segmentation_mask_for_edit

                seg = build_segmentation_mask_for_edit(out, region)
                mask_path = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "edit_mask.png"
                seg.mask.save(mask_path)
                extra.extend(["--mask", str(mask_path)])
            except Exception:
                pass
        return self._generate(out_suffix="_inpaint", extra_args=extra)

    def _self_correct(self, **_: Any) -> ToolResult:
        return ToolResult(AgentTool.self_correct, ok=True, message="handled in generate via --superior-self-correct")

    def _reflect_stub(self, **kwargs: Any) -> ToolResult:
        suggestion = kwargs.get("suggestion", "")
        if suggestion:
            self.session["prompt"] = (self.session["prompt"] + ", " + suggestion).strip(", ")
        return ToolResult(AgentTool.reflect, ok=True, data={"prompt": self.session["prompt"]})

    def _benchmark(self, **_: Any) -> ToolResult:
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.benchmark, ok=True, message="dry_run")
        work = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "benchmark"
        argv = [
            "--ckpt",
            str(self.ctx.ckpt),
            "--out-dir",
            str(work),
            "--preset",
            "superior",
            "--num",
            str(self.num_candidates),
            "--pick-best",
            self.pick_metric,
        ]
        from utils.agentic.quality_loop import run_tool

        rc = run_tool("benchmark_suite", argv, repo_root=self.repo_root)
        return ToolResult(AgentTool.benchmark, ok=rc == 0, data={"out_dir": str(work)})

    def _flywheel(self, **_: Any) -> ToolResult:
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.flywheel, ok=True, message="dry_run")
        work = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "flywheel"
        argv = ["--base-ckpt", str(self.ctx.ckpt), "--work-dir", str(work)]
        rag = str(getattr(self.ctx, "local_rag_jsonl", "") or "").strip()
        if rag:
            argv.extend(["--local-rag-jsonl", rag])
        vit = str(getattr(self.ctx, "vit_ckpt", "") or "").strip()
        if vit:
            argv.extend(["--vit-ckpt", vit])
        from utils.agentic.quality_loop import run_tool

        rc = run_tool("run_flywheel", argv, repo_root=self.repo_root)
        return ToolResult(AgentTool.flywheel, ok=rc == 0, data={"work_dir": str(work)})


@dataclass(slots=True)
class RoleStageResult:
    role: str
    ok: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolePipelineResult:
    stages: list[RoleStageResult] = field(default_factory=list)
    out_path: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    prompt_final: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.stages) and all(s.ok for s in self.stages)


class RolePipeline:
    """
    Run Reasoner → Designer → Verifier in order (no reflect loop).

    Use ``ImageGenerationAgent`` when you need Act–Reflect–Think–Act retries.
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        self.registry = registry

    @classmethod
    def from_context(
        cls,
        ctx: AgentContext,
        *,
        repo_root: Path,
        **registry_kw: Any,
    ) -> RolePipeline:
        reg = ToolRegistry(ctx, repo_root=repo_root, **registry_kw)
        return cls(reg)

    def run_reasoner(self) -> RoleStageResult:
        r1 = self.registry.execute(AgentTool.rag_retrieve)
        r2 = self.registry.execute(AgentTool.expand_prompt)
        ok = r1.ok and r2.ok
        return RoleStageResult(
            role="reasoner",
            ok=ok,
            message=f"rag={r1.message} expand={r2.message}",
            data={"prompt": self.registry.session["prompt"], "facts": r1.data.get("facts", [])},
        )

    def run_designer(self, *, out_suffix: str = "") -> RoleStageResult:
        r = self.registry.execute(AgentTool.generate, out_suffix=out_suffix)
        return RoleStageResult(
            role="designer",
            ok=r.ok,
            message=r.message,
            data={"out": r.data.get("out", ""), "cmd": r.data.get("cmd", [])},
        )

    def run_verifier(self) -> RoleStageResult:
        r = self.registry.execute(AgentTool.verify)
        return RoleStageResult(
            role="verifier",
            ok=r.ok,
            message=r.message,
            data=dict(r.data),
        )

    def run(self, *, out_suffix: str = "") -> RolePipelineResult:
        stages = [
            self.run_reasoner(),
            self.run_designer(out_suffix=out_suffix),
            self.run_verifier(),
        ]
        return RolePipelineResult(
            stages=stages,
            out_path=str(self.registry.session.get("last_out", "")),
            metrics=dict(self.registry.session.get("metrics", {})),
            prompt_final=str(self.registry.session.get("prompt", "")),
        )


@dataclass(slots=True)
class AgentRunResult:
    accepted: bool
    out_path: str
    trace: AgentTrace
    experience: TrajectoryExperience | None = None


class ImageGenerationAgent:
    """
    Autonomous agent: plan → tools → verify → reflect → retry.

    Supports optional experience memory (GenEvolve-style trajectory distillation).
    """

    def __init__(
        self,
        defaults: AgenticStackDefaults | None = None,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self.defaults = defaults or AgenticStackDefaults()
        self.repo_root = Path(repo_root or _sdx_repo_root())

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
        trajectories: list[TrajectoryRecord] = []

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
        tool_seq: list[str] = []

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

    def run_evolve(self, ctx: AgentContext, *, variants: int | None = None) -> AgentRunResult:
        """
        Run multiple tool trajectories (different expand/rag toggles), distill experience.
        """
        n = max(2, int(variants or self.defaults.trajectory_variants))
        work = Path(ctx.work_dir)
        all_traj: list[TrajectoryRecord] = []
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


@dataclass(slots=True)
class SceneElement:
    """One thing that must appear (or be preserved) in the output."""

    name: str
    source: str = "user_prompt"
    description: str = ""
    reference_index: int = -1
    must_include: bool = True
    ocr_text: str = ""


@dataclass(slots=True)
class SceneBrief:
    """What the Visual Brain decided belongs in the final image."""

    user_prompt: str
    negative_prompt: str = ""
    elements: list[SceneElement] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    reference_paths: list[str] = field(default_factory=list)
    control_image: str = ""
    init_image: str = ""
    inpaint_mask: str = ""
    expected_text: str = ""
    enriched_prompt: str = ""
    reasoning: str = ""

    def to_facts(self) -> list[str]:
        """Flatten brief into RAG-style fact strings."""
        facts = list(self.facts)
        facts.append(f"Primary user request (must satisfy): {self.user_prompt.strip()}")
        for el in self.elements:
            if el.must_include:
                facts.append(f"Include {el.name}: {el.description or el.source}".strip())
            if el.ocr_text:
                facts.append(f"Legible text from reference: {el.ocr_text[:120]}")
        if self.expected_text:
            facts.append(f"Rendered text must read: {self.expected_text}")
        return facts

    def build_generation_prompt(self) -> str:
        """Merge user prompt + facts for T5 encoding."""
        if self.enriched_prompt.strip():
            base = self.enriched_prompt.strip()
        else:
            base = self.user_prompt.strip()
        merged = merge_facts_into_prompt(base, self.to_facts())
        return merged.strip()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SceneBrief:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        els = [SceneElement(**e) for e in data.pop("elements", [])]
        return cls(elements=els, **data)


def synthesize_scene_brief(
    user_prompt: str,
    *,
    understandings: Sequence[ImageUnderstanding],
    dissection_facts: Sequence[str] | None = None,
    rag_facts: Sequence[str] | None = None,
    expected_text: str = "",
    negative_prompt: str = "",
    init_image: str = "",
    inpaint_mask: str = "",
    control_image: str = "",
    creative_enriched: str = "",
    creative_reasoning: str = "",
) -> SceneBrief:
    """
    Merge user prompt + reference understanding into a scene brief.
    """
    refs = [u.path for u in understandings if u.path]
    facts: list[str] = list(rag_facts or []) + list(dissection_facts or [])
    elements: list[SceneElement] = [
        SceneElement(name="user_subject", source="user_prompt", description=user_prompt.strip(), must_include=True)
    ]

    for i, u in enumerate(understandings):
        desc = u.caption or f"reference {i + 1}"
        elements.append(
            SceneElement(
                name=f"ref_{i + 1}",
                source=u.source,
                description=desc[:400],
                reference_index=i,
                must_include=False,
                ocr_text=u.ocr_text[:200] if u.ocr_text else "",
            )
        )
        if u.caption:
            facts.append(f"Reference image {i + 1} shows: {u.caption[:300]}")
        if u.ocr_text:
            facts.append(f"Reference image {i + 1} text: {u.ocr_text[:200]}")

    ctrl = control_image
    if not ctrl:
        for u in understandings:
            if u.control_maps.get("canny"):
                ctrl = u.control_maps["canny"]
                break

    return SceneBrief(
        user_prompt=user_prompt.strip(),
        negative_prompt=negative_prompt.strip(),
        elements=elements,
        facts=facts,
        reference_paths=refs,
        control_image=ctrl,
        init_image=init_image,
        inpaint_mask=inpaint_mask,
        expected_text=str(expected_text or "").strip(),
        enriched_prompt=creative_enriched.strip(),
        reasoning=creative_reasoning.strip(),
    )


def prompt_coverage_score(brief: SceneBrief, metrics: dict[str, float]) -> float:
    """
    Heuristic: how well verify metrics + OCR align with user intent.
    """
    comp = float(metrics.get("composite", 0.0) or 0.0)
    clip = float(metrics.get("clip", 0.0) or 0.0)
    ocr = float(metrics.get("ocr_match", 1.0) or 1.0)
    base = 0.45 * comp + 0.35 * min(1.0, max(0.0, (clip + 0.2) / 0.6)) + 0.20 * ocr
    if brief.expected_text and ocr < 0.65:
        base *= 0.85
    return float(max(0.0, min(1.0, base)))


@dataclass
class MemoryBox:
    session_id: str
    prompt: str = ""
    facets: dict[str, Any] = field(default_factory=dict)
    clarify: dict[str, Any] = field(default_factory=dict)
    rag_notes: list[str] = field(default_factory=list)
    refs: dict[str, list[str]] = field(default_factory=dict)
    style_mode: str = "instantstyle"
    lora_path: str = ""
    character_sheet: str = ""
    last_metrics: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=lambda: time.time())

    def remember(self, event: str, **payload: Any) -> None:
        self.history.append({"t": time.time(), "event": event, **payload})
        self.updated_at = time.time()
        if len(self.history) > 200:
            self.history = self.history[-200:]

    def set_refs(self, facet: str, paths: list[str]) -> None:
        self.refs[str(facet)] = [str(p) for p in paths]
        self.remember("refs", facet=facet, n=len(paths))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryBox:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


def load_memory_box(path: str | Path) -> MemoryBox | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return MemoryBox.from_dict(data)
    except Exception:
        return None


def save_memory_box(box: MemoryBox, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    box.updated_at = time.time()
    p.write_text(json.dumps(box.to_dict(), indent=2), encoding="utf-8")
    return p


@dataclass(slots=True)
class VisualBrainConfig:
    web_search: bool = True
    max_search_images: int = 4
    understand_ocr: bool = True
    understand_vlm: bool = True
    extract_control: bool = True
    dissect_heavy: bool = True
    creative_rag: bool = True
    creative_rag_level: float = 0.65
    max_edit_loops: int = 3
    min_coverage: float = 0.62
    min_clip: float = 0.22
    num_candidates: int = 4
    pick_metric: str = "superior_composite"
    self_correct: bool = True


@dataclass(slots=True)
class VisualBrainResult:
    accepted: bool
    out_path: str
    brief: SceneBrief
    reference_paths: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    trace_path: str = ""


class VisualBrain:
    """Coordinator for visual understanding + iterative generation."""

    def __init__(
        self,
        *,
        config: VisualBrainConfig | None = None,
        defaults: AgenticStackDefaults | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.config = config or VisualBrainConfig()
        self.defaults = defaults or AgenticStackDefaults()
        self.repo_root = Path(repo_root or _sdx_repo_root())

    def run(
        self,
        *,
        ckpt: str,
        prompt: str,
        work_dir: str = "visual_brain_run",
        out: str = "brain_out.png",
        device: str = "cuda",
        negative_prompt: str = "",
        reference_images: Sequence[str] | None = None,
        local_rag_jsonl: str = "",
        expected_text: str = "",
        vit_ckpt: str = "",
        dry_run: bool = False,
    ) -> VisualBrainResult:
        cfg = self.config
        work = Path(work_dir)
        if not dry_run:
            work.mkdir(parents=True, exist_ok=True)

        user_prompt = str(prompt).strip()
        trace: dict[str, Any] = {"goal": user_prompt, "steps": []}

        # Gather references (user + web search)
        ref_paths: list[str] = [str(p) for p in (reference_images or []) if str(p).strip()]
        search_dir = work / "search"
        if cfg.web_search and user_prompt and not dry_run:
            search_q = user_prompt[:120]
            sr = search_reference_images(search_q, max_results=cfg.max_search_images, allow_web=True)
            trace["steps"].append({"search": {"query": search_q, "hits": len(sr.hits), "notes": sr.notes}})
            if sr.hits:
                downloaded = download_search_hits(sr.hits, search_dir, max_download=cfg.max_search_images)
                for h in downloaded:
                    if h.local_path:
                        ref_paths.append(h.local_path)

        # Understand all references
        understandings = []
        if ref_paths and not dry_run:
            understandings = understand_images(
                ref_paths,
                user_prompt=user_prompt,
                work_dir=work,
                sources=["user" if i < len(reference_images or []) else "web" for i in range(len(ref_paths))],
                run_ocr=cfg.understand_ocr,
                run_vlm=cfg.understand_vlm,
                run_control=cfg.extract_control,
                device=device,
            )
            trace["steps"].append(
                {
                    "understand": [
                        {"path": u.path, "caption": u.caption[:200], "ocr": u.ocr_text[:80]} for u in understandings
                    ]
                }
            )

        # Dissect parts from prompt
        dissection_facts: list[str] = []
        init_image = ""
        inpaint_mask = ""
        dissect_refs_arg = ""
        if ref_paths:
            dissect_refs_arg = ",".join(ref_paths)
            try:
                from utils.generation.editing import dissect_images_to_parts

                _reqs, _parts, dissection_facts = dissect_images_to_parts(
                    user_prompt,
                    ref_paths,
                    output_dir=work / "dissection",
                    enable_heavy_models=cfg.dissect_heavy,
                )
                if _parts and not dry_run:
                    from utils.generation.editing import build_init_and_inpaint_mask

                    spec = build_init_and_inpaint_mask(
                        reference_images=ref_paths,
                        parts=_parts,
                        output_dir=work / "composite",
                        target_size=(512, 512),
                    )
                    init_image = spec.init_image_path
                    inpaint_mask = spec.mask_path
                    trace["steps"].append({"dissect": {"parts": len(_parts), "init": init_image}})
            except Exception as exc:
                trace["steps"].append({"dissect_error": str(exc)})

        # Local RAG facts
        rag_facts: list[str] = []
        if local_rag_jsonl and Path(local_rag_jsonl).is_file():
            try:
                from utils.agentic.quality_loop import build_tfidf_index_from_jsonl

                idx = build_tfidf_index_from_jsonl(local_rag_jsonl)
                rag_facts = idx.query(user_prompt, top_k=self.defaults.rag_top_k)
            except Exception:
                pass

        # Creative RAG enrichment
        creative_enriched = ""
        creative_reasoning = ""
        creative_neg = ""
        if cfg.creative_rag and not dry_run:
            try:
                from utils.prompt.rag import CreativeRAGEngine

                engine = CreativeRAGEngine()
                ref_for_creative = ref_paths[0] if ref_paths else None
                cr = engine.enrich(
                    prompt=user_prompt,
                    reference_image_path=ref_for_creative,
                    facts=rag_facts + dissection_facts,
                    creativity_level=float(cfg.creative_rag_level),
                )
                creative_enriched = cr.enriched_prompt
                creative_reasoning = cr.reasoning
                creative_neg = cr.negative_additions
                trace["steps"].append({"creative_rag": {"novelty": cr.novelty_score, "fallback": cr.fallback_used}})
            except Exception as exc:
                trace["steps"].append({"creative_rag_error": str(exc)})

        neg_eff = ", ".join(x for x in (negative_prompt, creative_neg) if x).strip(", ")

        brief = synthesize_scene_brief(
            user_prompt,
            understandings=understandings,
            dissection_facts=dissection_facts,
            rag_facts=rag_facts,
            expected_text=expected_text,
            negative_prompt=neg_eff,
            init_image=init_image,
            inpaint_mask=inpaint_mask,
            creative_enriched=creative_enriched,
            creative_reasoning=creative_reasoning,
        )
        brief.save(work / "scene_brief.json")

        if dry_run:
            return VisualBrainResult(
                accepted=True,
                out_path=str(work / out),
                brief=brief,
                reference_paths=ref_paths,
                metrics={"composite": 0.85, "clip": 0.35, "coverage": 0.9},
                trace_path=str(work / "brain_trace.json"),
            )

        # Generate + verify + edit loop
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = work / out_path
        gen_prompt = brief.build_generation_prompt()
        metrics: dict[str, float] = {}
        accepted = False

        def base_cmd(out_file: Path) -> list[str]:
            return sample_cmd(
                self.repo_root,
                ckpt=ckpt,
                prompt=gen_prompt,
                out=out_file,
                device=device,
                num=cfg.num_candidates,
                pick_metric=cfg.pick_metric,
                negative_prompt=brief.negative_prompt,
                expected_text=expected_text,
                vit_ckpt=vit_ckpt,
                local_rag_jsonl=local_rag_jsonl,
            )

        for loop in range(max(1, cfg.max_edit_loops)):
            extra = self._sample_args_from_brief(
                brief,
                dissect_refs=dissect_refs_arg,
                init_image=init_image,
                inpaint_mask=inpaint_mask,
                loop=loop,
            )
            cand = out_path if loop == 0 else out_path.with_name(f"{out_path.stem}_edit{loop}{out_path.suffix}")
            rc = subprocess.run([*base_cmd(cand), *extra], cwd=str(self.repo_root), check=False).returncode
            if rc != 0 or not cand.is_file():
                trace["steps"].append({"generate_fail": {"loop": loop, "rc": rc}})
                continue

            metrics = self._verify_image(
                cand, gen_prompt, expected_text=expected_text, device=device, vit_ckpt=vit_ckpt
            )
            cov = prompt_coverage_score(brief, metrics)
            metrics["coverage"] = cov
            trace["steps"].append({"loop": loop, "metrics": metrics, "out": str(cand)})

            if cov >= cfg.min_coverage and metrics.get("clip", 0.0) >= cfg.min_clip:
                accepted = True
                out_path = cand
                break

            # Edit pass: OCR fix or inpaint refine
            if expected_text and float(metrics.get("ocr_match", 0.0) or 0.0) < 0.65:
                fix_path = cand.with_name(f"{cand.stem}_ocrfix{loop}{cand.suffix}")
                fix = ["--ocr-fix", "--init-image", str(cand), "--strength", "0.35"]
                subprocess.run([*base_cmd(fix_path), *extra, *fix], cwd=str(self.repo_root), check=False)
                if fix_path.is_file():
                    cand = fix_path
                    metrics = self._verify_image(
                        cand, gen_prompt, expected_text=expected_text, device=device, vit_ckpt=vit_ckpt
                    )
                    metrics["coverage"] = prompt_coverage_score(brief, metrics)
                    if metrics["coverage"] >= cfg.min_coverage:
                        accepted = True
                        out_path = cand
                        break

            # Prompt patch from low CLIP — re-anchor to user prompt
            if float(metrics.get("clip", 0.0) or 0.0) < cfg.min_clip:
                gen_prompt = f"{user_prompt}. {gen_prompt}"[:2000]

        trace_path = work / "brain_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

        return VisualBrainResult(
            accepted=accepted,
            out_path=str(out_path),
            brief=brief,
            reference_paths=ref_paths,
            metrics=metrics,
            iterations=int(trace.get("steps", [{}])[-1].get("loop", 0) + 1 if trace["steps"] else 1),
            trace_path=str(trace_path),
        )

    def _sample_args_from_brief(
        self,
        brief: SceneBrief,
        *,
        dissect_refs: str,
        init_image: str,
        inpaint_mask: str,
        loop: int,
    ) -> list[str]:
        cfg = self.config
        args: list[str] = list(self.defaults.extra_sample_args)
        if cfg.creative_rag:
            args.append("--creative-rag")
            args.extend(["--creative-rag-level", str(cfg.creative_rag_level)])
        if dissect_refs:
            args.extend(["--dissect-refs", dissect_refs])
        if init_image and inpaint_mask and loop == 0:
            args.extend(
                [
                    "--init-image",
                    init_image,
                    "--mask",
                    inpaint_mask,
                    "--auto-init-from-dissection",
                ]
            )
        elif brief.control_image:
            args.extend(["--control-image", brief.control_image, "--control-type", "canny"])
        if cfg.self_correct:
            args.append("--superior-self-correct")
        args.extend(["--human-made", "standard"])
        return args

    def _verify_image(
        self,
        path: Path,
        prompt: str,
        *,
        expected_text: str,
        device: str,
        vit_ckpt: str,
    ) -> dict[str, float]:
        from utils.agentic.quality_loop import verify_metrics

        arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return verify_metrics(arr, prompt, device=device, vit_ckpt=vit_ckpt or "", expected_text=expected_text)
