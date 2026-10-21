"""Invention stack wiring for ``sample.py``: prompt rewrite, adaptive steps, box layout, self-healing plan.

Also hosts ``--cast-json`` (multi-character cast compiler) and ``--corpus-jsonl`` (training-corpus retrieval),
which rewrite the prompt before the box layout and character sheets they may supply are read.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.generation.inventions.failure_oracle import diagnose_failures
from utils.generation.inventions.stack import apply_invention_stack

__all__ = [
    "InventionRuntime",
    "SelfHealingPlan",
    "adaptive_nfe",
    "apply_cast_and_corpus_to_namespace",
    "apply_invention_to_namespace",
    "plan_self_healing",
    "prepare_invention_runtime",
    "write_self_healing_plan",
]

_OFF = ("off", "none", "0", "false", "")


@dataclass
class InventionRuntime:
    positive: str
    negative: str
    enable: str = "off"
    box_layout: dict[str, Any] | None = None
    box_layout_path: str = ""
    steps: int | None = None
    reports: dict[str, Any] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfHealingPlan:
    """Oracle -> stack -> APEX sampling -> critique-region redo, as a ``sample.py`` argv plus redo hints."""

    prompt: str
    negative: str = ""
    steps: int = 28
    sample_argv: list[str] = field(default_factory=list)
    critique_hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_json(obj: Any, work_dir: str | Path, name: str) -> str:
    """Write *obj* to ``work_dir/name`` (or ``$TMP/sdx_name`` without a work dir); return the path."""
    path = Path(work_dir) / name if work_dir else Path(tempfile.gettempdir()) / f"sdx_{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return str(path)


def adaptive_nfe(prompt: str, *, base_steps: int = 28, min_steps: int = 12, max_steps: int = 40) -> int:
    """Step budget from oracle risk: +3 steps per elevated risk, -2 for a risk-free prompt."""
    risks = diagnose_failures(prompt).risks
    steps = base_steps + 3 * len(risks) - (0 if risks else 2)
    return int(max(min_steps, min(max_steps, steps)))


def _redo_regions(risks: list[str]) -> list[str]:
    regions: list[str] = []
    for r in risks:
        if r == "anatomy":
            regions.extend(["hands", "face"])
        elif r == "glyph":
            regions.append("subject")
        elif r == "plastic":
            regions.append("face")
    return list(dict.fromkeys(regions))


def plan_self_healing(prompt: str, negative: str = "") -> SelfHealingPlan:
    inv = apply_invention_stack(prompt, negative, enable="auto")
    steps = adaptive_nfe(inv.positive)
    risks = (inv.reports.get("oracle") or {}).get("risks") or []
    argv = [
        "--prompt",
        inv.positive,
        "--negative-prompt",
        inv.negative or negative,
        "--steps",
        str(steps),
        "--scheduler",
        "apex",
        "--solver",
        "apex_adaptive",
        "--invention-stack",
        "off",  # already applied
    ]
    return SelfHealingPlan(
        prompt=inv.positive,
        negative=inv.negative,
        steps=steps,
        sample_argv=argv,
        critique_hints=_redo_regions(risks),
        notes=[f"risks={risks}"],
    )


def write_self_healing_plan(plan: SelfHealingPlan, work_dir: str | Path) -> str:
    return _write_json(plan.to_dict(), work_dir, "self_healing_plan.json")


def prepare_invention_runtime(
    prompt: str,
    negative: str = "",
    *,
    enable: str = "auto",
    base_steps: int | None = None,
    adaptive_steps: bool = False,
    work_dir: str | Path = "",
) -> InventionRuntime:
    """Run the invention stack and compute optional runtime knobs."""
    mode = str(enable or "off").strip().lower()
    pos, neg = str(prompt or ""), str(negative or "")
    if mode in _OFF:
        return InventionRuntime(positive=pos, negative=neg, enable="off")

    res = apply_invention_stack(pos, neg, enable=mode, work_dir=work_dir)
    notes: list[str] = []
    steps: int | None = None
    if adaptive_steps and base_steps is not None:
        steps = adaptive_nfe(res.positive, base_steps=int(base_steps))
        notes.append(f"adaptive_nfe={steps}")
    return InventionRuntime(
        positive=res.positive,
        negative=res.negative,
        enable=mode,
        box_layout=res.box_layout,
        box_layout_path=_write_json(res.box_layout, work_dir, "invention_box_layout.json") if res.box_layout else "",
        steps=steps,
        reports=dict(res.reports),
        repairs=list(res.repairs),
        notes=notes,
    )


def apply_invention_to_namespace(args: Any, *, work_dir: str = "outputs/_invention") -> InventionRuntime | None:
    """Mutate a ``sample.py`` namespace in place for ``--invention-stack``; None when it is off."""
    enable = str(getattr(args, "invention_stack", "off") or "off").strip().lower()
    if enable in _OFF:
        return None

    prompt = str(getattr(args, "prompt", "") or "")
    negative = str(getattr(args, "negative_prompt", "") or "")
    adaptive = bool(getattr(args, "invention_adaptive_steps", False))
    rt = prepare_invention_runtime(
        prompt,
        negative,
        enable=enable,
        base_steps=int(getattr(args, "steps", 28) or 28),
        adaptive_steps=adaptive,
        work_dir=work_dir,
    )
    args.prompt = rt.positive
    if rt.negative:
        args.negative_prompt = rt.negative
    if rt.box_layout_path and not str(getattr(args, "box_layout", "") or "").strip():
        args.box_layout = rt.box_layout_path
        args.anti_bleed = True
    if rt.steps is not None:
        args.steps = int(rt.steps)

    # Guidance knobs for the risks the oracle found; only fills values the user left unset.
    risks = set((rt.reports.get("oracle") or {}).get("risks") or [])
    if (
        risks & {"binding", "count", "negation", "multi_char"}
        and not str(getattr(args, "guidance_schedule", None) or "").strip()
        and not getattr(args, "holy_grail", False)  # gaussian_diffusion rejects a schedule with Holy Grail
    ):
        args.guidance_schedule = "piecewise"
    if risks & {"plastic", "binding"} and float(getattr(args, "apg_parallel_eta", -1.0)) < 0:
        args.apg_parallel_eta = 0.0

    if bool(getattr(args, "invention_self_heal", False)):
        rt.notes.append(f"self_heal_plan={write_self_healing_plan(plan_self_healing(prompt, negative), work_dir)}")
    return rt


def apply_cast_and_corpus_to_namespace(args: Any, *, work_dir: str | Path = "") -> None:
    """``--cast-json`` (ownership phrases, anti-blend negatives, box layout, sheets) and ``--corpus-jsonl``."""
    cast_json = str(getattr(args, "cast_json", "") or "").strip()
    if cast_json:
        from utils.generation.multi_char_cast import compile_cast_scene, load_cast_scene

        cast = compile_cast_scene(load_cast_scene(cast_json), base_prompt=str(getattr(args, "prompt", "") or ""))
        args.prompt = cast.positive
        args.negative_prompt = ", ".join(n for n in (getattr(args, "negative_prompt", ""), cast.negative) if n)
        if cast.box_layout and not str(getattr(args, "box_layout", "") or "").strip():
            args.box_layout = _write_json(cast.box_layout, work_dir, "cast_box_layout.json")
            args.anti_bleed = True
        if cast.sheet_paths and not str(getattr(args, "character_sheet", "") or "").strip():
            args.character_sheet = ",".join(cast.sheet_paths)
            args.label_multi_character_sheets = True

    corpus = str(getattr(args, "corpus_jsonl", "") or "").strip()
    if corpus:
        from utils.generation.corpus_retrieve import hits_to_moodboard_payload, load_corpus_index, retrieve_corpus_refs
        from utils.prompt.rag_prompt import merge_facts_into_prompt

        index = load_corpus_index(corpus, root=str(getattr(args, "corpus_root", "") or "") or None)
        hits = retrieve_corpus_refs(
            index,
            str(getattr(args, "prompt", "") or ""),
            top_k=max(1, int(getattr(args, "corpus_top_k", 4) or 4)),
            source_filter=str(getattr(args, "corpus_source", "") or ""),
        )
        facts = [h.caption for h in hits if h.caption]
        if facts:
            args.prompt = merge_facts_into_prompt(str(getattr(args, "prompt", "") or ""), facts)
        images = hits_to_moodboard_payload(hits)["images"]
        if images and not str(getattr(args, "moodboard_images", "") or "").strip():
            args.moodboard_images = ",".join(images)
