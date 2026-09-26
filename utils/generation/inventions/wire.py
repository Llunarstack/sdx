"""Central Invention Lab wiring for sample.py / pipelines / library APIs.

Keeps prompt-stack, adaptive NFE, SPECTRA, and box-layout side effects in one place
so CLI and library callers stay consistent.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "InventionRuntime",
    "prepare_invention_runtime",
    "apply_invention_to_namespace",
    "enrich_prompts_with_inventions",
    "collect_invention_sample_argv",
    "fold_invention_sample_argv",
]


@dataclass
class InventionRuntime:
    positive: str
    negative: str
    enable: str = "off"
    box_layout: dict[str, Any] | None = None
    box_layout_path: str = ""
    steps: int | None = None
    invention_spectra: bool = False
    reports: dict[str, Any] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_invention_runtime(
    prompt: str,
    negative: str = "",
    *,
    enable: str = "auto",
    base_steps: int | None = None,
    adaptive_steps: bool = False,
    auto_spectra: bool = False,
    invention_spectra: bool = False,
    work_dir: str | Path = "",
) -> InventionRuntime:
    """Run the invention stack and compute optional runtime knobs."""
    mode = str(enable or "off").strip().lower()
    pos, neg = str(prompt or ""), str(negative or "")
    if mode in ("off", "none", "0", "false", ""):
        return InventionRuntime(
            positive=pos,
            negative=neg,
            enable="off",
            invention_spectra=bool(invention_spectra),
        )

    from utils.generation.inventions.stack import apply_invention_stack
    from utils.generation.inventions.systems_ext import adaptive_nfe

    wd = work_dir or ""
    res = apply_invention_stack(pos, neg, enable=mode, work_dir=wd)
    notes: list[str] = []
    steps: int | None = None
    if adaptive_steps and base_steps is not None:
        steps = adaptive_nfe(res.positive, base_steps=int(base_steps))
        notes.append(f"adaptive_nfe={steps}")

    spectra = bool(invention_spectra)
    if auto_spectra and not spectra:
        risks = (res.reports.get("oracle") or {}).get("risks") or []
        # Structure-heavy prompts benefit from SPECTRA CFG phasing.
        if any(r in ("binding", "count", "multi_char", "negation") for r in risks):
            spectra = True
            notes.append("auto_spectra=True")

    box_path = ""
    if res.box_layout:
        if wd:
            p = Path(wd) / "invention_box_layout.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(res.box_layout, indent=2), encoding="utf-8")
            box_path = str(p)
        else:
            p = Path(tempfile.gettempdir()) / "sdx_invention_box_layout.json"
            p.write_text(json.dumps(res.box_layout), encoding="utf-8")
            box_path = str(p)

    return InventionRuntime(
        positive=res.positive,
        negative=res.negative,
        enable=mode,
        box_layout=res.box_layout,
        box_layout_path=box_path,
        steps=steps,
        invention_spectra=spectra,
        reports=dict(res.reports),
        repairs=list(res.repairs),
        notes=notes,
    )


def _argv_chunks_from_obj(obj: Any) -> list[list[str]]:
    chunks: list[list[str]] = []
    if isinstance(obj, dict):
        for key in ("argv", "sample_argv", "argv_hints"):
            val = obj.get(key)
            if isinstance(val, list) and val:
                if all(isinstance(x, str) for x in val):
                    chunks.append(list(val))
                elif all(isinstance(x, list) for x in val):
                    for sub in val:
                        if sub and all(isinstance(x, str) for x in sub):
                            chunks.append(list(sub))
        for val in obj.values():
            chunks.extend(_argv_chunks_from_obj(val))
    elif isinstance(obj, list):
        for item in obj:
            chunks.extend(_argv_chunks_from_obj(item))
    return chunks


def collect_invention_sample_argv(rt: InventionRuntime) -> list[str]:
    """Gather sample.py argv hints from invention repairs and module reports."""
    out: list[str] = []
    for rep in rt.repairs or []:
        if not isinstance(rep, dict):
            continue
        av = rep.get("argv") or rep.get("sample_argv")
        if isinstance(av, list):
            out.extend(str(x) for x in av)
    for chunk in _argv_chunks_from_obj(rt.reports or {}):
        out.extend(chunk)
    risks = (rt.reports.get("oracle") or {}).get("risks") or []
    if rt.invention_spectra:
        out.append("--invention-spectra")
    if any(r in ("binding", "count", "negation", "multi_char") for r in risks):
        if not any(t in out for t in ("--guidance-schedule", "--cfg-schedule")):
            out.extend(["--guidance-schedule", "piecewise"])
    if any(r in ("plastic", "binding") for r in risks):
        if not any(t.startswith("--apg-parallel") for t in out):
            out.extend(["--apg-parallel-eta", "0.0"])
    return out


def fold_invention_sample_argv(argv: list[str]) -> list[str]:
    """Normalize legacy invention argv tokens to current sample.py flags."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        if tok == "--cfg-schedule":
            out.extend(["--guidance-schedule", "piecewise"])
            i += 1
            if i < len(argv) and not str(argv[i]).startswith("-"):
                i += 1
            continue
        if tok.startswith("--cfg-schedule="):
            out.extend(["--guidance-schedule", "piecewise"])
            i += 1
            continue
        if tok == "--apg-parallel":
            out.append("--apg-parallel-eta")
            i += 1
            if i < len(argv) and not str(argv[i]).startswith("-"):
                out.append(str(argv[i]))
                i += 1
            continue
        if tok.startswith("--apg-parallel="):
            out.extend(["--apg-parallel-eta", tok.split("=", 1)[1]])
            i += 1
            continue
        if tok == "--internal-guidance":
            out.append("--internal-guidance-block")
            i += 1
            continue
        out.append(tok)
        i += 1
        if tok.startswith("--") and "=" not in tok and i < len(argv) and not str(argv[i]).startswith("-"):
            out.append(str(argv[i]))
            i += 1
    return out


def _apply_invention_argv_to_namespace(args: Any, argv: list[str]) -> None:
    """Soft-fill argparse namespace from folded invention argv (unset only)."""
    folded = fold_invention_sample_argv(argv)
    i = 0
    while i < len(folded):
        tok = str(folded[i])
        i += 1
        if tok == "--guidance-schedule":
            if i >= len(folded):
                break
            val = str(folded[i])
            i += 1
            if not str(getattr(args, "guidance_schedule", None) or "").strip():
                args.guidance_schedule = val
            continue
        if tok == "--apg-parallel-eta":
            if i >= len(folded):
                break
            val = float(folded[i])
            i += 1
            if float(getattr(args, "apg_parallel_eta", -1.0) or -1.0) < 0:
                args.apg_parallel_eta = val
            continue
        if tok in ("--internal-guidance-block", "--genesis-ops", "--invention-spectra"):
            name = tok.lstrip("-").replace("-", "_")
            if not bool(getattr(args, name, False)):
                setattr(args, name, True)
            continue
        if tok.startswith("--") and "=" in tok:
            flag, val = tok.split("=", 1)
            name = flag.lstrip("-").replace("-", "_")
            cur = getattr(args, name, None)
            if cur in (None, "", False, 0, -1.0):
                try:
                    setattr(args, name, float(val) if "." in val else int(val))
                except ValueError:
                    setattr(args, name, val)


def enrich_prompts_with_inventions(
    prompt: str,
    negative: str = "",
    *,
    enable: str = "auto",
) -> tuple[str, str, dict[str, Any] | None]:
    """Library helper: (positive, negative, optional_box_layout)."""
    rt = prepare_invention_runtime(prompt, negative, enable=enable)
    return rt.positive, rt.negative, rt.box_layout


def apply_invention_to_namespace(args: Any, *, work_dir: str = "outputs/_invention") -> InventionRuntime | None:
    """
    Mutate a sample.py argparse namespace in place.

    Returns the runtime descriptor, or None when inventions are off.
    """
    enable = str(getattr(args, "invention_stack", "off") or "off").strip().lower()
    if enable in ("off", "none", "0", "false", ""):
        return None

    prompt = str(getattr(args, "prompt", "") or "")
    negative = str(getattr(args, "negative_prompt", "") or "")
    adaptive = bool(getattr(args, "invention_adaptive_steps", False))
    auto_spectra = bool(getattr(args, "invention_auto_spectra", False))
    spectra_flag = bool(getattr(args, "invention_spectra", False))
    base_steps = int(getattr(args, "steps", 28) or 28)

    rt = prepare_invention_runtime(
        prompt,
        negative,
        enable=enable,
        base_steps=base_steps,
        adaptive_steps=adaptive,
        auto_spectra=auto_spectra,
        invention_spectra=spectra_flag,
        work_dir=work_dir,
    )
    args.prompt = rt.positive
    if rt.negative:
        args.negative_prompt = rt.negative
    if rt.box_layout_path and not str(getattr(args, "box_layout", "") or "").strip():
        args.box_layout = rt.box_layout_path
        args.anti_bleed = True
    if rt.steps is not None and adaptive:
        args.steps = int(rt.steps)
    if rt.invention_spectra:
        args.invention_spectra = True
    inv_argv = collect_invention_sample_argv(rt)
    if inv_argv:
        _apply_invention_argv_to_namespace(args, inv_argv)
    args._invention_runtime = rt.to_dict()
    return rt
