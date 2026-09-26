"""Creative Co-Pilot — 4-step *Decompose and Apply* style pipeline.

Modular (not a single LoRA): inspiration → style text+features → structure
control → InstantStyle / moodboard fusion into ``sample.py``.

Weight-safe: VLMs, depth nets, and dense encoders are optional; PIL canny and
CLIP InstantStyle always work when references exist.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STYLE_DECOMPOSE_PROMPT = (
    "Specifically analyze the art style of this image. Describe brushwork, texture, "
    "color palette, lighting scheme, line weight, rendering technique, and historical "
    "art movement influences. Ignore subject matter, characters, and objects — style only. "
    "Write dense prose suitable for a text-to-image prompt."
)

MOODBOARD_QUERY_PROMPT = (
    "Given this creative goal, list 3 short image-search queries (one per line) that would "
    "find strong visual style references (palette, lighting, brushwork). No commentary."
)


@dataclass(slots=True)
class CopilotConfig:
    web_search: bool = True
    max_search_images: int = 4
    use_vlm: bool = True
    extract_control: bool = True
    control_prefer: str = "canny"  # canny | depth | softedge
    reference_style_mode: str = "instantstyle"
    reference_strength: float = 0.85
    moodboard_strength: float = 1.0
    device: str = "cuda"
    content_prompt: str = "a photo"


@dataclass(slots=True)
class CopilotPlan:
    """Artifacts + argv for the fusion step."""

    goal_prompt: str
    fused_prompt: str
    style_descriptions: list[str] = field(default_factory=list)
    moodboard_paths: list[str] = field(default_factory=list)
    moodboard_json: str = ""
    control_image: str = ""
    control_type: str = ""
    search_queries: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sample_argv: list[str] = field(default_factory=list)
    work_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def step1_inspiration(
    goal: str,
    *,
    work_dir: Path,
    config: CopilotConfig,
    reference_images: Sequence[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Gather moodboard paths + search queries. Returns (paths, queries, notes)."""
    notes: list[str] = []
    queries: list[str] = []
    paths: list[str] = [str(p) for p in (reference_images or []) if str(p).strip()]

    # Heuristic search queries from the goal (always available).
    g = str(goal).strip()
    if g:
        queries.append(g[:100])
        queries.append(f"{g[:60]} art style lighting palette")
        queries.append(f"{g[:50]} cinematic moodboard reference")

    if config.use_vlm and not dry_run and paths:
        try:
            from utils.brain.understand import caption_image_vlm

            raw = caption_image_vlm(
                paths[0],
                user_prompt=f"{MOODBOARD_QUERY_PROMPT}\nGoal: {g}",
                device=config.device,
            ).strip()
            vlm_qs = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()][:3]
            if vlm_qs:
                queries = vlm_qs + queries
                notes.append(f"moodboard queries from VLM on {Path(paths[0]).name}")
        except Exception as exc:
            notes.append(f"vlm query skip: {exc}")
    elif config.use_vlm and not paths:
        notes.append("moodboard queries: heuristic (no seed image for VLM query gen)")

    if config.web_search and queries and not dry_run:
        try:
            from utils.brain.image_search import download_search_hits, search_reference_images

            search_dir = work_dir / "search"
            search_dir.mkdir(parents=True, exist_ok=True)
            for q in queries[:2]:
                sr = search_reference_images(q, max_results=config.max_search_images, allow_web=True)
                notes.append(f"search '{q[:40]}…': {len(sr.hits)} hits ({sr.notes})")
                if sr.hits:
                    downloaded = download_search_hits(
                        sr.hits, search_dir, max_download=max(1, config.max_search_images // 2)
                    )
                    for h in downloaded:
                        if h.local_path and h.local_path not in paths:
                            paths.append(h.local_path)
        except Exception as exc:
            notes.append(f"web search failed: {exc}")
    elif not config.web_search:
        notes.append("web search disabled")

    return paths, queries, notes


def step2_style_decompose(
    image_paths: Sequence[str],
    *,
    work_dir: Path,
    config: CopilotConfig,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """Style-only VLM prose (+ notes). Dense DINO/SigLIP is optional and recorded in notes."""
    notes: list[str] = []
    descriptions: list[str] = []
    style_dir = work_dir / "style"
    if not dry_run:
        style_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(image_paths):
        p = Path(path)
        if not p.is_file():
            notes.append(f"missing ref: {path}")
            continue
        desc = ""
        if config.use_vlm and not dry_run:
            try:
                from utils.brain.understand import caption_image_vlm

                desc = caption_image_vlm(str(p), user_prompt=STYLE_DECOMPOSE_PROMPT, device=config.device).strip()
            except Exception as exc:
                notes.append(f"style VLM failed ({p.name}): {exc}")
        if not desc and not dry_run:
            try:
                import numpy as np
                from PIL import Image

                img = Image.open(p).convert("RGB")
                arr = np.array(img.resize((48, 48)))
                mean = arr.mean(axis=(0, 1)).astype(int).tolist()
                desc = (
                    f"stylized rendering with dominant palette RGB {mean}, "
                    "soft atmospheric lighting, painterly texture, cinematic contrast"
                )
                notes.append(f"style fallback heuristic for {p.name}")
            except Exception:
                desc = "painterly cinematic style, rich color grading, detailed brushwork"
        if desc:
            descriptions.append(desc)
            if not dry_run:
                (style_dir / f"style_{i:02d}.txt").write_text(desc + "\n", encoding="utf-8")

    # Optional dense encoder + native accel inventory.
    try:
        from utils.ai_accel import native_stack_summary

        notes.append(f"ai_accel: {native_stack_summary()}")
    except Exception as exc:
        notes.append(f"ai_accel inventory failed: {exc}")
    try:
        from utils.modeling.hf_scaffold import has_local_weights
        from utils.modeling.model_paths import (
            default_dinov3_vitb16_path,
            default_dinov3_vith16plus_path,
            default_instantstyle_path,
            default_ip_adapter_path,
            default_siglip2_base_256_path,
            default_siglip2_so400m_path,
        )

        for label, fn in (
            ("DINOv3-H+", default_dinov3_vith16plus_path),
            ("DINOv3-B16", default_dinov3_vitb16_path),
            ("SigLIP2-SO400M", default_siglip2_so400m_path),
            ("SigLIP2-base-256", default_siglip2_base_256_path),
            ("InstantStyle", default_instantstyle_path),
            ("IP-Adapter", default_ip_adapter_path),
        ):
            try:
                local = fn()
                if local and has_local_weights(local):
                    notes.append(f"{label} weights present at {local}")
                else:
                    notes.append(f"{label}: scaffold/config only — path resolves to {local}")
            except Exception:
                notes.append(f"{label}: path unresolved")
    except Exception:
        notes.append("dense encoder inventory skipped")

    return descriptions, notes


def step3_condition(
    *,
    work_dir: Path,
    config: CopilotConfig,
    structure_image: str = "",
    moodboard_paths: Sequence[str] = (),
    dry_run: bool = False,
) -> tuple[str, str, list[str]]:
    """Build a control map. Returns (control_image, control_type, notes)."""
    notes: list[str] = []
    if not config.extract_control:
        notes.append("control extraction disabled")
        return "", "", notes

    src = str(structure_image or "").strip()
    if not src and moodboard_paths:
        src = str(moodboard_paths[0])
    if not src or dry_run:
        if dry_run:
            notes.append("control: dry-run skip")
        else:
            notes.append("control: no structure image")
        return "", "", notes

    ctrl_dir = work_dir / "control"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    prefer = str(config.control_prefer or "canny").strip().lower()
    types = [prefer]
    if prefer != "canny":
        types.append("canny")

    try:
        from utils.brain.understand import extract_control_maps

        maps = extract_control_maps(src, ctrl_dir, types=types, device=config.device)
        if prefer in maps:
            notes.append(f"control: {prefer} from {Path(src).name}")
            return maps[prefer], prefer, notes
        if "canny" in maps:
            notes.append(f"control: fell back to canny from {Path(src).name}")
            return maps["canny"], "canny", notes
        notes.append(f"control: extract returned empty ({list(maps)})")
    except Exception as exc:
        notes.append(f"control failed: {exc}")
    return "", "", notes


def fuse_prompt(goal: str, style_descriptions: Sequence[str], *, max_style_chars: int = 900) -> str:
    """Combine user intent with style prose (subject from goal, style from refs)."""
    g = str(goal).strip()
    styles = [s.strip() for s in style_descriptions if str(s).strip()]
    if not styles:
        return g
    blob = " ".join(styles)
    if len(blob) > max_style_chars:
        blob = blob[: max_style_chars - 1].rstrip() + "…"
    return f"{g}. Art style (ignore reference subjects): {blob}"


def build_sample_argv(
    *,
    ckpt: str,
    plan: CopilotPlan,
    out: str,
    device: str = "cuda",
    extra: Sequence[str] | None = None,
    config: CopilotConfig | None = None,
) -> list[str]:
    cfg = config or CopilotConfig()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "sample.py"),
        "--ckpt",
        str(ckpt),
        "--prompt",
        plan.fused_prompt or plan.goal_prompt,
        "--out",
        str(out),
        "--device",
        str(device),
        "--reference-style-mode",
        str(cfg.reference_style_mode),
        "--reference-strength",
        str(cfg.reference_strength),
        "--reference-content-prompt",
        str(cfg.content_prompt),
    ]
    if plan.moodboard_json:
        cmd.extend(["--moodboard-json", plan.moodboard_json, "--moodboard-strength", str(cfg.moodboard_strength)])
    elif plan.moodboard_paths:
        cmd.extend(
            [
                "--moodboard-images",
                ",".join(plan.moodboard_paths),
                "--moodboard-strength",
                str(cfg.moodboard_strength),
            ]
        )
    if plan.control_image:
        cmd.extend(["--control-image", plan.control_image])
        if plan.control_type:
            # Native DiT control uses the map as-is; type is informational in argv if supported.
            cmd.extend(["--control-type", plan.control_type])
    if extra:
        cmd.extend(list(extra))
    return cmd


def run_creative_copilot(
    *,
    goal: str,
    ckpt: str = "",
    work_dir: str | Path = "creative_copilot_run",
    out: str = "copilot_out.png",
    reference_images: Sequence[str] | None = None,
    structure_image: str = "",
    config: CopilotConfig | None = None,
    dry_run: bool = False,
    execute: bool = False,
    extra_sample_args: Sequence[str] | None = None,
) -> CopilotPlan:
    """
    Run the 4-step Decompose-and-Apply stack.

    If ``execute`` and ``ckpt`` are set, shells out to ``sample.py`` with InstantStyle fusion.
    """
    cfg = config or CopilotConfig()
    work = Path(work_dir)
    if not dry_run:
        work.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    paths, queries, n1 = step1_inspiration(
        goal, work_dir=work, config=cfg, reference_images=reference_images, dry_run=dry_run
    )
    notes.extend(n1)

    descs, n2 = step2_style_decompose(paths, work_dir=work, config=cfg, dry_run=dry_run)
    notes.extend(n2)

    control_image, control_type, n3 = step3_condition(
        work_dir=work,
        config=cfg,
        structure_image=structure_image,
        moodboard_paths=paths,
        dry_run=dry_run,
    )
    notes.extend(n3)

    moodboard_json = ""
    if paths and not dry_run:
        moodboard_json = str(work / "moodboard.json")
        _write_json(Path(moodboard_json), {"images": paths})

    fused = fuse_prompt(goal, descs)
    plan = CopilotPlan(
        goal_prompt=str(goal).strip(),
        fused_prompt=fused,
        style_descriptions=descs,
        moodboard_paths=list(paths),
        moodboard_json=moodboard_json,
        control_image=control_image,
        control_type=control_type,
        search_queries=list(queries),
        notes=notes,
        work_dir=str(work),
    )
    if ckpt:
        plan.sample_argv = build_sample_argv(
            ckpt=ckpt,
            plan=plan,
            out=out,
            device=cfg.device,
            extra=extra_sample_args,
            config=cfg,
        )

    if not dry_run:
        _write_json(work / "copilot_plan.json", plan.to_dict())
        (work / "fused_prompt.txt").write_text(fused + "\n", encoding="utf-8")

    if execute and plan.sample_argv and not dry_run:
        notes.append("executing sample.py fusion")
        subprocess.run(plan.sample_argv, check=False)
    elif execute and dry_run:
        notes.append("execute skipped (dry-run)")
    elif execute and not ckpt:
        notes.append("execute skipped (no --ckpt)")

    plan.notes = notes
    if not dry_run:
        _write_json(work / "copilot_plan.json", plan.to_dict())
    return plan


def format_plan_summary(plan: CopilotPlan) -> str:
    lines = [
        f"goal: {plan.goal_prompt[:120]}",
        f"moodboard refs: {len(plan.moodboard_paths)}",
        f"style captions: {len(plan.style_descriptions)}",
        f"control: {plan.control_type or 'none'} {plan.control_image or ''}",
        f"fused prompt chars: {len(plan.fused_prompt)}",
    ]
    if plan.sample_argv:
        lines.append("sample: " + shlex.join(plan.sample_argv))
    for n in plan.notes[:12]:
        lines.append(f"note: {n}")
    return "\n".join(lines)


__all__ = [
    "CopilotConfig",
    "CopilotPlan",
    "MOODBOARD_QUERY_PROMPT",
    "STYLE_DECOMPOSE_PROMPT",
    "build_sample_argv",
    "format_plan_summary",
    "fuse_prompt",
    "run_creative_copilot",
    "step1_inspiration",
    "step2_style_decompose",
    "step3_condition",
]
