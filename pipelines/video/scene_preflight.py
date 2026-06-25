"""Pre-flight validation before running a scene graph pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping

__all__ = ["PreflightIssue", "PreflightReport", "run_preflight"]


@dataclass(slots=True)
class PreflightIssue:
    level: str  # error | warn | info
    code: str
    message: str
    path: str = ""


@dataclass(slots=True)
class PreflightReport:
    ok: bool
    issues: List[PreflightIssue] = field(default_factory=list)

    def errors(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.level == "error"]

    def warnings(self) -> List[PreflightIssue]:
        return [i for i in self.issues if i.level == "warn"]


def _check_path(issues: List[PreflightIssue], p: str, *, label: str, required: bool = False) -> None:
    if not p or p.startswith("http"):
        return
    path = Path(p)
    if path.is_file():
        return
    level = "error" if required else "warn"
    issues.append(
        PreflightIssue(
            level=level,
            code="missing_file",
            message=f"{label} not found: {p}",
            path=p,
        )
    )


def run_preflight(
    scene_path: str | Path,
    *,
    ckpt: str = "",
    require_ckpt: bool = False,
) -> PreflightReport:
    from .scene_graph import load_scene_graph

    issues: List[PreflightIssue] = []
    try:
        graph = load_scene_graph(scene_path)
    except Exception as exc:
        return PreflightReport(ok=False, issues=[PreflightIssue("error", "parse_failed", str(exc))])

    if not graph.scene_prompt and not graph.shots:
        issues.append(PreflightIssue("error", "empty_scene", "scene prompt or shots required"))

    if graph.duration_sec <= 0:
        issues.append(PreflightIssue("error", "bad_duration", "duration_sec must be > 0"))

    if require_ckpt and not ckpt:
        issues.append(PreflightIssue("error", "missing_ckpt", "checkpoint required for generation"))
    elif ckpt and not Path(ckpt).is_file():
        issues.append(PreflightIssue("warn", "ckpt_not_found", f"checkpoint not found: {ckpt}", ckpt))

    _check_path(issues, str(graph.anchor_image or ""), label="anchor_image")
    _check_path(issues, str(graph.motion_clip or ""), label="motion_clip")

    for inp in graph.inputs:
        img = str(getattr(inp, "image", "") or "")
        _check_path(issues, img, label=f"input:{getattr(inp, 'id', '?')}.image")

    lib = str(graph.retrieval.get("local_library") or "")
    if lib and not Path(lib).is_dir():
        issues.append(PreflightIssue("warn", "library_missing", f"local_library not found: {lib}", lib))

    edit = graph.edit or {}
    if edit.get("parallel_segments") and int(edit.get("max_segment_workers", 2) or 2) > 4:
        issues.append(
            PreflightIssue(
                "warn",
                "high_parallelism",
                "max_segment_workers > 4 may OOM on single GPU",
            )
        )

    from .continuity_validators import parse_validator_config, run_continuity_validation
    from .thumbnail_rehearsal import parse_thumbnail_config, plan_thumbnails, thumbnail_gate_issues

    shots = list(graph.shots) if graph.shots else []
    if shots or graph.continuity:
        cfg = parse_validator_config(graph.continuity)
        report = run_continuity_validation(shots, continuity=graph.continuity, config=cfg)
        for ci in report.issues:
            issues.append(
                PreflightIssue(
                    level="error" if ci.level == "error" else ("warn" if ci.level == "warn" else "info"),
                    code=ci.code,
                    message=ci.message,
                    path=ci.shot_id,
                )
            )

    thumb_cfg = parse_thumbnail_config(
        graph.continuity,
        studio=graph.raw.get("studio") if isinstance(graph.raw.get("studio"), Mapping) else None,
        edit=graph.edit,
    )
    if thumb_cfg.enabled and shots:
        tplan = plan_thumbnails(
            shots,
            config=thumb_cfg,
            base_prompt=graph.scene_prompt,
            aspect_width=graph.width,
            aspect_height=graph.height,
        )
        for msg in thumbnail_gate_issues(tplan):
            level = "error" if thumb_cfg.gate == "require_approval" else "warn"
            issues.append(PreflightIssue(level, "thumbnail_gate", msg))
        if tplan.enabled:
            issues.append(
                PreflightIssue(
                    "info",
                    "thumbnail_plan",
                    f"{len(tplan.specs)} thumbnail(s), {tplan.pending_count} pending approval",
                )
            )

    ok = not any(i.level == "error" for i in issues)
    return PreflightReport(ok=ok, issues=issues)


def format_preflight_report(report: PreflightReport) -> str:
    lines = [f"Preflight: {'OK' if report.ok else 'FAILED'}"]
    for i in report.issues:
        lines.append(f"  [{i.level.upper()}] {i.code}: {i.message}")
    return "\n".join(lines)
