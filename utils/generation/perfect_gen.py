"""Perfect-gen glue: facet decompose → memory → clarify → per-facet refs → sample plan.

Implements the handwritten agentic stills loop without replacing VisualBrain or
Creative Co-Pilot — it feeds them better structured inputs.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.agentic.clarify_pose_place import (
    apply_clarify_answers,
    assume_pose_place_defaults,
    build_clarify_questions,
)
from utils.agentic.gen_interview import (
    apply_interview_answers,
    build_gen_interview,
    format_interview_for_user,
)
from utils.brain.facet_ref_search import gather_facet_references
from utils.brain.memory_box import MemoryBox, load_memory_box, save_memory_box
from utils.prompt.facet_decompose import PromptFacets, decompose_prompt_facets

__all__ = [
    "PerfectGenConfig",
    "PerfectGenPlan",
    "plan_perfect_gen",
    "build_sample_argv",
]


@dataclass
class PerfectGenConfig:
    work_dir: str = ""
    session_id: str = ""
    allow_web: bool = True
    max_refs_per_facet: int = 4
    assume_missing: bool = True  # if False, only emit questions
    interview: bool = False  # full Q+photo checklist before gen
    clarify_answers: dict[str, str] = field(default_factory=dict)
    interview_answers: dict[str, str] = field(default_factory=dict)
    interview_photos: dict[str, list[str]] = field(default_factory=dict)
    reference_style_mode: str = "instantstyle"
    reference_strength: float = 0.85
    run_rag: bool = True
    character_sheet: str = ""
    user_face_refs: list[str] = field(default_factory=list)
    user_style_refs: list[str] = field(default_factory=list)
    extra_sample_args: list[str] = field(default_factory=list)
    invention_stack: str = "auto"
    invention_spectra: bool = True
    invention_adaptive_steps: bool = True
    invention_auto_spectra: bool = True


@dataclass
class PerfectGenPlan:
    prompt: str
    enriched_prompt: str
    facets: dict[str, Any]
    clarify: dict[str, Any]
    refs: dict[str, list[str]] = field(default_factory=dict)
    moodboard_paths: list[str] = field(default_factory=list)
    moodboard_json: str = ""
    memory_path: str = ""
    sample_argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_user: bool = False
    questions: list[str] = field(default_factory=list)
    interview: dict[str, Any] = field(default_factory=dict)
    interview_text: str = ""
    photo_requests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _maybe_rag(prompt: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    try:
        from utils.prompt.creative_rag import enrich_prompt

        result = enrich_prompt(prompt, creativity_level=0.45, device="cpu")
        enriched = str(getattr(result, "enriched_prompt", "") or "").strip()
        if enriched and enriched != prompt:
            return enriched, ["creative_rag"]
        notes.append("creative_rag: no change")
    except Exception as exc:
        notes.append(f"rag skipped: {exc}")
    return prompt, notes


def build_sample_argv(
    prompt: str,
    *,
    moodboard_paths: list[str] | None = None,
    moodboard_json: str = "",
    reference_style_mode: str = "instantstyle",
    reference_strength: float = 0.85,
    character_sheet: str = "",
    face_refs: list[str] | None = None,
    extra: list[str] | None = None,
    invention_stack: str = "auto",
    invention_spectra: bool = True,
    invention_adaptive_steps: bool = True,
    invention_auto_spectra: bool = True,
) -> list[str]:
    argv = ["--prompt", prompt]
    paths = list(moodboard_paths or [])
    if moodboard_json:
        argv.extend(["--moodboard-json", moodboard_json])
    elif paths:
        # Prefer first style ref as InstantStyle anchor; rest as moodboard list if supported.
        argv.extend(["--reference-image", paths[0]])
        argv.extend(["--reference-style-mode", reference_style_mode])
        argv.extend(["--reference-strength", str(reference_strength)])
        if len(paths) > 1:
            argv.extend(["--moodboard-images", ",".join(paths[:8])])
    faces = [p for p in (face_refs or []) if p]
    if faces and "--reference-image" not in argv:
        argv.extend(["--reference-image", faces[0]])
        argv.extend(["--reference-style-mode", reference_style_mode])
        argv.extend(["--reference-strength", str(reference_strength)])
    if character_sheet:
        argv.extend(["--character-sheet", character_sheet])
    inv = str(invention_stack or "off").strip().lower()
    if inv not in ("off", "none", "0", "false", ""):
        argv.extend(["--invention-stack", inv])
        if invention_spectra:
            argv.append("--invention-spectra")
        if invention_adaptive_steps:
            argv.append("--invention-adaptive-steps")
        if invention_auto_spectra:
            argv.append("--invention-auto-spectra")
    if extra:
        argv.extend(list(extra))
    return argv


def plan_perfect_gen(prompt: str, config: PerfectGenConfig | None = None) -> PerfectGenPlan:
    """Build an end-to-end plan (no GPU sample). Persist memory box under work_dir."""
    cfg = config or PerfectGenConfig()
    sid = (cfg.session_id or uuid.uuid4().hex[:12]).strip()
    work = Path(cfg.work_dir or f"outputs/perfect_gen/{sid}")
    work.mkdir(parents=True, exist_ok=True)
    mem_path = work / "memory_box.json"

    box = load_memory_box(mem_path) or MemoryBox(session_id=sid, prompt=prompt)
    box.prompt = prompt
    box.style_mode = cfg.reference_style_mode
    if cfg.character_sheet:
        box.character_sheet = cfg.character_sheet

    facets: PromptFacets = decompose_prompt_facets(prompt)
    box.facets = facets.to_dict()
    box.remember("decompose", missing=list(facets.missing), nsfw=facets.nsfw)

    # Full interview mode: questions + photo requests (blocks unless answered / assumed).
    if cfg.interview:
        interview = build_gen_interview(
            facets,
            has_face_ref=bool(cfg.user_face_refs),
            has_style_ref=bool(cfg.user_style_refs),
            has_character_sheet=bool(cfg.character_sheet),
            aggressive=True,
        )
        enriched, interview, photo_by_facet = apply_interview_answers(
            prompt,
            interview,
            answers={**cfg.clarify_answers, **cfg.interview_answers},
            photos=cfg.interview_photos,
        )
        # Seed user-provided face/style refs into facet buckets.
        if cfg.user_face_refs:
            photo_by_facet.setdefault("identity", []).extend(cfg.user_face_refs)
        if cfg.user_style_refs:
            photo_by_facet.setdefault("style", []).extend(cfg.user_style_refs)

        interview_text = format_interview_for_user(interview)
        (work / "interview.txt").write_text(interview_text, encoding="utf-8")
        photo_asks = [it.prompt for it in interview.items if it.kind == "photo" and not it.photo_paths]

        if interview.needs_user and not cfg.assume_missing:
            plan = PerfectGenPlan(
                prompt=prompt,
                enriched_prompt=enriched,
                facets=facets.to_dict(),
                clarify={"interview": True},
                memory_path=str(mem_path),
                needs_user=True,
                questions=[it.prompt for it in interview.pending()],
                interview=interview.to_dict(),
                interview_text=interview_text,
                photo_requests=photo_asks,
                notes=["awaiting interview answers/photos; re-run with --interview answers"],
            )
            box.clarify = plan.clarify
            box.remember("interview_block", n=len(plan.questions))
            save_memory_box(box, mem_path)
            (work / "interview.json").write_text(json.dumps(interview.to_dict(), indent=2), encoding="utf-8")
            return plan

        if interview.needs_user and cfg.assume_missing:
            assumed_res = assume_pose_place_defaults(facets)
            enriched = apply_clarify_answers(enriched, assumed_res.assumed)
            box.remember("interview_assume", **assumed_res.assumed)

        # Merge user photos into work refs before web search.
        for facet, paths in photo_by_facet.items():
            dest = work / "refs" / facet
            dest.mkdir(parents=True, exist_ok=True)
            kept: list[str] = []
            for src in paths:
                sp = Path(src)
                if sp.is_file():
                    kept.append(str(sp))
            if kept:
                box.set_refs(facet, kept)

        box.clarify = {"interview": interview.to_dict(), "photos": photo_by_facet}
        (work / "interview.json").write_text(json.dumps(interview.to_dict(), indent=2), encoding="utf-8")

        rag_notes: list[str] = []
        if cfg.run_rag:
            enriched, rag_notes = _maybe_rag(enriched)
            box.rag_notes.extend([str(n) for n in rag_notes])

        if facets.artist_hint and facets.artist_hint.lower() not in enriched.lower():
            enriched = f"{enriched.rstrip(',')}, {facets.artist_hint} style"

        # Web scrape only facets still empty (user photos win).
        ref_pack = gather_facet_references(
            decompose_prompt_facets(enriched),
            work_dir=work,
            max_per_facet=cfg.max_refs_per_facet,
            allow_web=cfg.allow_web,
        )
        for k, v in ref_pack.paths.items():
            if not (box.refs.get(k) or []):
                box.set_refs(k, v)

        mood_paths = list(box.refs.get("style") or []) + list(box.refs.get("subject") or [])
        mood_paths = list(dict.fromkeys(mood_paths + ref_pack.moodboard_paths()))
        face_paths = list(box.refs.get("identity") or []) + list(cfg.user_face_refs or [])
        mood_json = ""
        if mood_paths:
            mood_payload = {
                "created": time.time(),
                "paths": mood_paths,
                "images": mood_paths,
                "by_facet": dict(box.refs),
                "queries": ref_pack.queries,
            }
            mood_path = work / "moodboard.json"
            mood_path.write_text(json.dumps(mood_payload, indent=2), encoding="utf-8")
            mood_json = str(mood_path)

        save_memory_box(box, mem_path)
        argv = build_sample_argv(
            enriched,
            moodboard_paths=mood_paths,
            moodboard_json=mood_json,
            reference_style_mode=cfg.reference_style_mode,
            reference_strength=cfg.reference_strength,
            character_sheet=cfg.character_sheet or box.character_sheet,
            face_refs=face_paths,
            extra=cfg.extra_sample_args,
            invention_stack=cfg.invention_stack,
            invention_spectra=cfg.invention_spectra,
            invention_adaptive_steps=cfg.invention_adaptive_steps,
            invention_auto_spectra=cfg.invention_auto_spectra,
        )
        notes = list(ref_pack.notes) + list(rag_notes)
        notes.append(interview.summary)
        return PerfectGenPlan(
            prompt=prompt,
            enriched_prompt=enriched,
            facets=facets.to_dict(),
            clarify=box.clarify,
            refs=dict(box.refs),
            moodboard_paths=mood_paths,
            moodboard_json=mood_json,
            memory_path=str(mem_path),
            sample_argv=argv,
            notes=notes,
            needs_user=False,
            questions=[],
            interview=interview.to_dict(),
            interview_text=interview_text,
            photo_requests=[],
        )

    clarify_info = build_clarify_questions(facets)
    enriched = prompt
    assumed: dict[str, str] = {}
    if cfg.clarify_answers:
        assumed = dict(cfg.clarify_answers)
        enriched = apply_clarify_answers(prompt, assumed)
        clarify_info.needs_user = False
        clarify_info.questions = []
    elif cfg.assume_missing and facets.missing:
        assumed_res = assume_pose_place_defaults(facets)
        assumed = dict(assumed_res.assumed)
        enriched = assumed_res.enriched_prompt or prompt
        clarify_info.notes = assumed_res.notes
    elif clarify_info.needs_user:
        # Stop early — caller should ask user.
        plan = PerfectGenPlan(
            prompt=prompt,
            enriched_prompt=prompt,
            facets=facets.to_dict(),
            clarify={"questions": clarify_info.questions, "assumed": {}},
            memory_path=str(mem_path),
            needs_user=True,
            questions=list(clarify_info.questions),
            notes=["awaiting clarify answers; re-run with clarify_answers or assume_missing=True"],
        )
        box.clarify = plan.clarify
        save_memory_box(box, mem_path)
        return plan

    rag_notes = []
    if cfg.run_rag:
        enriched, rag_notes = _maybe_rag(enriched)
        box.rag_notes.extend([str(n) for n in rag_notes])

    # Artist style token into preferences / prompt if extracted.
    if facets.artist_hint and facets.artist_hint.lower() not in enriched.lower():
        enriched = f"{enriched.rstrip(',')}, {facets.artist_hint} style"

    ref_pack = gather_facet_references(
        decompose_prompt_facets(enriched) if enriched != prompt else facets,
        work_dir=work,
        max_per_facet=cfg.max_refs_per_facet,
        allow_web=cfg.allow_web,
    )
    for k, v in ref_pack.paths.items():
        box.set_refs(k, v)

    mood_paths = ref_pack.moodboard_paths()
    mood_json = ""
    if mood_paths:
        mood_payload = {
            "created": time.time(),
            "paths": mood_paths,
            "by_facet": ref_pack.paths,
            "queries": ref_pack.queries,
        }
        mood_path = work / "moodboard.json"
        mood_path.write_text(json.dumps(mood_payload, indent=2), encoding="utf-8")
        mood_json = str(mood_path)

    box.clarify = {"questions": clarify_info.questions, "assumed": assumed}
    box.preferences.setdefault("nsfw", facets.nsfw)
    save_memory_box(box, mem_path)

    argv = build_sample_argv(
        enriched,
        moodboard_paths=mood_paths,
        moodboard_json=mood_json,
        reference_style_mode=cfg.reference_style_mode,
        reference_strength=cfg.reference_strength,
        character_sheet=cfg.character_sheet or box.character_sheet,
        face_refs=cfg.user_face_refs,
        extra=cfg.extra_sample_args,
        invention_stack=cfg.invention_stack,
        invention_spectra=cfg.invention_spectra,
        invention_adaptive_steps=cfg.invention_adaptive_steps,
        invention_auto_spectra=cfg.invention_auto_spectra,
    )

    notes = list(ref_pack.notes) + list(rag_notes)
    if assumed:
        notes.append(f"assumed: {assumed}")
    if facets.nsfw:
        notes.append("nsfw prompt detected — facet search used broader art-reference wording")

    return PerfectGenPlan(
        prompt=prompt,
        enriched_prompt=enriched,
        facets=facets.to_dict(),
        clarify=box.clarify,
        refs=dict(ref_pack.paths),
        moodboard_paths=mood_paths,
        moodboard_json=mood_json,
        memory_path=str(mem_path),
        sample_argv=argv,
        notes=notes,
        needs_user=False,
        questions=[],
    )
