"""Stills planners that build ``sample.py`` runs: user taste, perfect-gen, the agentic stills orchestrator and the creative co-pilot."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.agentic.agent import MemoryBox, load_memory_box, save_memory_box
from utils.agentic.interview import (
    apply_clarify_answers,
    apply_interview_answers,
    assume_pose_place_defaults,
    build_clarify_questions,
    build_gen_interview,
    build_post_gen_critique,
    critique_answers_to_fix_plan,
    format_interview_for_user,
)
from utils.captioning import gather_facet_references
from utils.prompt.scene import PromptFacets, decompose_prompt_facets


def default_taste_path() -> Path:
    return Path("outputs") / "user_taste.json"


@dataclass
class UserTaste:
    """Persistent preferences across perfect-gen / sample sessions."""

    favorite_artists: list[str] = field(default_factory=list)
    style_axis: str = ""  # photoreal | anime | painterly | 3d | mixed
    lighting_bias: str = ""  # dark | bright | neon | soft | dramatic
    default_pose: str = ""
    default_place: str = ""
    aspect_intent: str = ""  # phone | poster | square | desktop | portrait
    negative_bank: list[str] = field(default_factory=list)
    hates: list[str] = field(default_factory=list)  # free-text hates → negatives
    likes_notes: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserTaste:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


_QUIZ: list[dict[str, Any]] = [
    {
        "id": "style_axis",
        "prompt": "Realism vs stylized — what do you usually want?",
        "choices": ["photoreal", "anime", "painterly", "3d", "mixed"],
    },
    {
        "id": "lighting_bias",
        "prompt": "Default lighting mood?",
        "choices": ["soft", "bright", "dark", "neon", "dramatic"],
    },
    {
        "id": "favorite_artists",
        "prompt": "Favorite artists/styles to lean on? (comma-separated, or 'none')",
        "choices": [],
    },
    {
        "id": "hates",
        "prompt": "What do you hate in AI art? (comma-separated → personal negative bank)",
        "choices": [],
    },
    {
        "id": "aspect_intent",
        "prompt": "Typical output shape?",
        "choices": ["phone", "poster", "square", "desktop", "portrait"],
    },
    {
        "id": "default_pose",
        "prompt": "Default pose when you don't specify one?",
        "choices": [
            "standing, looking at viewer, full body",
            "sitting, relaxed",
            "portrait upper body",
            "dynamic action",
        ],
    },
]


def build_taste_quiz() -> list[dict[str, Any]]:
    return [dict(q) for q in _QUIZ]


def apply_taste_quiz_answers(taste: UserTaste, answers: dict[str, str]) -> UserTaste:
    if "style_axis" in answers:
        taste.style_axis = str(answers["style_axis"]).strip()
    if "lighting_bias" in answers:
        taste.lighting_bias = str(answers["lighting_bias"]).strip()
    if "aspect_intent" in answers:
        taste.aspect_intent = str(answers["aspect_intent"]).strip()
    if "default_pose" in answers:
        taste.default_pose = str(answers["default_pose"]).strip()
    if "default_place" in answers:
        taste.default_place = str(answers["default_place"]).strip()
    if "favorite_artists" in answers:
        raw = str(answers["favorite_artists"]).strip()
        if raw.lower() not in ("none", "n/a", ""):
            taste.favorite_artists = [a.strip() for a in raw.split(",") if a.strip()]
    if "hates" in answers:
        raw = str(answers["hates"]).strip()
        hates = [h.strip() for h in raw.split(",") if h.strip()]
        taste.hates = hates
        # Expand into negative bank tokens
        bank = list(taste.negative_bank)
        for h in hates:
            if h.lower() not in {b.lower() for b in bank}:
                bank.append(h)
        # Common AI-hate expansions
        expansions = {
            "plastic": "plastic skin, waxy skin, oversmoothed",
            "hands": "bad hands, extra fingers, fused fingers",
            "text": "watermark, signature, text artifacts",
            "blurry": "blurry, low detail, soft focus mush",
        }
        blob = " ".join(hates).lower()
        for key, extra in expansions.items():
            if key in blob:
                for tok in extra.split(","):
                    t = tok.strip()
                    if t and t.lower() not in {b.lower() for b in bank}:
                        bank.append(t)
        taste.negative_bank = bank
    taste.updated_at = time.time()
    return taste


def apply_taste_to_prompts(
    positive: str,
    negative: str = "",
    taste: UserTaste | None = None,
) -> tuple[str, str]:
    """Merge taste defaults into pos/neg prompts."""
    if taste is None:
        return positive, negative
    pos = str(positive or "").rstrip(",")
    neg = str(negative or "").rstrip(",")
    bits: list[str] = []
    if taste.style_axis and taste.style_axis not in pos.lower():
        axis_map = {
            "photoreal": "photorealistic, natural skin texture",
            "anime": "anime illustration, clean lineart",
            "painterly": "painterly brushwork, traditional media feel",
            "3d": "3d render, subsurface scattering",
            "mixed": "",
        }
        add = axis_map.get(taste.style_axis.lower(), taste.style_axis)
        if add:
            bits.append(add)
    if taste.lighting_bias:
        light_map = {
            "soft": "soft diffused lighting",
            "bright": "bright airy lighting",
            "dark": "moody low-key lighting",
            "neon": "neon rim lighting, cyberpunk glow",
            "dramatic": "dramatic chiaroscuro lighting",
        }
        bits.append(light_map.get(taste.lighting_bias.lower(), taste.lighting_bias))
    for artist in taste.favorite_artists[:3]:
        token = f"{artist} style"
        if token.lower() not in pos.lower():
            bits.append(token)
    for b in bits:
        if b.lower() not in pos.lower():
            pos = f"{pos}, {b}" if pos else b
    # Negatives
    bank = list(taste.negative_bank) + list(taste.hates)
    for tok in bank:
        if tok and tok.lower() not in neg.lower():
            neg = f"{neg}, {tok}" if neg else tok
    return pos.strip(", "), neg.strip(", ")


def load_user_taste(path: str | Path | None = None) -> UserTaste:
    p = Path(path) if path else default_taste_path()
    if not p.is_file():
        return UserTaste()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return UserTaste.from_dict(data)
    except Exception:
        pass
    return UserTaste()


def save_user_taste(taste: UserTaste, path: str | Path | None = None) -> Path:
    p = Path(path) if path else default_taste_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    taste.updated_at = time.time()
    p.write_text(json.dumps(taste.to_dict(), indent=2), encoding="utf-8")
    return p


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
    reference_strength: float = 0.85
    run_rag: bool = True
    character_sheet: str = ""
    user_face_refs: list[str] = field(default_factory=list)
    user_style_refs: list[str] = field(default_factory=list)
    extra_sample_args: list[str] = field(default_factory=list)
    invention_stack: str = "auto"
    invention_adaptive_steps: bool = True


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
        from utils.prompt.rag import enrich_prompt

        result = enrich_prompt(prompt, creativity_level=0.45, device="cpu")
        enriched = str(getattr(result, "enriched_prompt", "") or "").strip()
        if enriched and enriched != prompt:
            return enriched, ["creative_rag"]
        notes.append("creative_rag: no change")
    except Exception as exc:
        notes.append(f"rag skipped: {exc}")
    return prompt, notes


def perfect_gen_argv(
    prompt: str,
    *,
    moodboard_paths: list[str] | None = None,
    moodboard_json: str = "",
    reference_strength: float = 0.85,
    character_sheet: str = "",
    face_refs: list[str] | None = None,
    extra: list[str] | None = None,
    invention_stack: str = "auto",
    invention_adaptive_steps: bool = True,
) -> list[str]:
    """``sample.py`` argv (no ckpt/out): moodboard, a reference image (first style ref, else first face), sheet."""
    from utils.generation.sample_argv import sample_argv

    paths = list(moodboard_paths or [])
    faces = [p for p in (face_refs or []) if p]
    values: dict[str, Any] = {"prompt": prompt}
    if moodboard_json:
        values["moodboard_json"] = moodboard_json
    elif len(paths) > 1:
        values["moodboard_images"] = ",".join(paths[:8])
    ref = paths[0] if paths and not moodboard_json else (faces[0] if faces else "")
    if ref:
        values.update(reference_image=ref, reference_strength=reference_strength)
    values["character_sheet"] = character_sheet
    inv = str(invention_stack or "off").strip().lower()
    if inv not in ("off", "none", "0", "false", ""):
        values.update(invention_stack=inv, invention_adaptive_steps=bool(invention_adaptive_steps))
    return sample_argv(values) + list(extra or ())


def plan_perfect_gen(prompt: str, config: PerfectGenConfig | None = None) -> PerfectGenPlan:
    """Build an end-to-end plan (no GPU sample). Persist memory box under work_dir."""
    cfg = config or PerfectGenConfig()
    sid = (cfg.session_id or uuid.uuid4().hex[:12]).strip()
    work = Path(cfg.work_dir or f"outputs/perfect_gen/{sid}")
    work.mkdir(parents=True, exist_ok=True)
    mem_path = work / "memory_box.json"

    box = load_memory_box(mem_path) or MemoryBox(session_id=sid, prompt=prompt)
    box.prompt = prompt
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
        argv = perfect_gen_argv(
            enriched,
            moodboard_paths=mood_paths,
            moodboard_json=mood_json,
            reference_strength=cfg.reference_strength,
            character_sheet=cfg.character_sheet or box.character_sheet,
            face_refs=face_paths,
            extra=cfg.extra_sample_args,
            invention_stack=cfg.invention_stack,
            invention_adaptive_steps=cfg.invention_adaptive_steps,
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

    argv = perfect_gen_argv(
        enriched,
        moodboard_paths=mood_paths,
        moodboard_json=mood_json,
        reference_strength=cfg.reference_strength,
        character_sheet=cfg.character_sheet or box.character_sheet,
        face_refs=cfg.user_face_refs,
        extra=cfg.extra_sample_args,
        invention_stack=cfg.invention_stack,
        invention_adaptive_steps=cfg.invention_adaptive_steps,
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


# Output-size presets for ``print_intent`` / the taste profile's ``aspect_intent``.
PRINT_INTENTS: dict[str, tuple[int, int, str]] = {
    "phone": (576, 1024, "phone wallpaper"),
    "portrait": (768, 1024, "portrait"),
    "square": (1024, 1024, "square social"),
    "desktop": (1280, 720, "desktop wallpaper"),
    "poster": (768, 1152, "poster print"),
    "album": (1024, 1024, "album cover"),
}


@dataclass(slots=True)
class PrintIntent:
    key: str
    width: int
    height: int
    label: str

    def sample_argv(self) -> list[str]:
        return ["--width", str(self.width), "--height", str(self.height)]


def resolve_print_intent(name: str) -> PrintIntent | None:
    key = str(name or "").strip().lower()
    if key not in PRINT_INTENTS:
        return None
    w, h, label = PRINT_INTENTS[key]
    return PrintIntent(key=key, width=w, height=h, label=label)


@dataclass
class DraftThumbPlan:
    """Plan for a low-step multi-seed draft pass, then a locked final."""

    prompt: str
    num_drafts: int = 4
    draft_steps: int = 12
    draft_width: int = 512
    draft_height: int = 512
    final_steps: int = 40
    final_width: int = 1024
    final_height: int = 1024
    pick_metric: str = "combo"
    base_seed: int = 0
    draft_argv: list[str] = field(default_factory=list)
    final_argv_template: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_draft_thumbnails(
    prompt: str,
    *,
    num_drafts: int = 4,
    draft_steps: int = 12,
    draft_size: int = 512,
    final_steps: int = 40,
    final_width: int = 1024,
    final_height: int = 1024,
    pick_metric: str = "combo",
    base_seed: int = 0,
    extra: list[str] | None = None,
) -> DraftThumbPlan:
    """
    Build argv for ``sample.py`` draft pass (``--num N --steps low --pick-best``)
    and a final argv template (caller fills ``--seed`` after user/auto pick).
    """
    n = max(2, int(num_drafts))
    extra = list(extra or [])
    draft = [
        "--prompt",
        prompt,
        "--num",
        str(n),
        "--steps",
        str(max(4, int(draft_steps))),
        "--width",
        str(int(draft_size)),
        "--height",
        str(int(draft_size)),
        "--seed",
        str(int(base_seed)),
        "--pick-best",
        str(pick_metric),
    ] + extra
    final_tmpl = [
        "--prompt",
        prompt,
        "--num",
        "1",
        "--steps",
        str(max(draft_steps, int(final_steps))),
        "--width",
        str(int(final_width)),
        "--height",
        str(int(final_height)),
        # --seed filled by promote_draft_to_final
    ] + extra
    return DraftThumbPlan(
        prompt=prompt,
        num_drafts=n,
        draft_steps=int(draft_steps),
        draft_width=int(draft_size),
        draft_height=int(draft_size),
        final_steps=int(final_steps),
        final_width=int(final_width),
        final_height=int(final_height),
        pick_metric=str(pick_metric),
        base_seed=int(base_seed),
        draft_argv=draft,
        final_argv_template=final_tmpl,
        notes=[
            f"Run draft: sample.py {' '.join(draft)}",
            "User picks index 0..N-1 (or trust --pick-best winner), then promote with seed=base+index",
        ],
    )


def promote_draft_to_final(
    plan: DraftThumbPlan,
    *,
    chosen_index: int = 0,
    user_picked: bool = False,
    draft_paths: list[str] | None = None,
    log_feedback: bool = True,
) -> list[str]:
    """Lock final seed from draft index (seed = base_seed + index)."""
    idx = max(0, min(int(chosen_index), plan.num_drafts - 1))
    seed = int(plan.base_seed) + idx
    argv = list(plan.final_argv_template)
    if "--seed" not in argv:
        argv.extend(["--seed", str(seed)])
    else:
        i = argv.index("--seed")
        if i + 1 < len(argv):
            argv[i + 1] = str(seed)
    if user_picked and log_feedback and draft_paths and len(draft_paths) >= 2:
        try:
            from utils.training.preference import record_pick

            win = draft_paths[idx] if idx < len(draft_paths) else draft_paths[0]
            losers = [p for i, p in enumerate(draft_paths) if i != idx]
            record_pick(win, losers, prompt=plan.prompt, seed=seed)
        except Exception:
            pass
    note = "user-picked" if user_picked else "auto-pick-best index"
    plan.notes.append(f"final seed={seed} ({note}, draft[{idx}])")
    return argv


@dataclass
class AgenticStillsPlan:
    session_id: str
    work_dir: str
    perfect: dict[str, Any] = field(default_factory=dict)
    draft: dict[str, Any] = field(default_factory=dict)
    critique: dict[str, Any] = field(default_factory=dict)
    taste: dict[str, Any] = field(default_factory=dict)
    interview_text: str = ""
    needs_user: bool = False
    questions: list[str] = field(default_factory=list)
    sample_draft_argv: list[str] = field(default_factory=list)
    sample_final_argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_agentic_stills(
    prompt: str,
    *,
    work_dir: str = "",
    session_id: str = "",
    interview: bool = True,
    drafts: bool = True,
    num_drafts: int = 4,
    assume_missing: bool = True,
    taste_path: str = "",
    print_intent: str = "",
    allow_web: bool = True,
    critique_image: str = "",
    critique_answers: dict[str, str] | None = None,
    interview_answers: dict[str, str] | None = None,
    face_refs: list[str] | None = None,
    chosen_draft_index: int | None = None,
) -> AgenticStillsPlan:
    """End-to-end stills plan: taste → perfect_gen → draft thumbs → optional critique."""
    sid = (session_id or uuid.uuid4().hex[:12]).strip()
    work = Path(work_dir or f"outputs/agentic_stills/{sid}")
    work.mkdir(parents=True, exist_ok=True)

    taste = load_user_taste(taste_path or str(work / "user_taste.json"))
    pos, neg = apply_taste_to_prompts(prompt, "", taste)
    notes = []
    if pos != prompt:
        notes.append("applied user taste to prompt")

    intent = resolve_print_intent(print_intent or taste.aspect_intent)
    extra: list[str] = []
    if intent:
        extra.extend(intent.sample_argv())
        notes.append(f"print intent: {intent.label} {intent.width}x{intent.height}")
    if neg:
        extra.extend(["--negative-prompt", neg])

    pg_cfg = PerfectGenConfig(
        work_dir=str(work / "perfect"),
        session_id=sid,
        allow_web=allow_web,
        assume_missing=assume_missing,
        interview=interview,
        interview_answers=dict(interview_answers or {}),
        user_face_refs=list(face_refs or []),
        run_rag=True,
        extra_sample_args=list(extra),
    )
    perfect: PerfectGenPlan = plan_perfect_gen(pos, pg_cfg)

    out = AgenticStillsPlan(
        session_id=sid,
        work_dir=str(work),
        perfect=perfect.to_dict(),
        taste=taste.to_dict(),
        interview_text=perfect.interview_text,
        needs_user=perfect.needs_user,
        questions=list(perfect.questions),
        notes=notes + list(perfect.notes),
    )

    if perfect.needs_user:
        (work / "plan.json").write_text(json.dumps(out.to_dict(), indent=2), encoding="utf-8")
        return out

    enriched = perfect.enriched_prompt
    draft_plan: DraftThumbPlan | None = None
    if drafts:
        draft_plan = plan_draft_thumbnails(
            enriched,
            num_drafts=num_drafts,
            extra=list(extra),
        )
        # Merge moodboard/ref flags from perfect plan into draft/final
        # perfect.sample_argv starts with --prompt X; append the rest after draft's prompt
        ref_flags: list[str] = []
        argv = perfect.sample_argv
        skip_next = False
        i = 0
        while i < len(argv):
            if skip_next:
                skip_next = False
                i += 1
                continue
            if argv[i] == "--prompt":
                skip_next = True
                i += 1
                continue
            ref_flags.append(argv[i])
            i += 1
        draft_plan.draft_argv = [
            "--prompt",
            enriched,
            "--num",
            str(draft_plan.num_drafts),
            "--steps",
            str(draft_plan.draft_steps),
            "--width",
            str(draft_plan.draft_width),
            "--height",
            str(draft_plan.draft_height),
            "--seed",
            str(draft_plan.base_seed),
            "--pick-best",
            draft_plan.pick_metric,
        ] + ref_flags
        draft_plan.final_argv_template = [
            "--prompt",
            enriched,
            "--num",
            "1",
            "--steps",
            str(draft_plan.final_steps),
            "--width",
            str(intent.width if intent else draft_plan.final_width),
            "--height",
            str(intent.height if intent else draft_plan.final_height),
        ] + ref_flags
        out.draft = draft_plan.to_dict()
        out.sample_draft_argv = list(draft_plan.draft_argv)
        idx = 0 if chosen_draft_index is None else int(chosen_draft_index)
        out.sample_final_argv = promote_draft_to_final(
            draft_plan, chosen_index=idx, user_picked=chosen_draft_index is not None
        )
    else:
        out.sample_final_argv = list(perfect.sample_argv)

    if critique_image:
        cplan = build_post_gen_critique(critique_image, enriched)
        if critique_answers:
            w = intent.width if intent else 1024
            h = intent.height if intent else 1024
            cplan = critique_answers_to_fix_plan(cplan, critique_answers, work_dir=work / "critique", width=w, height=h)
        out.critique = cplan.to_dict()
        out.notes.extend(cplan.notes)

    save_user_taste(taste, work / "user_taste.json")
    (work / "plan.json").write_text(json.dumps(out.to_dict(), indent=2), encoding="utf-8")
    return out


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
    moodboard_strength: float = 1.0
    device: str = "cuda"


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
            from utils.captioning import caption_image_vlm

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
            from utils.captioning import download_search_hits, search_reference_images

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
                from utils.captioning import caption_image_vlm

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
        from utils.native import native_stack_summary

        notes.append(f"ai_accel: {native_stack_summary()}")
    except Exception as exc:
        notes.append(f"ai_accel inventory failed: {exc}")
    try:
        from utils.modeling.model_paths import (
            default_dinov3_vitl16_path,
            default_dinov3_vits16_path,
            default_siglip2_so400m_path,
            has_local_weights,
        )

        for label, fn in (
            ("DINOv3-L16", default_dinov3_vitl16_path),
            ("DINOv3-S16", default_dinov3_vits16_path),
            ("SigLIP2-SO400M", default_siglip2_so400m_path),
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
        from utils.captioning import extract_control_maps

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


def copilot_sample_argv(
    *,
    ckpt: str,
    plan: CopilotPlan,
    out: str,
    device: str = "cuda",
    extra: Sequence[str] | None = None,
    config: CopilotConfig | None = None,
) -> list[str]:
    from utils.generation.sample_argv import sample_argv

    cfg = config or CopilotConfig()
    values: dict[str, Any] = {
        "ckpt": str(ckpt),
        "prompt": plan.fused_prompt or plan.goal_prompt,
        "out": str(out),
        "device": str(device),
    }
    if plan.moodboard_json:
        values.update(moodboard_json=plan.moodboard_json, moodboard_strength=cfg.moodboard_strength)
    elif plan.moodboard_paths:
        values.update(moodboard_images=",".join(plan.moodboard_paths), moodboard_strength=cfg.moodboard_strength)
    if plan.control_image:
        values.update(control_image=plan.control_image, control_type=plan.control_type or None)
    return [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "sample.py"),
        *sample_argv(values),
        *(extra or ()),
    ]


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

    If ``execute`` and ``ckpt`` are set, shells out to ``sample.py`` with moodboard fusion.
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
        plan.sample_argv = copilot_sample_argv(
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
