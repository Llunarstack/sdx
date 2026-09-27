"""Append-only user feedback bus for RSI / preference learning.

Captures likes, dislikes, and pairwise picks on generated images (and video
keyframes), then exports Diffusion-DPO JSONL that feeds the existing
``preference_flywheel`` / ``train_diffusion_dpo`` / ``auto_improve_loop`` path.

Schema (one JSON object per line)::

    {
        "event": "like" | "dislike" | "pair" | "pick" | "generate",
        "prompt": "...",
        "image_path": "...",  # primary asset for like/dislike/generate
        "win_image_path": "...",  # pair / pick
        "lose_image_path": "...",
        "score": 1.0,  # optional user score 0..1
        "ckpt": "...",
        "seed": 0,
        "run_id": "...",
        "media_type": "image" | "video",
        "source": "user",
        "timestamp": 1710000000.0,
        "notes": "",
    }
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "FeedbackEvent",
    "default_feedback_log",
    "append_feedback",
    "iter_feedback",
    "pairs_from_feedback",
    "record_like",
    "record_dislike",
    "record_pair",
    "record_pick",
    "record_generation",
    "export_dpo_jsonl",
    "update_user_taste_from_feedback",
]

EventKind = Literal["like", "dislike", "pair", "pick", "generate"]


def default_feedback_log() -> Path:
    return Path("outputs") / "feedback" / "feedback.jsonl"


@dataclass
class FeedbackEvent:
    event: EventKind
    prompt: str = ""
    image_path: str = ""
    win_image_path: str = ""
    lose_image_path: str = ""
    score: float | None = None
    ckpt: str = ""
    seed: int | None = None
    run_id: str = ""
    media_type: str = "image"
    source: str = "user"
    timestamp: float = field(default_factory=lambda: time.time())
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in d or d[k] in (None, "", []):
                    d[k] = v
        # Drop empty optionals for compact JSONL
        return {k: v for k, v in d.items() if v is not None and v != "" and v != {}}


def append_feedback(
    event: FeedbackEvent | dict[str, Any],
    *,
    log_path: str | Path | None = None,
) -> Path:
    """Append one feedback event. Returns the log path."""
    path = Path(log_path) if log_path else default_feedback_log()
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(event, FeedbackEvent):
        row = event.to_dict()
    else:
        row = dict(event)
        row.setdefault("timestamp", time.time())
        row.setdefault("source", "user")
        row.setdefault("run_id", row.get("run_id") or uuid.uuid4().hex[:12])
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def iter_feedback(log_path: str | Path | None = None) -> Iterable[dict[str, Any]]:
    path = Path(log_path) if log_path else default_feedback_log()
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def record_like(
    image_path: str,
    *,
    prompt: str = "",
    score: float = 1.0,
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    log_path: str | Path | None = None,
    notes: str = "",
) -> Path:
    return append_feedback(
        FeedbackEvent(
            event="like",
            prompt=prompt,
            image_path=str(image_path),
            score=float(score),
            ckpt=ckpt,
            seed=seed,
            run_id=run_id or uuid.uuid4().hex[:12],
            media_type=media_type,
            notes=notes,
        ),
        log_path=log_path,
    )


def record_dislike(
    image_path: str,
    *,
    prompt: str = "",
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    log_path: str | Path | None = None,
    notes: str = "",
) -> Path:
    return append_feedback(
        FeedbackEvent(
            event="dislike",
            prompt=prompt,
            image_path=str(image_path),
            score=0.0,
            ckpt=ckpt,
            seed=seed,
            run_id=run_id or uuid.uuid4().hex[:12],
            media_type=media_type,
            notes=notes,
        ),
        log_path=log_path,
    )


def record_pair(
    win_image_path: str,
    lose_image_path: str,
    *,
    prompt: str = "",
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    log_path: str | Path | None = None,
    notes: str = "",
) -> Path:
    return append_feedback(
        FeedbackEvent(
            event="pair",
            prompt=prompt,
            win_image_path=str(win_image_path),
            lose_image_path=str(lose_image_path),
            image_path=str(win_image_path),
            ckpt=ckpt,
            seed=seed,
            run_id=run_id or uuid.uuid4().hex[:12],
            media_type=media_type,
            notes=notes,
        ),
        log_path=log_path,
    )


def record_pick(
    win_image_path: str,
    lose_paths: list[str],
    *,
    prompt: str = "",
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    log_path: str | Path | None = None,
) -> list[Path]:
    """User picked one winner among candidates → one pair per loser."""
    out: list[Path] = []
    rid = run_id or uuid.uuid4().hex[:12]
    for lose in lose_paths:
        if not lose or str(lose) == str(win_image_path):
            continue
        out.append(
            record_pair(
                win_image_path,
                lose,
                prompt=prompt,
                ckpt=ckpt,
                seed=seed,
                run_id=rid,
                media_type=media_type,
                log_path=log_path,
                notes="pick",
            )
        )
    # Also record a like on the winner for taste updates.
    record_like(
        win_image_path,
        prompt=prompt,
        ckpt=ckpt,
        seed=seed,
        run_id=rid,
        media_type=media_type,
        log_path=log_path,
        notes="pick_winner",
    )
    return out


def record_generation(
    image_path: str,
    *,
    prompt: str = "",
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    candidates: list[str] | None = None,
    log_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Provenance row after sample.py / video keyframe save (not a preference)."""
    return append_feedback(
        FeedbackEvent(
            event="generate",
            prompt=prompt,
            image_path=str(image_path),
            ckpt=ckpt,
            seed=seed,
            run_id=run_id or uuid.uuid4().hex[:12],
            media_type=media_type,
            extra={
                **(extra or {}),
                **({"candidates": list(candidates)} if candidates else {}),
            },
        ),
        log_path=log_path,
    )


def pairs_from_feedback(
    log_path: str | Path | None = None,
    *,
    same_prompt_only: bool = True,
    include_synthetic: bool = True,
    max_pairs_per_prompt: int = 8,
) -> list[dict[str, Any]]:
    """
    Convert feedback events into DPO rows ``{win_image_path, lose_image_path, caption, source}``.

    - Explicit ``pair`` / ``pick`` events → direct pairs (margin 0).
    - Optional synthetic: each like vs dislike with matching prompt.
    """
    likes: list[dict[str, Any]] = []
    dislikes: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    per_prompt: dict[str, int] = {}

    def _budget(prompt: str) -> bool:
        key = prompt or ""
        n = per_prompt.get(key, 0)
        if n >= max_pairs_per_prompt:
            return False
        per_prompt[key] = n + 1
        return True

    for row in iter_feedback(log_path):
        ev = str(row.get("event") or "").lower()
        prompt = str(row.get("prompt") or row.get("caption") or "").strip()
        if ev in ("pair", "pick"):
            win = str(row.get("win_image_path") or row.get("image_path") or "").strip()
            lose = str(row.get("lose_image_path") or "").strip()
            if win and lose and win != lose and _budget(prompt):
                pairs.append(
                    {
                        "win_image_path": win,
                        "lose_image_path": lose,
                        "caption": prompt,
                        "source": "user",
                        "weight": 1.5,
                        "margin": 0.0,
                    }
                )
            continue
        if ev == "like":
            likes.append(row)
        elif ev == "dislike":
            dislikes.append(row)

    if include_synthetic:
        for win in likes:
            wp = str(win.get("prompt") or "").strip()
            wimg = str(win.get("image_path") or "").strip()
            if not wimg:
                continue
            for lose in dislikes:
                lp = str(lose.get("prompt") or "").strip()
                limg = str(lose.get("image_path") or "").strip()
                if not limg or limg == wimg:
                    continue
                if same_prompt_only and wp and lp and wp != lp:
                    continue
                caption = wp or lp
                if not _budget(caption):
                    continue
                pairs.append(
                    {
                        "win_image_path": wimg,
                        "lose_image_path": limg,
                        "caption": caption,
                        "source": "user_synthetic",
                        "weight": 1.2,
                        "margin": 0.0,
                    }
                )
    return pairs


def export_dpo_jsonl(
    out_path: str | Path,
    *,
    log_path: str | Path | None = None,
    same_prompt_only: bool = True,
    include_synthetic: bool = True,
) -> int:
    pairs = pairs_from_feedback(
        log_path,
        same_prompt_only=same_prompt_only,
        include_synthetic=include_synthetic,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(pairs)


def update_user_taste_from_feedback(
    *,
    log_path: str | Path | None = None,
    taste_path: str | Path | None = None,
    max_notes: int = 64,
) -> Path | None:
    """Push recent like prompts into ``UserTaste.likes_notes`` / hates from dislikes."""
    try:
        from utils.generation.user_taste import UserTaste, default_taste_path, load_user_taste, save_user_taste
    except Exception:
        return None
    tp = Path(taste_path) if taste_path else default_taste_path()
    taste = load_user_taste(tp) if tp.is_file() else UserTaste()
    likes_notes = list(taste.likes_notes or [])
    hates = list(taste.hates or [])
    for row in iter_feedback(log_path):
        ev = str(row.get("event") or "").lower()
        prompt = str(row.get("prompt") or "").strip()
        notes = str(row.get("notes") or "").strip()
        if ev == "like" and prompt and prompt not in likes_notes:
            likes_notes.append(prompt[:240])
        if ev == "dislike":
            frag = notes or prompt
            if frag and frag not in hates:
                hates.append(frag[:160])
    taste.likes_notes = likes_notes[-max_notes:]
    taste.hates = hates[-max_notes:]
    taste.updated_at = time.time()
    save_user_taste(taste, tp)
    return tp
