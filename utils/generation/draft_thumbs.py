"""Cheap draft thumbnails → user pick → full-quality sample argv."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "DraftThumbPlan",
    "plan_draft_thumbnails",
    "promote_draft_to_final",
]


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
            from utils.training.feedback_bus import record_pick

            win = draft_paths[idx] if idx < len(draft_paths) else draft_paths[0]
            losers = [p for i, p in enumerate(draft_paths) if i != idx]
            record_pick(win, losers, prompt=plan.prompt, seed=seed)
        except Exception:
            pass
    note = "user-picked" if user_picked else "auto-pick-best index"
    plan.notes.append(f"final seed={seed} ({note}, draft[{idx}])")
    return argv
