"""
Superior pass — one orchestrator for all competitor-pain post repairs.

Keeps segment_processor thin: run permanence → identity → hands → logos →
contact → secondary → occlusion → shimmer → shutter → score axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["SuperiorPassResult", "run_superior_pass"]


@dataclass(slots=True)
class SuperiorPassResult:
    ops: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


def run_superior_pass(
    frame_paths: list[Path],
    opts: Any,
    *,
    prompt: str = "",
    anchor: str = "",
    identity_refs: list[str] | None = None,
) -> SuperiorPassResult:
    """Apply enabled superiority repairs in a stable order; collect scores."""
    out = SuperiorPassResult()
    paths = list(frame_paths)
    if len(paths) < 2:
        return out

    if bool(getattr(opts, "permanence_repair", True)):
        from .permanence import apply_permanence_pass

        r = apply_permanence_pass(
            paths, repair=True, strength=float(getattr(opts, "permanence_strength", 0.55) or 0.55)
        )
        if r.repaired:
            out.ops.append(f"permanence_repair:{r.repaired}")
        out.scores["permanence"] = float(r.score)
        out.ops.append(f"permanence_score:{r.score:.2f}")

    refs = list(identity_refs or getattr(opts, "identity_refs", ()) or ())
    if bool(getattr(opts, "identity_bind", True)):
        from .identity_bind import apply_identity_bind

        r = apply_identity_bind(
            paths,
            anchor=anchor or None,
            refs=refs or None,
            strength=float(getattr(opts, "identity_bind_strength", 0.45) or 0.45),
        )
        if r.repaired:
            out.ops.append(f"identity_bind:{r.repaired}")
        out.scores["identity"] = float(r.score)
        out.ops.append(f"identity_score:{r.score:.2f}")

    if bool(getattr(opts, "extremity_lock", True)):
        from .extremity_lock import apply_extremity_lock

        r = apply_extremity_lock(paths, strength=float(getattr(opts, "extremity_strength", 0.50) or 0.50))
        if r.repaired:
            out.ops.append(f"extremity_lock:{r.repaired}")
        out.scores["extremity"] = float(r.score)

    glyph_on = bool(getattr(opts, "glyph_lock", False)) or str(getattr(opts, "motion_grammar", "")) == "product"
    if glyph_on:
        from .glyph_lock import apply_glyph_lock

        r = apply_glyph_lock(paths, strength=float(getattr(opts, "glyph_strength", 0.70) or 0.70))
        if r.repaired:
            out.ops.append(f"glyph_lock:{r.repaired}")
        out.scores["glyph"] = float(r.score)

    if bool(getattr(opts, "contact_ground", True)):
        from .contact_ground import apply_contact_ground

        r = apply_contact_ground(paths, strength=float(getattr(opts, "contact_strength", 0.55) or 0.55))
        if r.repaired:
            out.ops.append(f"contact_ground:{r.repaired}")
        out.scores["contact"] = float(r.score)

    if bool(getattr(opts, "secondary_track", True)):
        from .secondary_track import apply_secondary_track

        r = apply_secondary_track(paths, strength=float(getattr(opts, "secondary_strength", 0.55) or 0.55))
        if r.repaired:
            out.ops.append(f"secondary_track:{r.repaired}")
        out.scores["secondary"] = float(r.score)

    if bool(getattr(opts, "occlusion_resolve", False)):
        from .occlusion_resolve import apply_occlusion_resolve

        r = apply_occlusion_resolve(paths, strength=float(getattr(opts, "occlusion_strength", 0.40) or 0.40))
        if r.repaired:
            out.ops.append(f"occlusion_resolve:{r.repaired}")
        out.scores["occlusion"] = float(r.score)

    if bool(getattr(opts, "hf_deshimmer", True)):
        from .hf_shimmer import apply_hf_deshimmer

        r = apply_hf_deshimmer(paths, strength=float(getattr(opts, "shimmer_strength", 0.55) or 0.55))
        if r.repaired:
            out.ops.append(f"hf_deshimmer:{r.repaired}")
        out.scores["shimmer"] = float(r.score)

    mg = str(getattr(opts, "motion_grammar", "") or "")
    shutter_on = bool(getattr(opts, "motion_shutter", False)) or mg in ("film", "realistic")
    if shutter_on and len(paths) >= 3:
        from .motion_shutter import apply_motion_shutter

        r = apply_motion_shutter(paths, amount=float(getattr(opts, "shutter_amount", 0.45) or 0.45))
        if r.applied:
            out.ops.append(f"motion_shutter:{r.applied}")
        out.scores["shutter"] = float(r.score)

    if bool(getattr(opts, "physics_gate", True)) and len(paths) >= 3:
        from .physics_gate import score_physics_invariance

        r = score_physics_invariance(paths)
        out.scores["physics"] = float(r.score)
        out.ops.append(f"physics_score:{r.score:.2f}")
        out.ops.extend(r.notes[:3])

    if bool(getattr(opts, "count_bind", True)):
        from .count_binder import score_count_stability

        r = score_count_stability(paths, prompt=prompt)
        out.scores["count"] = float(r.score)
        if r.notes:
            out.ops.extend(r.notes[:2])

    return out
