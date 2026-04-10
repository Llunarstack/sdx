"""
Bridge **visual memory** JSON to other book artifacts:

- Minimal **consistency-style** dict (merge with ``--consistency-json``).
- Human-readable **cast sheet** lines for writers / editors.
"""

from __future__ import annotations

from typing import Any, Dict, List


def export_cast_sheet_lines(mem: Any) -> List[str]:
    """Bullet lines summarizing each entity (for README, story bible, or QC)."""
    lines: List[str] = []
    for eid in mem.entity_ids():
        ent = mem.entities.get(eid, {})
        kind = str(ent.get("kind", "character") or "character")
        name = str(ent.get("display_name", "") or eid).strip()
        look = str(ent.get("canonical_look", "") or "").strip()
        costume = str(ent.get("costume_lock", "") or "").strip()
        head = f"- [{kind}] {name} (id={eid})"
        bits = [head]
        if look:
            bits.append(f"  look: {look}")
        if costume:
            bits.append(f"  outfit: {costume}")
        st = ent.get("structure")
        if isinstance(st, dict):
            for k in ("proportions", "default_viewing_angle", "head_to_body_ratio", "scale_notes"):
                v = str(st.get(k, "") or "").strip()
                if v:
                    bits.append(f"  {k}: {v}")
        lines.append("\n".join(bits))
    return lines


def minimal_consistency_dict_from_visual_memory(
    mem: Any,
    *,
    page_index: int = 0,
    safety_mode: str = "",
) -> Dict[str, Any]:
    """
    Build a dict compatible with ``consistency_helpers.positive_block_from_mapping`` **additions**.

    Merge into your hand-authored spec (do not replace entire file blindly): this sets
    ``visual_extra`` and a simple ``character`` string synthesized from the first
    ``kind=character`` entity at *page_index* (including page overrides).
    """
    frag = mem.prompt_fragment_for_page(page_index, safety_mode=safety_mode)
    out: Dict[str, Any] = {"visual_extra": frag}

    for eid in mem.entity_ids():
        eff = mem.effective_entity(eid, page_index)
        if str(eff.get("kind", "character")).lower().strip() != "character":
            continue
        name = str(eff.get("display_name", "") or eid).strip()
        look = str(eff.get("canonical_look", "") or "").strip()
        costume = str(eff.get("costume_lock", "") or "").strip()
        parts = [f"memory-locked lead: {name}"]
        if look:
            parts.append(look)
        if costume:
            parts.append(f"outfit: {costume}")
        out["character"] = ", ".join(parts)
        break

    return out


def merge_consistency_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge with special cases: ``visual_extra`` and ``props`` concatenate."""
    out = dict(base)
    for k, v in extra.items():
        if k == "visual_extra":
            b = str(out.get("visual_extra", "") or "").strip()
            e = str(v or "").strip()
            out["visual_extra"] = ", ".join(x for x in (b, e) if x)
            continue
        if k == "props":
            bp = out.get("props")
            ep = v
            lst: List[Any] = []
            if isinstance(bp, list):
                lst.extend(bp)
            elif isinstance(bp, str) and bp.strip():
                lst.append(bp.strip())
            if isinstance(ep, list):
                lst.extend(ep)
            elif isinstance(ep, str) and ep.strip():
                lst.append(ep.strip())
            if lst:
                out["props"] = lst
            continue
        out[k] = v
    return out
