"""
Persistent **visual memory** for sequential art (manga, webtoon, graphic novel, US comics,
illustrated books): recurring designs, proportions, camera language, and **page-scoped**
overrides when the user changes a look from a given page onward.

This is **prompt + metadata** memory (text the diffusion stack sees). Pair with
``--character-sheet``, inpaint anchors, and ``consistency_helpers`` for strongest results.

JSON format (``version`` = 1):

- ``book_style``: ``manga`` | ``webtoon`` | ``graphic_novel`` | ``comic_us`` | ``illustration`` | ``other``
- ``global_*``: strings merged into every page (camera, rendering, negative hints as positives)
- ``entities``: map id -> character / prop / vehicle / setting records
- ``page_overrides`` on each entity: patches active for ``from_page``..``to_page`` (inclusive, 0-based)
- ``page_patches``: run-level extras for page ranges (wardrobe arc, weather, etc.)
- ``cover``: optional block for cover-only prompt fragments
- ``lettering``: balloon/typography/sfx/orthography locks (see ``book_text_continuity``)
- ``style_mix``: ``{{"preset": "manga_comic", "secondary": "comic_us"}}`` hybrid idiom fusion
- ``user_style_anchor``: freeform user aesthetic repeated every page (combines with book pipeline style)
- ``content_rating``: optional ``mature`` / ``unrestricted`` / ``adult`` / ``nsfw`` (unlocks mature narrative-fidelity hints when paired with pipeline NSFW)
- ``book_challenge_pack``: same names as ``--book-challenge-pack`` in ``generate_book.py``
- ``challenge_tags``: list of keys from ``book_challenging_content.CHALLENGE_TAG_FRAGMENTS``
- ``challenging_content``: object ``{pack, tags, extra}`` (see ``book_challenging_content``)
- ``weird_character_notes`` / ``unusual_character_notes``: lock strange OC silhouettes across pages
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from pipelines.book_comic.book_challenging_content import visual_memory_challenge_clause
from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

MEMORY_VERSION = 1

BOOK_STYLE_HINTS: Dict[str, str] = {
    "manga": (
        "Japanese manga readability: clear silhouette reads, screentone-friendly values, "
        "consistent line weight hierarchy face vs costume"
    ),
    "webtoon": (
        "vertical-scroll webtoon: bold readable shapes, generous negative space, "
        "soft gradients and clean separation between vertical tiers"
    ),
    "graphic_novel": (
        "graphic novel sequential art: painterly cohesion, cinematic framing, stable inking "
        "across panels"
    ),
    "comic_us": (
        "American comic style: confident figure drawing, bold holds, dynamic foreshortening, "
        "consistent costume construction"
    ),
    "illustration": (
        "illustrated book plates: single-image clarity, rich materials, stable character read "
        "when repeated across chapters"
    ),
    "other": "",
}


ENTITY_KIND_LABEL = {
    "character": "recurring character",
    "prop": "recurring prop/object",
    "object": "recurring prop/object",
    "vehicle": "recurring vehicle",
    "setting": "recurring setting element",
}


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _normalize_entities(root: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = root.get("entities")
    if isinstance(raw, dict):
        return {str(k): dict(v) if isinstance(v, dict) else {} for k, v in raw.items()}
    if isinstance(raw, list):
        out: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("id", "")).strip()
            if not eid:
                continue
            body = {k: v for k, v in item.items() if k != "id"}
            out[eid] = body
        return out
    return {}


def _structure_clause(structure: Mapping[str, Any]) -> str:
    if not structure:
        return ""
    parts: List[str] = []
    mapping = {
        "proportions": "body proportions",
        "head_to_body_ratio": "head-to-body ratio",
        "height_class": "relative height",
        "default_viewing_angle": "preferred camera / viewing angle",
        "scale_notes": "scale vs other cast",
        "silhouette": "silhouette read",
        "pose_language": "typical pose language",
    }
    for key, label in mapping.items():
        val = structure.get(key)
        s = str(val).strip() if val is not None else ""
        if s:
            parts.append(f"{label}: {s}")
    extra = str(structure.get("extra", "")).strip()
    if extra:
        parts.append(extra)
    if not parts:
        return ""
    return "structured design lock: " + "; ".join(parts)


def _entity_prompt(eid: str, data: Mapping[str, Any]) -> str:
    kind = str(data.get("kind", "character") or "character").lower().strip()
    head = ENTITY_KIND_LABEL.get(kind, ENTITY_KIND_LABEL["character"])
    name = str(data.get("display_name", "") or eid).strip()
    label = f'{head} "{name}" (id {eid})'
    bits: List[str] = [label]

    look = str(data.get("canonical_look", "")).strip()
    if look:
        bits.append(f"established look: {look}")

    st = data.get("structure")
    if isinstance(st, dict):
        sc = _structure_clause(st)
        if sc:
            bits.append(sc)

    line = str(data.get("line_and_rendering", "")).strip()
    if line:
        bits.append(f"line/rendering: {line}")

    costume = str(data.get("costume_lock", "")).strip()
    if costume:
        bits.append(f"locked wardrobe: {costume}")

    sig = str(data.get("signature_props", "")).strip()
    if sig:
        bits.append(f"signature items: {sig}")

    notes = str(data.get("notes", "")).strip()
    if notes:
        bits.append(notes)

    return merge_prompt_fragments(*bits)


def _page_window(ov: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    try:
        lo = int(ov.get("from_page", 0))
    except (TypeError, ValueError):
        return None
    hi_raw = ov.get("to_page", lo)
    try:
        hi = int(hi_raw)
    except (TypeError, ValueError):
        hi = lo
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _applies(page_index: int, lo: int, hi: int) -> bool:
    return lo <= page_index <= hi


def _merge_entity_for_page(base_entity: Mapping[str, Any], page_index: int) -> Dict[str, Any]:
    ent = copy.deepcopy(dict(base_entity))
    overrides = ent.get("page_overrides")
    if not isinstance(overrides, list):
        return ent
    ordered = []
    for ov in overrides:
        if not isinstance(ov, dict):
            continue
        w = _page_window(ov)
        if w is None:
            continue
        lo, hi = w
        if _applies(page_index, lo, hi):
            ordered.append((lo, ov))
    ordered.sort(key=lambda t: t[0])
    for _, ov in ordered:
        patch = ov.get("patch")
        if isinstance(patch, dict):
            ent = _deep_merge(ent, patch)
    return ent


def _merge_entity_for_cover(base_entity: Mapping[str, Any]) -> Dict[str, Any]:
    """Cover: canonical design only (no page_overrides)."""
    ent = copy.deepcopy(dict(base_entity))
    ent.pop("page_overrides", None)
    return ent


@dataclass
class BookVisualMemory:
    """Loaded visual-memory document with helpers for prompts and updates."""

    root: Dict[str, Any]
    entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Union[str, Path]) -> BookVisualMemory:
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("visual memory JSON root must be an object")
        ver = int(raw.get("version", MEMORY_VERSION))
        if ver != MEMORY_VERSION:
            raise ValueError(f"unsupported visual memory version {ver!r} (expected {MEMORY_VERSION})")
        ent = _normalize_entities(raw)
        root = dict(raw)
        root["entities"] = ent
        return cls(root=root, entities=ent)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.root, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def entity_ids(self) -> List[str]:
        return sorted(self.entities.keys())

    def effective_entity(self, entity_id: str, page_index: int) -> Dict[str, Any]:
        if entity_id not in self.entities:
            raise KeyError(f"unknown entity {entity_id!r}")
        return _merge_entity_for_page(self.entities[entity_id], page_index)

    def apply_entity_page_patch(
        self,
        entity_id: str,
        *,
        from_page: int,
        to_page: Optional[int] = None,
        patch: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Append a page-range patch (user \"remember this change from page N\").

        *patch* is deep-merged into the entity for prompts on pages in range.
        """
        if entity_id not in self.entities:
            self.entities[entity_id] = {"kind": "character", "display_name": entity_id}
        ent = self.entities[entity_id]
        lst = ent.get("page_overrides")
        if not isinstance(lst, list):
            lst = []
            ent["page_overrides"] = lst
        hi = int(to_page if to_page is not None else from_page)
        lst.append(
            {
                "from_page": int(from_page),
                "to_page": hi,
                "patch": dict(patch or {}),
            }
        )
        self.root["entities"] = self.entities

    def add_page_patch(
        self,
        *,
        from_page: int,
        to_page: Optional[int] = None,
        extra_prompt: str = "",
    ) -> None:
        s = (extra_prompt or "").strip()
        if not s:
            return
        patches = self.root.get("page_patches")
        if not isinstance(patches, list):
            patches = []
            self.root["page_patches"] = patches
        hi = int(to_page if to_page is not None else from_page)
        patches.append({"from_page": int(from_page), "to_page": hi, "extra_prompt": s})

    def _global_fragment(self) -> str:
        style = str(self.root.get("book_style", "manga") or "manga").lower().strip()
        hint = BOOK_STYLE_HINTS.get(style, "")
        bits: List[str] = []
        if hint:
            bits.append(f"book visual language ({style}): {hint}")
        for key in (
            "global_camera_language",
            "global_rendering",
            "global_color_script",
            "global_continuity",
        ):
            s = str(self.root.get(key, "") or "").strip()
            if s:
                bits.append(s)
        return merge_prompt_fragments(*bits)

    def _lettering_fragment(self) -> str:
        letter = self.root.get("lettering")
        if isinstance(letter, dict):
            from pipelines.book_comic.book_text_continuity import lettering_visual_memory_fragment

            return lettering_visual_memory_fragment(letter)
        return ""

    def _style_mix_fragment(self) -> str:
        sm = self.root.get("style_mix")
        if not isinstance(sm, dict):
            return ""
        preset = str(sm.get("preset", "") or "").strip()
        sec = str(sm.get("secondary", "") or "").strip()
        from pipelines.book_comic.book_style_fusion import fusion_from_cli

        prim = str(self.root.get("book_style", "manga") or "manga")
        return fusion_from_cli(preset=preset, secondary=sec, primary_book_style=prim)

    def _user_style_anchor_fragment(self) -> str:
        return str(self.root.get("user_style_anchor", "") or "").strip()

    def _page_patch_fragments(self, page_index: int) -> str:
        patches = self.root.get("page_patches")
        if not isinstance(patches, list):
            return ""
        frags: List[str] = []
        for p in patches:
            if not isinstance(p, dict):
                continue
            w = _page_window(p)
            if w is None:
                continue
            lo, hi = w
            if not _applies(page_index, lo, hi):
                continue
            ex = str(p.get("extra_prompt", "")).strip()
            if ex:
                frags.append(ex)
        return merge_prompt_fragments(*frags)

    def _entities_fragment_for_page(self, page_index: int, *, for_cover: bool) -> str:
        if not self.entities:
            return ""
        parts: List[str] = []
        for eid in sorted(self.entities.keys()):
            base = self.entities[eid]
            merged = _merge_entity_for_cover(base) if for_cover else _merge_entity_for_page(base, page_index)
            parts.append(_entity_prompt(eid, merged))
        return merge_prompt_fragments(*parts)

    def prompt_fragment_for_cover(self, *, safety_mode: str = "") -> str:
        bits: List[str] = []
        cov = self.root.get("cover")
        if isinstance(cov, dict):
            ex = str(cov.get("extra_prompt", "")).strip()
            if ex:
                bits.append(ex)
            tr = str(cov.get("entity_treatment", "")).strip()
            if tr:
                bits.append(tr)
        g = self._global_fragment()
        ent = self._entities_fragment_for_page(0, for_cover=True)
        ch = visual_memory_challenge_clause(self.root, safety_mode=safety_mode)
        return merge_prompt_fragments(
            g,
            ent,
            ch,
            self._lettering_fragment(),
            self._style_mix_fragment(),
            self._user_style_anchor_fragment(),
            *bits,
        )

    def prompt_fragment_for_page(self, page_index: int, *, safety_mode: str = "") -> str:
        """0-based page index; merges globals, entities (with overrides), and run page patches."""
        g = self._global_fragment()
        ent = self._entities_fragment_for_page(page_index, for_cover=False)
        pp = self._page_patch_fragments(page_index)
        tail = str(self.root.get("extra_prompt_all_pages", "")).strip()
        ch = visual_memory_challenge_clause(self.root, safety_mode=safety_mode)
        return merge_prompt_fragments(
            g,
            ent,
            ch,
            self._lettering_fragment(),
            self._style_mix_fragment(),
            self._user_style_anchor_fragment(),
            pp,
            tail,
        )


def load_visual_memory(path: Union[str, Path]) -> BookVisualMemory:
    """Load from JSON path (see module docstring)."""
    return BookVisualMemory.load(path)
