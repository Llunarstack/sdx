"""
Persistent **visual memory** for sequential art (manga, webtoon, graphic novel, US comics,
illustrated books): recurring designs, proportions, camera language, and **page-scoped**
overrides when the user changes a look from a given page onward.

This is **prompt + metadata** memory (text the diffusion stack sees). Pair with
``--character-sheet``, inpaint anchors, and ``consistency_helpers`` for strongest results.

JSON format (``version`` = 1):

- ``book_style``: ``manga`` | ``webtoon`` | ``graphic_novel`` | ``comic_us`` | ``illustration`` | ``other``
- ``global_*``: strings merged into every page (camera, rendering, color script, continuity)
- ``entities``: map id -> character / prop / vehicle / setting records (a list of ``{"id": ...}`` also works)
- ``page_overrides`` on each entity: patches active for ``from_page``..``to_page`` (inclusive, 0-based)
- ``page_patches``: run-level extras for page ranges (wardrobe arc, weather, etc.)
- ``extra_prompt_all_pages``: appended to every interior page (not the cover)
- ``cover``: optional block for cover-only prompt fragments
- ``lettering``: balloon/typography/sfx/orthography locks (see ``book_text_continuity``)
- ``style_mix``: ``{"preset": "manga_comic", "secondary": "comic_us"}`` hybrid idiom fusion
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipelines.book_comic.book_challenging_content import visual_memory_challenge_clause
from pipelines.book_comic.book_style_fusion import fusion_from_cli
from pipelines.book_comic.book_text_continuity import lettering_visual_memory_fragment
from pipelines.book_comic.prompt_lexicon import merge_prompt_fragments

MEMORY_VERSION = 1

BOOK_STYLE_HINTS: dict[str, str] = {
    "manga": (
        "Japanese manga readability: clear silhouette reads, screentone-friendly values, "
        "consistent line weight hierarchy face vs costume"
    ),
    "webtoon": (
        "vertical-scroll webtoon: bold readable shapes, generous negative space, "
        "soft gradients and clean separation between vertical tiers"
    ),
    "graphic_novel": (
        "graphic novel sequential art: painterly cohesion, cinematic framing, stable inking across panels"
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

_STRUCTURE_LABELS = {
    "proportions": "body proportions",
    "head_to_body_ratio": "head-to-body ratio",
    "height_class": "relative height",
    "default_viewing_angle": "preferred camera / viewing angle",
    "scale_notes": "scale vs other cast",
    "silhouette": "silhouette read",
    "pose_language": "typical pose language",
}

_GLOBAL_KEYS = ("global_camera_language", "global_rendering", "global_color_script", "global_continuity")


def _text(value: Any) -> str:
    """Stripped string form of a JSON value; ``None`` becomes ``""`` rather than ``"None"``."""
    return "" if value is None else str(value).strip()


def _labelled(label: str, value: Any) -> str:
    s = _text(value)
    return f"{label}: {s}" if s else ""


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _normalize_entities(root: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = root.get("entities")
    if isinstance(raw, dict):
        return {str(k): dict(v) if isinstance(v, dict) else {} for k, v in raw.items()}
    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            eid = _text(item.get("id"))
            if eid:
                out[eid] = {k: v for k, v in item.items() if k != "id"}
        return out
    return {}


def _structure_clause(structure: Mapping[str, Any]) -> str:
    parts = [_labelled(label, structure.get(key)) for key, label in _STRUCTURE_LABELS.items()]
    parts.append(_text(structure.get("extra")))
    parts = [p for p in parts if p]
    return "structured design lock: " + "; ".join(parts) if parts else ""


def _entity_prompt(eid: str, data: Mapping[str, Any]) -> str:
    kind = _text(data.get("kind") or "character").lower()
    head = ENTITY_KIND_LABEL.get(kind, ENTITY_KIND_LABEL["character"])
    name = _text(data.get("display_name") or eid)
    structure = data.get("structure")
    return merge_prompt_fragments(
        f'{head} "{name}" (id {eid})',
        _labelled("established look", data.get("canonical_look")),
        _structure_clause(structure) if isinstance(structure, dict) else "",
        _labelled("line/rendering", data.get("line_and_rendering")),
        _labelled("locked wardrobe", data.get("costume_lock")),
        _labelled("signature items", data.get("signature_props")),
        _text(data.get("notes")),
    )


def _page_window(ov: Mapping[str, Any]) -> tuple[int, int] | None:
    """Inclusive ``(lo, hi)`` page range of an override; ``None`` when ``from_page`` is unparseable."""
    try:
        lo = int(ov.get("from_page", 0))
    except (TypeError, ValueError):
        return None
    try:
        hi = int(ov.get("to_page", lo))
    except (TypeError, ValueError):
        hi = lo
    return (lo, hi) if lo <= hi else (hi, lo)


def _merge_entity_for_page(base_entity: Mapping[str, Any], page_index: int) -> dict[str, Any]:
    """Apply every override covering *page_index*, earliest ``from_page`` first so later arcs win."""
    ent = copy.deepcopy(dict(base_entity))
    overrides = ent.get("page_overrides")
    if not isinstance(overrides, list):
        return ent
    active: list[tuple[int, dict[str, Any]]] = []
    for ov in overrides:
        window = _page_window(ov) if isinstance(ov, dict) else None
        if window is not None and window[0] <= page_index <= window[1]:
            active.append((window[0], ov))
    active.sort(key=lambda t: t[0])
    for _, ov in active:
        patch = ov.get("patch")
        if isinstance(patch, dict):
            ent = _deep_merge(ent, patch)
    return ent


@dataclass(slots=True)
class BookVisualMemory:
    """Loaded visual-memory document with helpers for prompts and updates."""

    root: dict[str, Any]
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> BookVisualMemory:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("visual memory JSON root must be an object")
        ver = int(raw.get("version", MEMORY_VERSION))
        if ver != MEMORY_VERSION:
            raise ValueError(f"unsupported visual memory version {ver!r} (expected {MEMORY_VERSION})")
        ent = _normalize_entities(raw)
        root = dict(raw)
        root["entities"] = ent
        return cls(root=root, entities=ent)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.root, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def entity_ids(self) -> list[str]:
        return sorted(self.entities)

    def effective_entity(self, entity_id: str, page_index: int) -> dict[str, Any]:
        if entity_id not in self.entities:
            raise KeyError(f"unknown entity {entity_id!r}")
        return _merge_entity_for_page(self.entities[entity_id], page_index)

    def apply_entity_page_patch(
        self,
        entity_id: str,
        *,
        from_page: int,
        to_page: int | None = None,
        patch: Mapping[str, Any] | None = None,
    ) -> None:
        """Remember a design change from *from_page* (to *to_page*, inclusive) onward.

        *patch* is deep-merged into the entity for prompts on pages in range; an unknown
        *entity_id* is created as a character.
        """
        ent = self.entities.setdefault(entity_id, {"kind": "character", "display_name": entity_id})
        overrides = ent.get("page_overrides")
        if not isinstance(overrides, list):
            overrides = ent["page_overrides"] = []
        overrides.append(
            {
                "from_page": int(from_page),
                "to_page": int(from_page if to_page is None else to_page),
                "patch": dict(patch or {}),
            }
        )
        self.root["entities"] = self.entities

    def add_page_patch(
        self,
        *,
        from_page: int,
        to_page: int | None = None,
        extra_prompt: str = "",
    ) -> None:
        s = (extra_prompt or "").strip()
        if not s:
            return
        patches = self.root.get("page_patches")
        if not isinstance(patches, list):
            patches = self.root["page_patches"] = []
        patches.append(
            {"from_page": int(from_page), "to_page": int(from_page if to_page is None else to_page), "extra_prompt": s}
        )

    def _global_fragment(self) -> str:
        style = _text(self.root.get("book_style") or "manga").lower()
        hint = BOOK_STYLE_HINTS.get(style, "")
        return merge_prompt_fragments(
            f"book visual language ({style}): {hint}" if hint else "",
            *(_text(self.root.get(key)) for key in _GLOBAL_KEYS),
        )

    def _lettering_fragment(self) -> str:
        letter = self.root.get("lettering")
        return lettering_visual_memory_fragment(letter) if isinstance(letter, dict) else ""

    def _style_mix_fragment(self) -> str:
        sm = self.root.get("style_mix")
        if not isinstance(sm, dict):
            return ""
        return fusion_from_cli(
            preset=_text(sm.get("preset")),
            secondary=_text(sm.get("secondary")),
            primary_book_style=str(self.root.get("book_style") or "manga"),
        )

    def _page_patch_fragments(self, page_index: int) -> str:
        patches = self.root.get("page_patches")
        if not isinstance(patches, list):
            return ""
        frags: list[str] = []
        for p in patches:
            window = _page_window(p) if isinstance(p, dict) else None
            if window is not None and window[0] <= page_index <= window[1]:
                frags.append(_text(p.get("extra_prompt")))
        return merge_prompt_fragments(*frags)

    def _entities_fragment(self, page_index: int | None) -> str:
        """Entity prompts for *page_index*; ``None`` means the cover (canonical design, no overrides)."""
        parts: list[str] = []
        for eid in sorted(self.entities):
            base = self.entities[eid]
            if page_index is None:
                merged = {k: v for k, v in base.items() if k != "page_overrides"}
            else:
                merged = _merge_entity_for_page(base, page_index)
            parts.append(_entity_prompt(eid, merged))
        return merge_prompt_fragments(*parts)

    def _compose(self, entities: str, *tail: str) -> str:
        return merge_prompt_fragments(
            self._global_fragment(),
            entities,
            visual_memory_challenge_clause(self.root),
            self._lettering_fragment(),
            self._style_mix_fragment(),
            _text(self.root.get("user_style_anchor")),
            *tail,
        )

    def prompt_fragment_for_cover(self) -> str:
        cov = self.root.get("cover")
        cover_bits = (
            (_text(cov.get("extra_prompt")), _text(cov.get("entity_treatment"))) if isinstance(cov, dict) else ()
        )
        return self._compose(self._entities_fragment(None), *cover_bits)

    def prompt_fragment_for_page(self, page_index: int) -> str:
        """0-based page index; merges globals, entities (with overrides), and run page patches."""
        return self._compose(
            self._entities_fragment(page_index),
            self._page_patch_fragments(page_index),
            _text(self.root.get("extra_prompt_all_pages")),
        )


def load_visual_memory(path: str | Path) -> BookVisualMemory:
    """Load from JSON path (see module docstring)."""
    return BookVisualMemory.load(path)
