"""
On-disk layout for a generated book using the **same DiT-Text checkpoint** as ``sample.py``.

Canonical tree::

    <project_dir>/
      book.json              # metadata + generation manifest entries
      covers/
        front.png
        back.png
      pages/
        page_000.png
        ...

Legacy ``cover/cover.png`` is still written when requested for older tooling.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BOOK_JSON = "book.json"
COVERS_DIR = "covers"
PAGES_DIR = "pages"
LEGACY_COVER_DIR = "cover"

_BOOK_TOP_KEYS = frozenset(
    {
        "version",
        "updated_at",
        "title",
        "ckpt",
        "book_type",
        "model_note",
        "layout",
        "entries",
    }
)


@dataclass
class BookProject:
    """Mutable book project rooted at *root*."""

    root: Path
    title: str = ""
    ckpt: str = ""
    book_type: str = "comic"
    model_note: str = "Uses the same DiT-Text checkpoint as train.py / sample.py (one generative model)."
    entries: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, path: str | Path, *, create: bool = True) -> BookProject:
        root = Path(path).resolve()
        if create:
            root.mkdir(parents=True, exist_ok=True)
            (root / COVERS_DIR).mkdir(exist_ok=True)
            (root / PAGES_DIR).mkdir(exist_ok=True)
        book_json = root / BOOK_JSON
        if book_json.is_file():
            data = json.loads(book_json.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{book_json}: root must be an object")
            return cls.from_dict(data, root=root)
        proj = cls(root=root)
        if create:
            proj.flush()
        return proj

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, root: Path) -> BookProject:
        ent = data.get("entries")
        entries = [e for e in ent if isinstance(e, dict)] if isinstance(ent, list) else []
        return cls(
            root=root,
            title=str(data.get("title") or ""),
            ckpt=str(data.get("ckpt") or ""),
            book_type=str(data.get("book_type") or "comic"),
            model_note=str(data.get("model_note") or cls.model_note),
            entries=entries,
            extra={k: v for k, v in data.items() if k not in _BOOK_TOP_KEYS},
        )

    def cover_path(self, side: str) -> Path:
        side_n = side.strip().lower().replace("-", "_")
        if side_n in ("front", "front_cover", "cover"):
            return self.root / COVERS_DIR / "front.png"
        if side_n in ("back", "back_cover", "rear"):
            return self.root / COVERS_DIR / "back.png"
        raise ValueError(f"Unknown cover side: {side!r} (use front or back)")

    def legacy_cover_path(self) -> Path:
        return self.root / LEGACY_COVER_DIR / "cover.png"

    def page_path(self, index: int) -> Path:
        return self.root / PAGES_DIR / f"page_{int(index):03d}.png"

    def pages_dir(self) -> Path:
        return self.root / PAGES_DIR

    def covers_dir(self) -> Path:
        return self.root / COVERS_DIR

    def add_entry(self, entry: dict[str, Any]) -> None:
        self.entries.append(dict(entry))

    def rel(self, path: Path) -> str:
        """POSIX path relative to the project root (bare file name if outside it)."""
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.name

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "title": self.title,
            "ckpt": self.ckpt,
            "book_type": self.book_type,
            "model_note": self.model_note,
            "layout": {
                "covers": {"front": f"{COVERS_DIR}/front.png", "back": f"{COVERS_DIR}/back.png"},
                "pages": f"{PAGES_DIR}/page_{{index:03d}}.png",
                "legacy_cover": f"{LEGACY_COVER_DIR}/cover.png",
            },
            "entries": self.entries,
        }
        out.update(self.extra)
        return out

    def flush(self) -> Path:
        path = self.root / BOOK_JSON
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def sync_legacy_front_cover(self, front: Path) -> None:
        """Copy front cover to ``cover/cover.png`` for older scripts."""
        if not front.is_file():
            return
        legacy = self.legacy_cover_path()
        legacy.parent.mkdir(exist_ok=True)
        shutil.copyfile(front, legacy)
