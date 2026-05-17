"""Persistent bank of successful invented style genomes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .style_genome import StyleGenome


def _default_bank_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "style_genomes" / "bank.jsonl"


@dataclass
class StyleGenomeRecord:
    genome: StyleGenome
    score: float = 0.0
    prompt: str = ""
    pick_metric: str = ""
    image_path: str = ""
    saved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "score": self.score,
            "prompt": self.prompt,
            "pick_metric": self.pick_metric,
            "image_path": self.image_path,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StyleGenomeRecord:
        return cls(
            genome=StyleGenome.from_dict(dict(data.get("genome") or {})),
            score=float(data.get("score") or 0.0),
            prompt=str(data.get("prompt") or ""),
            pick_metric=str(data.get("pick_metric") or ""),
            image_path=str(data.get("image_path") or ""),
            saved_at=str(data.get("saved_at") or ""),
        )


class StyleGenomeBank:
    """Append-only JSONL store of high-scoring genomes."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else _default_bank_path()

    def load(self, limit: int = 200) -> List[StyleGenomeRecord]:
        if not self.path.is_file():
            return []
        rows: List[StyleGenomeRecord] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(StyleGenomeRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return rows[-limit:]

    def append(self, record: StyleGenomeRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def top_genomes(self, k: int = 5) -> List[StyleGenome]:
        rows = sorted(self.load(), key=lambda r: r.score, reverse=True)
        seen: set[str] = set()
        out: List[StyleGenome] = []
        for row in rows:
            if row.genome.id in seen:
                continue
            seen.add(row.genome.id)
            out.append(row.genome)
            if len(out) >= k:
                break
        return out


__all__ = ["StyleGenomeBank", "StyleGenomeRecord"]
