"""Provenance logging for video segments."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import ProvenanceRecord, SegmentAssignment

__all__ = ["ProvenanceLog", "record_segment_provenance", "write_provenance_json"]


class ProvenanceLog:
    def __init__(self) -> None:
        self.records: List[ProvenanceRecord] = []

    def add(self, record: ProvenanceRecord) -> None:
        self.records.append(record)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "segments": [
                {
                    "segment_index": r.segment_index,
                    "source": r.source,
                    "source_path": r.source_path,
                    "license": r.license,
                    "url": r.url,
                    "operations": list(r.operations),
                    "retrieved_at": r.retrieved_at,
                }
                for r in self.records
            ],
        }


def record_segment_provenance(
    seg: SegmentAssignment,
    *,
    operations: Optional[List[str]] = None,
) -> ProvenanceRecord:
    clip = seg.clip
    return ProvenanceRecord(
        segment_index=seg.shot.index,
        source=str(clip.source.value if clip else "synthetic"),
        source_path=str(clip.path if clip else ""),
        license=str(clip.license if clip else "generated"),
        url=str(clip.url if clip else ""),
        operations=list(operations or []),
        retrieved_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


def write_provenance_json(log: ProvenanceLog, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(log.to_dict(), indent=2), encoding="utf-8")
    return p
