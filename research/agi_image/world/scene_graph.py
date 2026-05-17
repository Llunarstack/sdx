from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4


@dataclass(slots=True)
class EntityNode:
    entity_id: str = field(default_factory=lambda: uuid4().hex[:10])
    kind: Literal["person", "object", "animal", "background", "text_block", "unknown"] = "unknown"
    name: str = ""
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    embeddings_ref: Optional[str] = None  # opaque path / key


@dataclass(slots=True)
class RelationEdge:
    """Structured triple for spatial / social / causal links."""

    subj_id: str
    predicate: str
    obj_id: str
    strength: float = 1.0


@dataclass(slots=True)
class SceneGraph:
    entities: Dict[str, EntityNode] = field(default_factory=dict)
    relations: List[RelationEdge] = field(default_factory=list)

    def upsert_entity(self, e: EntityNode) -> None:
        self.entities[e.entity_id] = e

    def export_jsonish(self) -> Dict[str, Any]:
        return {
            "entities": {k: asdict(e) for k, e in self.entities.items()},
            "relations": [asdict(r) for r in self.relations],
        }


__all__ = ["EntityNode", "RelationEdge", "SceneGraph"]
