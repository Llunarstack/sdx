"""
Subject–relation–object graph from prompts for layout + regional prompting.

Complements ``innovations/semantics`` (production) with a lightweight research parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class RelationKind(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    HOLDING = "holding"
    WEARING = "wearing"
    INSIDE = "inside"
    NEAR = "near"


@dataclass(frozen=True)
class RelationEdge:
    subject: str
    kind: RelationKind
    object: str
    confidence: float


@dataclass
class SceneRelationGraph:
    entities: List[str] = field(default_factory=list)
    edges: List[RelationEdge] = field(default_factory=list)

    def to_regional_hints(self) -> List[Tuple[str, str]]:
        """Map spatial relations to (entity, anchor) for Omost-style layout."""
        hints: List[Tuple[str, str]] = []
        for e in self.edges:
            if e.kind == RelationKind.LEFT_OF:
                hints.append((e.subject, "left"))
                hints.append((e.object, "right"))
            elif e.kind == RelationKind.RIGHT_OF:
                hints.append((e.subject, "right"))
                hints.append((e.object, "left"))
            elif e.kind == RelationKind.ABOVE:
                hints.append((e.subject, "top"))
            elif e.kind == RelationKind.BELOW:
                hints.append((e.subject, "bottom"))
        return hints


_REL_PATTERNS: Tuple[Tuple[str, RelationKind, str], ...] = (
    (r"(\w+(?:\s+\w+)?)\s+on the left of\s+(\w+(?:\s+\w+)?)", RelationKind.LEFT_OF, "left_of"),
    (r"(\w+(?:\s+\w+)?)\s+to the left of\s+(\w+(?:\s+\w+)?)", RelationKind.LEFT_OF, "left_of"),
    (r"(\w+(?:\s+\w+)?)\s+on the right of\s+(\w+(?:\s+\w+)?)", RelationKind.RIGHT_OF, "right_of"),
    (r"(\w+(?:\s+\w+)?)\s+holding\s+(?:a\s+)?(\w+(?:\s+\w+)?)", RelationKind.HOLDING, "holding"),
    (r"(\w+(?:\s+\w+)?)\s+wearing\s+(?:a\s+)?(\w+(?:\s+\w+)?)", RelationKind.WEARING, "wearing"),
    (r"(\w+(?:\s+\w+)?)\s+inside\s+(?:a\s+)?(\w+(?:\s+\w+)?)", RelationKind.INSIDE, "inside"),
)


class SceneRelationParser:
    def parse(self, prompt: str) -> SceneRelationGraph:
        text = (prompt or "").strip()
        graph = SceneRelationGraph()
        if not text:
            return graph
        for pattern, kind, _ in _REL_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                sub, obj = m.group(1).strip(), m.group(2).strip()
                graph.edges.append(RelationEdge(sub, kind, obj, confidence=0.7))
                for ent in (sub, obj):
                    if ent not in graph.entities:
                        graph.entities.append(ent)
        return graph


__all__ = ["RelationEdge", "RelationKind", "SceneRelationGraph", "SceneRelationParser"]
