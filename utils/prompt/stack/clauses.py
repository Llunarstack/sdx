"""Composable intent clauses — registry of named prompt fragments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from .tokens import append_unique


@dataclass(frozen=True)
class IntentClause:
    """Named bundle of positive/negative tags with optional activation predicate."""

    name: str
    positive: Sequence[str] = ()
    negative: Sequence[str] = ()
    description: str = ""

    def apply_positive(self, text: str) -> str:
        return append_unique(text, self.positive)

    def apply_negative(self, text: str) -> str:
        return append_unique(text, self.negative)


# Built-in clauses for stack extensions and book/visual-memory hooks.
CLAUSE_REGISTRY: Dict[str, IntentClause] = {
    "uncensored.fidelity": IntentClause(
        name="uncensored.fidelity",
        positive=(
            "faithful to prompt description",
            "no arbitrary censorship",
            "anatomically coherent when nude",
        ),
        negative=(
            "random censor bar",
            "lens censorship blur",
            "sanitized rewrite of scene",
            "modesty costume insertion",
        ),
        description="Narrative fidelity for uncensored model runs.",
    ),
    "hands.stable": IntentClause(
        name="hands.stable",
        positive=("detailed hands", "correct finger count", "natural hand pose"),
        negative=("extra fingers", "fused fingers", "malformed hands", "missing fingers"),
        description="Hand stability booster.",
    ),
    "composition.single": IntentClause(
        name="composition.single",
        positive=("solo", "single subject", "one person"),
        negative=("duplicate person", "cloned face", "twin duplicate"),
        description="Single-subject guardrails.",
    ),
    "quality.micro": IntentClause(
        name="quality.micro",
        positive=(
            "fine texture detail",
            "material-accurate surface",
            "clean edges",
            "subtle micro-contrast",
        ),
        negative=("mushy detail", "plastic texture", "smudged fine detail"),
        description="Micro-detail quality clause.",
    ),
    "style.chaos": IntentClause(
        name="style.chaos",
        positive=(
            "visually unforgettable",
            "deliberately uncanny art direction",
            "impossible material honesty",
            "aggressive compositional tension",
        ),
        negative=(
            "forgettable average",
            "template AI look",
            "instagram filter pack",
            "boring safe composition",
            "generic stock photo",
        ),
        description="Maximum originality / anti-slop clause for style explore.",
    ),
    "style.surreal": IntentClause(
        name="style.surreal",
        positive=(
            "dream logic scale",
            "liminal atmosphere",
            "synesthetic color mood",
            "designed accident framing",
        ),
        negative=("literal boring snapshot", "flat realism only", "explainable mundane scene"),
        description="Surreal / liminal style explore booster.",
    ),
    "style.glitch": IntentClause(
        name="style.glitch",
        positive=(
            "datamoshed accents",
            "chromatic aberration design",
            "CRT phosphor aesthetic",
            "intentional compression artifact beauty",
        ),
        negative=("clean 4k digital", "sterile corporate render", "perfect smooth gradients"),
        description="Digital corruption / glitch art clause.",
    ),
    "style.apocalypse": IntentClause(
        name="style.apocalypse",
        positive=(
            "epic scale ruin",
            "dust-choked atmosphere",
            "weathered mythic weight",
            "particulate air density",
        ),
        negative=("cheerful picnic", "pristine modern city", "empty greybox void"),
        description="Post-collapse epic atmosphere clause.",
    ),
}


def apply_clauses(
    positive: str,
    negative: str,
    clause_names: Sequence[str],
    *,
    registry: Optional[Mapping[str, IntentClause]] = None,
) -> tuple[str, str]:
    reg = registry or CLAUSE_REGISTRY
    p, n = positive, negative
    for name in clause_names:
        clause = reg.get(str(name).strip())
        if clause is None:
            continue
        p = clause.apply_positive(p)
        n = clause.apply_negative(n)
    return p, n


def list_clauses() -> List[str]:
    return sorted(CLAUSE_REGISTRY.keys())
