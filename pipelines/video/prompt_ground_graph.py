"""
Prompt ground graph — accurate structured understanding of T2V prompts.

Seedance-style adherence starts with binding subjects/attributes/counts/negations
*before* generation. We wrap ``PromptParser`` into a video scene graph and rewrite
prompts so attributes cannot leak and @-refs bind cleanly to roles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "GroundEntity",
    "PromptGroundGraph",
    "parse_prompt_ground",
    "rewrite_prompt_for_adherence",
    "bind_at_mentions",
]


@dataclass(slots=True)
class GroundEntity:
    name: str
    attributes: list[str] = field(default_factory=list)
    count: int = 1
    negated: bool = False
    relations: list[str] = field(default_factory=list)
    bind_ref: str = ""  # @Image1 / element id


@dataclass(slots=True)
class PromptGroundGraph:
    raw: str
    entities: list[GroundEntity] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    negations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    camera: str = ""
    rewritten: str = ""
    negative_extra: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def subject_count(self) -> int:
        return sum(max(1, e.count) for e in self.entities if not e.negated)


_PEOPLE = (
    "woman",
    "man",
    "girl",
    "boy",
    "person",
    "people",
    "child",
    "adult",
    "hero",
    "character",
    "dog",
    "cat",
    "bird",
    "car",
    "robot",
)


def parse_prompt_ground(prompt: str) -> PromptGroundGraph:
    from models.prompt_adherence import PromptParser

    raw = (prompt or "").strip()
    parsed = PromptParser().parse(raw)
    entities: list[GroundEntity] = []
    negations: list[str] = []
    seen: set[str] = set()
    for t in getattr(parsed, "triples", []) or []:
        name = str(getattr(t, "subject", "") or "").strip() or "subject"
        ent = GroundEntity(
            name=name,
            attributes=[str(a) for a in (getattr(t, "attributes", []) or [])],
            count=int(getattr(t, "count", 1) or 1),
            negated=bool(getattr(t, "negated", False)),
            relations=[str(r) for r in (getattr(t, "relations", []) or [])],
        )
        entities.append(ent)
        seen.add(name.lower())
        if ent.negated:
            negations.append(ent.name)
            negations.extend(ent.attributes)
    counts = {str(k): int(v) for k, v in (getattr(parsed, "count_constraints", {}) or {}).items()}

    # People / animate subjects PromptParser may miss (it focuses on clothing/objects).
    words = raw.lower().split()
    attr_words = {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "black",
        "white",
        "brown",
        "tall",
        "short",
        "young",
        "old",
    }
    for i, w in enumerate(words):
        stem = w.rstrip("s.,;:")
        if stem not in _PEOPLE or stem in seen:
            continue
        attrs = [words[j] for j in range(max(0, i - 3), i) if words[j] in attr_words]
        cnt = 1
        if i > 0 and words[i - 1] in ("two", "three", "four", "five"):
            cnt = {"two": 2, "three": 3, "four": 4, "five": 5}[words[i - 1]]
            counts.setdefault(stem, cnt)
        entities.insert(
            0,
            GroundEntity(name=stem, attributes=attrs, count=counts.get(stem, cnt)),
        )
        seen.add(stem)

    # Action verbs (temporal layer)
    actions: list[str] = []
    low = raw.lower()
    for verb in (
        "walk",
        "run",
        "turn",
        "look",
        "speak",
        "sing",
        "dance",
        "sit",
        "stand",
        "open",
        "close",
        "pick",
        "hold",
        "drive",
        "fly",
        "fall",
        "jump",
    ):
        if verb in low or f"{verb}s" in low or f"{verb}ing" in low:
            actions.append(verb)

    camera = ""
    for cam in ("orbit", "dolly", "handheld", "crane", "pan", "close-up", "wide", "tracking"):
        if cam in low:
            camera = cam
            break

    g = PromptGroundGraph(
        raw=raw,
        entities=entities,
        counts=counts,
        negations=negations,
        actions=actions,
        camera=camera,
    )
    g.rewritten, g.negative_extra = rewrite_prompt_for_adherence(g)
    g.notes.append(f"entities={len(entities)}")
    if counts:
        g.notes.append(f"counts={counts}")
    return g


def rewrite_prompt_for_adherence(graph: PromptGroundGraph) -> tuple[str, str]:
    """
    Emit an engineering-style prompt that hard-binds attributes to subjects.

    Example: "a woman in a red dress and a man in a blue suit" stays ordered and
    repeats bindings so the model can't swap colors.
    """
    if not graph.entities and not graph.raw:
        return "cinematic scene", "blurry, deformed"
    bits: list[str] = []
    neg: list[str] = []
    for e in graph.entities:
        if e.negated:
            neg.append(f"{e.name}")
            neg.extend(e.attributes)
            continue
        attr = ", ".join(e.attributes) if e.attributes else ""
        count_w = "" if e.count <= 1 else f"{e.count} "
        if attr:
            # Repeat binding: "red dress on the woman (red, not blue)"
            bound = f"{count_w}{e.name} with {attr} (attributes locked to {e.name} only)"
        else:
            bound = f"{count_w}{e.name}".strip()
        if e.bind_ref:
            bound = f"{bound} bound to {e.bind_ref}"
        if e.relations:
            bound = f"{bound}, {', '.join(e.relations)}"
        bits.append(bound)
    if not bits:
        bits.append(graph.raw)
    if graph.actions:
        bits.append("actions: " + ", ".join(graph.actions))
    if graph.camera:
        bits.append(f"camera: {graph.camera}")
    # Preserve original for coverage of uncovered details
    core = "; ".join(bits)
    if graph.raw and graph.raw.lower() not in core.lower():
        core = f"{core}. Full brief: {graph.raw}"
    for n in graph.negations:
        neg.append(n)
    neg.extend(
        [
            "attribute swap between subjects",
            "wrong object count",
            "ignored negation",
            "identity morph",
            "extra limbs",
        ]
    )
    return core, ", ".join(dict.fromkeys(neg))


def bind_at_mentions(
    prompt: str,
    refs: dict[str, str] | None = None,
    *,
    graph: PromptGroundGraph | None = None,
) -> str:
    """
    Seedance-style @ binding: ``@Image1`` / ``@hero`` → explicit role sentences.

    ``refs`` maps mention → role description, e.g. ``{"Image1": "identity of red-haired woman"}``.
    """
    text = prompt or ""
    mapping = dict(refs or {})
    g = graph or parse_prompt_ground(text)
    # Auto-bind first identity entity to first ImageN if present
    if mapping and g.entities:
        first_key = next(iter(mapping))
        if not g.entities[0].bind_ref:
            g.entities[0].bind_ref = f"@{first_key}"
    out = text
    for key, role in mapping.items():
        token = f"@{key}"
        if token.lower() in out.lower() or key in out:
            clause = f"[{token} = {role}; use only for that role]"
            if clause not in out:
                out = f"{clause} {out}"
        else:
            # Inject if prompt doesn't mention but refs provided
            out = f"[{token} = {role}] {out}"
    rewritten, _ = rewrite_prompt_for_adherence(g)
    # Prefer rewritten structure + @ clauses
    if mapping:
        return f"{out}. Structured: {rewritten}"
    return rewritten
