"""
Style explore planner — invent genomes, compile prompts, optional mutation expansion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .style_genome import StyleGenome, nearest_catalog_style_overlap
from .style_genome_chaos import (
    InventionMode,
    auto_chaos_clauses,
    fuse_genomes,
    hypermutate,
    list_insane_presets,
    preset_genome,
)
from .style_inventor import StyleInventor
from .style_memory import StyleGenomeBank, StyleGenomeRecord


@dataclass
class ExploreCandidate:
    """One render-ready prompt bundle from a genome."""

    genome: StyleGenome
    positive: str
    negative: str
    style: str
    mutation_strategy: str = ""
    catalog_overlap: float = 0.0
    candidate_kind: str = "base"

    def to_manifest_row(self) -> Dict[str, Any]:
        return {
            "caption": self.positive,
            "negative_caption": self.negative,
            "style": self.style,
            "style_genome_id": self.genome.id,
            "style_genome_name": self.genome.name,
            "mutation_strategy": self.mutation_strategy,
            "catalog_overlap": self.catalog_overlap,
            "candidate_kind": self.candidate_kind,
        }


@dataclass
class ExplorePlan:
    """Full explore session output."""

    base_prompt: str
    candidates: List[ExploreCandidate] = field(default_factory=list)
    reasoning: str = ""

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.candidates:
                f.write(json.dumps(row.to_manifest_row(), ensure_ascii=False) + "\n")

    def write_genomes_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [c.genome.to_dict() for c in self.candidates]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def compile_genome_pair(
    genome: StyleGenome,
    base_prompt: str,
    base_negative: str = "",
) -> tuple[str, str, str]:
    pos = genome.compile_positive(base_prompt)
    neg = genome.compile_negative(base_negative)
    style = genome.style_head_string()
    return pos, neg, style


def resolve_style_genome_for_args(args: Any, base_prompt: str) -> Optional[StyleGenome]:
    """
    Load or invent a genome and attach to *args* for PromptStack / sample.py.

    Sets: ``_active_style_genome``, ``style`` (if empty), ``_style_genome_catalog_overlap``.
    """
    genome: Optional[StyleGenome] = None
    path = str(getattr(args, "style_genome_file", "") or "").strip()
    if path:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, list):
            idx = int(getattr(args, "style_genome_index", 0) or 0)
            data = data[idx] if data else {}
        genome = StyleGenome.from_dict(data if isinstance(data, dict) else {})

    invent_n = int(getattr(args, "invent_styles", 0) or 0)
    if genome is None and invent_n > 0:
        inventor = StyleInventor(
            device=str(getattr(args, "style_inventor_device", "cpu") or "cpu"),
            use_qwen=not bool(getattr(args, "no_style_inventor_qwen", False)),
        )
        mode = str(getattr(args, "style_inventor_mode", "normal") or "normal").strip().lower()
        genomes = inventor.invent(
            base_prompt,
            n=invent_n,
            creativity_level=float(getattr(args, "style_inventor_creativity", 0.75) or 0.75),
            seed=int(getattr(args, "seed", 42) or 42),
            invention_mode=mode,  # type: ignore[arg-type]
            chaos_level=float(getattr(args, "style_chaos_level", 0.0) or 0.0),
        )
        preset = str(getattr(args, "style_genome_preset", "") or "").strip()
        if preset:
            pg = preset_genome(preset)
            if pg is not None:
                genomes = [pg] + [g for g in genomes if g.id != pg.id]
        if bool(getattr(args, "style_genome_hypermutate", False)) and genomes:
            genomes = [
                hypermutate(g, intensity=float(getattr(args, "style_chaos_level", 0.85) or 0.85), seed=int(getattr(args, "seed", 42)))
                for g in genomes
            ]
        clauses = str(getattr(args, "prompt_clauses", "") or "")
        auto = auto_chaos_clauses(float(getattr(args, "style_chaos_level", 0) or 0))
        if auto and not clauses:
            setattr(args, "prompt_clauses", ",".join(auto))
        elif auto:
            setattr(args, "prompt_clauses", f"{clauses},{','.join(auto)}")
        idx = int(getattr(args, "style_genome_index", 0) or 0)
        if genomes:
            genome = genomes[min(idx, len(genomes) - 1)]
            setattr(args, "_invented_style_genomes", genomes)

    if genome is None:
        return None

    setattr(args, "_active_style_genome", genome)
    _, overlap = nearest_catalog_style_overlap(genome)
    setattr(args, "_style_genome_catalog_overlap", overlap)

    if not (getattr(args, "style", None) or "").strip():
        setattr(args, "style", genome.style_head_string())

    return genome


class StyleExplorePlanner:
    """Build a manifest of genome × optional prompt mutations."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        use_qwen: bool = True,
    ) -> None:
        self.inventor = StyleInventor(device=device, use_qwen=use_qwen)

    def plan(
        self,
        base_prompt: str,
        *,
        n_genomes: int = 3,
        mutations_per_genome: int = 0,
        base_negative: str = "",
        creativity_level: float = 0.75,
        seed: int = 42,
        mutation_strategies: Optional[Sequence[str]] = None,
        invention_mode: InventionMode = "normal",
        chaos_level: float = 0.0,
        fusion_pairs: bool = False,
        hypermutate_siblings: int = 0,
        preset: str = "",
    ) -> ExplorePlan:
        base_prompt = (base_prompt or "").strip()
        genomes = self.inventor.invent(
            base_prompt,
            n=n_genomes,
            creativity_level=creativity_level,
            seed=seed,
            invention_mode=invention_mode,
            chaos_level=chaos_level,
        )
        if preset:
            pg = preset_genome(preset)
            if pg is not None:
                genomes.insert(0, pg)
        candidates: List[ExploreCandidate] = []
        mutator = None
        if mutations_per_genome > 0:
            try:
                from utils.prompt.prompt_mutation import PromptMutationEngine

                mutator = PromptMutationEngine(
                    n_mutations=mutations_per_genome,
                    strategies=list(mutation_strategies) if mutation_strategies else None,
                    seed=seed,
                )
            except ImportError:
                mutator = None

        for genome in genomes:
            pos, neg, style = compile_genome_pair(genome, base_prompt, base_negative)
            _, overlap = nearest_catalog_style_overlap(genome)
            candidates.append(
                ExploreCandidate(
                    genome=genome,
                    positive=pos,
                    negative=neg,
                    style=style,
                    catalog_overlap=overlap,
                    candidate_kind="base",
                )
            )
            if fusion_pairs and len(genomes) >= 2:
                import random

                r = random.Random(seed)
                others = [g for g in genomes if g.id != genome.id and g.name != genome.name]
                if len(others) >= 1:
                    partner = r.choice(others)
                    chimera = fuse_genomes(genome, partner, ratio=r.random())
                    cpos, cneg, cstyle = compile_genome_pair(chimera, base_prompt, base_negative)
                    _, cov = nearest_catalog_style_overlap(chimera)
                    candidates.append(
                        ExploreCandidate(
                            genome=chimera,
                            positive=cpos,
                            negative=cneg,
                            style=cstyle,
                            catalog_overlap=cov,
                            candidate_kind="chimera",
                        )
                    )
            if hypermutate_siblings > 0:
                for j in range(hypermutate_siblings):
                    sib = hypermutate(genome, intensity=max(0.5, chaos_level, 0.75), seed=seed + j + 3)
                    spos, sneg, sstyle = compile_genome_pair(sib, base_prompt, base_negative)
                    _, sov = nearest_catalog_style_overlap(sib)
                    candidates.append(
                        ExploreCandidate(
                            genome=sib,
                            positive=spos,
                            negative=sneg,
                            style=sstyle,
                            catalog_overlap=sov,
                            candidate_kind="hypermutate",
                        )
                    )
            if mutator is not None:
                for mut in mutator.mutate(pos)[:mutations_per_genome]:
                    mpos, mneg, mstyle = compile_genome_pair(genome, mut.prompt, neg)
                    candidates.append(
                        ExploreCandidate(
                            genome=genome,
                            positive=mpos,
                            negative=mneg,
                            style=mstyle,
                            mutation_strategy=mut.strategy,
                            catalog_overlap=overlap,
                        )
                    )

        return ExplorePlan(
            base_prompt=base_prompt,
            candidates=candidates,
            reasoning=f"Invented {len(genomes)} genomes; {len(candidates)} total candidates.",
        )


def record_explore_winner(
    genome: StyleGenome,
    *,
    score: float,
    prompt: str,
    pick_metric: str = "",
    image_path: str = "",
    bank_path: Optional[Path] = None,
) -> None:
    bank = StyleGenomeBank(bank_path)
    bank.append(
        StyleGenomeRecord(
            genome=genome,
            score=score,
            prompt=prompt,
            pick_metric=pick_metric,
            image_path=image_path,
        )
    )


__all__ = [
    "ExploreCandidate",
    "ExplorePlan",
    "StyleExplorePlanner",
    "compile_genome_pair",
    "list_insane_presets",
    "record_explore_winner",
    "resolve_style_genome_for_args",
]
