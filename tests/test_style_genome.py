"""Tests for StyleGenome invention and explore planner."""

from __future__ import annotations

import json
from pathlib import Path

from utils.prompt.style_explore import StyleExplorePlanner, compile_genome_pair
from utils.prompt.style_genome import StyleGenome, genome_from_json, is_genome_novel_enough
from utils.prompt.style_inventor import StyleInventor
from utils.prompt.style_memory import StyleGenomeBank, StyleGenomeRecord


def test_genome_compile_and_json_roundtrip():
    g = StyleGenome(
        id="test1",
        name="Ash ember wash",
        palette="desaturated umber with ember rim",
        line="dry ink breaks",
        surface="matte gouache on toothy paper",
        camera="low hero angle",
        lighting="single hard key",
        signature="volcanic stillness",
        anti_clone=("not ghibli",),
        negative_fragments=("generic stock photo",),
    )
    pos, neg, style = compile_genome_pair(g, "1girl, forest", "low quality")
    assert "forest" in pos
    assert "ember" in pos.lower() or "gouache" in pos.lower()
    assert "ghibli" in neg.lower()
    assert style

    raw = json.dumps(g.to_dict())
    g2 = genome_from_json(raw)
    assert g2.id == g.id
    assert g2.palette == g.palette


def test_fallback_inventor_produces_novel_genomes():
    inv = StyleInventor(use_qwen=False)
    genomes = inv.invent("samurai at sunset, dramatic", n=3, seed=99, creativity_level=0.8)
    assert len(genomes) == 3
    names = {g.signature for g in genomes}
    assert len(names) >= 2
    assert all(is_genome_novel_enough(g) for g in genomes)


def test_explore_planner_manifest(tmp_path: Path):
    planner = StyleExplorePlanner(use_qwen=False)
    plan = planner.plan(
        "portrait of a scientist in a lab",
        n_genomes=2,
        mutations_per_genome=1,
        seed=1,
    )
    assert len(plan.candidates) >= 2
    manifest = tmp_path / "manifest.jsonl"
    plan.write_manifest(manifest)
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(plan.candidates)
    row = json.loads(lines[0])
    assert "caption" in row and "style" in row


def test_chaos_fusion_hypermutate():
    from utils.prompt.style_genome_chaos import (
        apply_chaos_level,
        fuse_genomes,
        hypermutate,
        invent_insane_batch,
        preset_genome,
    )

    g0 = preset_genome("glitch_cathedral")
    assert g0 is not None
    wild = apply_chaos_level(g0, 0.9)
    assert len(wild.positive_fragments) >= len(g0.positive_fragments)

    batch = invent_insane_batch("cosmic horror library", n=3, mode="insane", chaos_level=0.8)
    assert len(batch) == 3

    chimeras = invent_insane_batch("portrait", n=2, mode="chimera", seed=1)
    assert len(chimeras) == 2
    assert " x " in chimeras[0].name or "Chimera" in chimeras[0].signature

    fused = fuse_genomes(batch[0], batch[1])
    sib = hypermutate(fused, intensity=0.9, seed=99)
    assert sib.id != fused.id
    assert sib.palette


def test_insane_explore_plan():
    from utils.prompt.style_explore import StyleExplorePlanner

    planner = StyleExplorePlanner(use_qwen=False)
    plan = planner.plan(
        "witch in storm",
        n_genomes=2,
        invention_mode="apocalypse",
        chaos_level=0.95,
        fusion_pairs=True,
        hypermutate_siblings=1,
        seed=7,
    )
    kinds = {c.candidate_kind for c in plan.candidates}
    assert "base" in kinds
    assert len(plan.candidates) >= 4


def test_style_genome_bank_roundtrip(tmp_path: Path):
    g = StyleGenome(id="bank1", name="Test bank", palette="cool grey", signature="minimal fog")
    bank = StyleGenomeBank(tmp_path / "bank.jsonl")
    bank.append(StyleGenomeRecord(genome=g, score=0.91, prompt="test"))
    loaded = bank.load()
    assert len(loaded) == 1
    assert loaded[0].genome.id == "bank1"
    assert loaded[0].score == 0.91
