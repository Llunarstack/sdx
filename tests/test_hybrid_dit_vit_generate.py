from __future__ import annotations

from pathlib import Path

from scripts.tools.ops.hybrid_dit_vit_generate import (
    _candidate_paths,
    anneal_weight,
    consensus_score,
    constraint_consensus_score,
    extract_expected_text,
    fuse_vit_scores,
    maybe_self_correct_prompt,
    next_candidate_budget,
    pareto_front_rows,
    reflective_prompt_update,
    seed_for_iteration,
    signature_novelty,
    uncertainty_score,
)
from utils.prompt.shape_scaffold import compile_shape_scaffold, infer_shape_blueprint


def test_candidate_paths_pattern(tmp_path: Path) -> None:
    out = tmp_path / "result.png"
    got = _candidate_paths(out, 3)
    assert [p.name for p in got] == ["result_cand0.png", "result_cand1.png", "result_cand2.png"]


def test_fuse_vit_scores_normalizes_weights() -> None:
    q = 0.8
    a = 0.2
    s1 = fuse_vit_scores(q, a, quality_weight=11.0, adherence_weight=9.0)
    s2 = fuse_vit_scores(q, a, quality_weight=1.1, adherence_weight=0.9)
    assert abs(s1 - s2) < 1e-9


def test_fuse_vit_scores_zero_weights_falls_back_even() -> None:
    q = 0.9
    a = 0.1
    s = fuse_vit_scores(q, a, quality_weight=0.0, adherence_weight=0.0)
    assert abs(s - 0.5) < 1e-9


def test_consensus_score_penalizes_disagreement() -> None:
    hi_disagree = consensus_score(
        0.95, 0.40, quality_weight=0.55, adherence_weight=0.45, disagreement_penalty=0.2
    )
    lo_disagree = consensus_score(
        0.75, 0.70, quality_weight=0.55, adherence_weight=0.45, disagreement_penalty=0.2
    )
    assert lo_disagree > hi_disagree


def test_seed_for_iteration_stride() -> None:
    assert seed_for_iteration(100, 0, 7) == 100
    assert seed_for_iteration(100, 3, 7) == 121


def test_maybe_self_correct_prompt() -> None:
    base = "portrait of a warrior in moonlight"
    corrected = maybe_self_correct_prompt(base, best_adherence=0.4, threshold=0.63, enable=True)
    assert corrected != base
    assert "precise prompt adherence" in corrected
    unchanged = maybe_self_correct_prompt(base, best_adherence=0.9, threshold=0.63, enable=True)
    assert unchanged == base


def test_infer_shape_blueprint_includes_constraints() -> None:
    bp = infer_shape_blueprint("full body warrior left of dragon, cinematic moonlight")
    assert isinstance(bp.get("constraints"), list)
    assert "shape-first composition" in (bp.get("composition") or [])
    assert (bp.get("actors") or [])


def test_compile_shape_scaffold_outputs_positive_negative() -> None:
    pos, neg, bp = compile_shape_scaffold("1girl holding sword, low angle, neon")
    assert isinstance(bp, dict)
    assert "coherent perspective" in pos.lower()
    assert "warped anatomy" in neg.lower()


def test_extract_expected_text_quotes() -> None:
    assert extract_expected_text('poster with text "HELLO WORLD"') == "HELLO WORLD"
    assert extract_expected_text("poster says 'SALE NOW'") == "SALE NOW"
    assert extract_expected_text("no quoted text here") == ""


def test_pareto_front_rows_filters_dominated() -> None:
    rows = [
        {"id": 1.0, "vit_quality_prob": 0.9, "vit_adherence_score": 0.5},
        {"id": 2.0, "vit_quality_prob": 0.7, "vit_adherence_score": 0.7},
        {"id": 3.0, "vit_quality_prob": 0.6, "vit_adherence_score": 0.6},
    ]
    front = pareto_front_rows(rows, ["vit_quality_prob", "vit_adherence_score"])
    ids = {int(r["id"]) for r in front}
    assert 3 not in ids
    assert 1 in ids and 2 in ids


def test_constraint_consensus_score_adds_constraints() -> None:
    s0 = constraint_consensus_score(
        quality_prob=0.8,
        adherence_score=0.8,
        ocr_score=0.1,
        count_score=0.1,
        saturation_score=0.1,
        quality_weight=0.5,
        adherence_weight=0.5,
        disagreement_penalty=0.1,
        ocr_weight=0.0,
        count_weight=0.0,
        saturation_weight=0.0,
    )
    s1 = constraint_consensus_score(
        quality_prob=0.8,
        adherence_score=0.8,
        ocr_score=0.9,
        count_score=0.9,
        saturation_score=0.9,
        quality_weight=0.5,
        adherence_weight=0.5,
        disagreement_penalty=0.1,
        ocr_weight=0.1,
        count_weight=0.1,
        saturation_weight=0.1,
    )
    assert s1 > s0


def test_next_candidate_budget_increases_when_low() -> None:
    assert next_candidate_budget(4, 0.5, threshold=0.8, step=2, max_num=8) == 6
    assert next_candidate_budget(8, 0.5, threshold=0.8, step=2, max_num=8) == 8
    assert next_candidate_budget(6, 0.9, threshold=0.8, step=2, max_num=8) == 6


def test_reflective_prompt_update_adds_hints() -> None:
    p = "epic scene"
    row = {"vit_adherence_score": 0.3, "count_score": 0.2, "ocr_score": 0.2, "saturation_score": 0.2}
    out = reflective_prompt_update(p, row, enable=True)
    assert out != p
    assert "exact object/person counts" in out


def test_anneal_weight_up_down() -> None:
    base = 0.1
    up0 = anneal_weight(base, 0, 5, mode="up")
    up4 = anneal_weight(base, 4, 5, mode="up")
    dn0 = anneal_weight(base, 0, 5, mode="down")
    dn4 = anneal_weight(base, 4, 5, mode="down")
    assert up4 > up0
    assert dn4 < dn0


def test_uncertainty_score_tracks_confidence() -> None:
    hi = {
        "vit_consensus_score": 0.92,
        "vit_quality_prob": 0.92,
        "vit_adherence_score": 0.91,
        "ocr_score": 0.9,
        "count_score": 0.9,
        "saturation_score": 0.9,
    }
    lo = {
        "vit_consensus_score": 0.42,
        "vit_quality_prob": 0.9,
        "vit_adherence_score": 0.4,
        "ocr_score": 0.3,
        "count_score": 0.2,
        "saturation_score": 0.3,
    }
    assert uncertainty_score(lo) > uncertainty_score(hi)


def test_signature_novelty_increases_with_distance() -> None:
    mem = [(10.0, 10.0, 10.0, 5.0, 5.0, 5.0)]
    near = (12.0, 11.0, 10.0, 5.0, 5.0, 5.0)
    far = (220.0, 210.0, 215.0, 70.0, 68.0, 72.0)
    assert signature_novelty(far, mem) > signature_novelty(near, mem)
