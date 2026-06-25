"""Tests for sample feature helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from utils.generation.comfy_export import sample_args_to_comfy_workflow
from utils.generation.per_region_cads import PerRegionCADSConfig, merge_cads_into_holy_grail
from utils.generation.sample_features import (
    append_benchmark_history,
    export_adherence_heatmap,
    load_character_session,
    save_character_session,
    score_image_heuristic,
)


def test_score_image_heuristic_range():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = 200
    s = score_image_heuristic(img)
    assert 0.0 <= s <= 1.0


def test_character_session_roundtrip(tmp_path: Path):
    p = tmp_path / "sess.json"
    data = {"prompt_additions": "silver hair", "negative_prompt": "extra limbs", "reference_images": ["a.png"]}
    save_character_session(p, data)
    loaded = load_character_session(p)
    assert loaded["prompt_additions"] == "silver hair"
    assert loaded["reference_images"] == ["a.png"]


def test_per_region_cads_merge():
    cfg = PerRegionCADSConfig(region_strengths=(0.4, 0.4))
    out = merge_cads_into_holy_grail({"holy_grail_cads_strength": 0.0}, cfg)
    assert out["holy_grail_cads_strength"] == 0.15


def test_comfy_export_has_sampler_node():
    args = {"prompt": "cat", "steps": 30, "cfg_scale": 7.0, "seed": 1}
    wf = sample_args_to_comfy_workflow(args)
    assert "nodes" in wf
    assert any(n.get("class_type") == "SDXSampler" for n in wf["nodes"].values())


def test_adherence_heatmap_export(tmp_path: Path):

    attn = torch.rand(4, 64, 32)
    out = tmp_path / "heat.png"
    export_adherence_heatmap(attn, "a woman in a red dress", out, latent_h=8, latent_w=8)
    assert out.is_file()


def test_benchmark_history_append(tmp_path: Path):
    lb = tmp_path / "leaderboard.json"
    hist = tmp_path / "history.json"
    lb.write_text(json.dumps({"rows": [{"name": "a", "score": 1.0}]}), encoding="utf-8")
    append_benchmark_history(lb, hist)
    data = json.loads(hist.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert "leaderboard" in data[0]


def test_apply_character_session_mutates_prompt():
    from utils.generation.sample_features import _apply_character_session

    args = SimpleNamespace(
        character_session="",
        prompt="base",
        dissect_refs="",
        negative_prompt="",
    )
    # inline session file via monkeypatch path — use tmp_path
    sess = {"prompt_additions": "blue cape", "reference_images": ["ref.png"], "negative_prompt": "blur"}
    import utils.generation.sample_features as sf

    orig = sf.load_character_session
    sf.load_character_session = lambda _p: sess  # type: ignore
    try:
        args.character_session = "x.json"
        _apply_character_session(args)
        assert "blue cape" in args.prompt
        assert args.dissect_refs == "ref.png"
        assert "blur" in args.negative_prompt
    finally:
        sf.load_character_session = orig
