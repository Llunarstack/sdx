"""End-to-end wiring tests for Invention Lab integration points."""

from __future__ import annotations

import argparse

from PIL import Image
from utils.generation.inventions.wire import (
    apply_cast_and_corpus_to_namespace,
    apply_invention_to_namespace,
    prepare_invention_runtime,
)
from utils.generation.perfect_gen import build_sample_argv
from utils.generation.sample_cli_parser import build_sample_parser
from utils.generation.sample_cli_passthrough import append_sample_repair_passthrough


def test_prepare_runtime_and_namespace() -> None:
    rt = prepare_invention_runtime(
        "four red cubes left of blue spheres, no watermark",
        enable="auto",
        base_steps=28,
        adaptive_steps=True,
    )
    assert rt.positive
    assert rt.steps is not None

    args = argparse.Namespace(
        prompt="2girls holding hands, no hat",
        negative_prompt="",
        invention_stack="auto",
        invention_adaptive_steps=True,
        steps=28,
        box_layout="",
        anti_bleed=False,
    )
    out = apply_invention_to_namespace(args, work_dir="")
    assert out is not None
    assert args.prompt
    assert int(args.steps) >= 12


def test_cli_parser_invention_flags() -> None:
    p = build_sample_parser()
    ns = p.parse_args(
        [
            "--ckpt",
            "x.pt",
            "--prompt",
            "a cat",
            "--invention-stack",
            "auto",
            "--invention-adaptive-steps",
            "--invention-self-heal",
        ]
    )
    assert ns.invention_stack == "auto"
    assert ns.invention_adaptive_steps is True
    assert ns.invention_self_heal is True


def test_passthrough_forwards_invention_flags() -> None:
    args = argparse.Namespace(
        invention_stack="auto",
        invention_adaptive_steps=True,
        invention_self_heal=False,
        corpus_jsonl="data/train.jsonl",
        corpus_top_k=4,
        corpus_source="danbooru",
        corpus_root="",
        cast_json="cast.json",
    )
    cmd: list[str] = ["python", "sample.py"]
    append_sample_repair_passthrough(cmd, args)
    # The repair run receives the already-rewritten prompt / box layout / steps; re-applying
    # these prompt transforms would duplicate them.
    for flag in ("--invention-stack", "--invention-adaptive-steps", "--cast-json", "--corpus-jsonl"):
        assert flag not in cmd


def test_cast_and_corpus_apply_to_namespace(tmp_path) -> None:
    import json

    img = tmp_path / "ref.png"
    Image.new("RGB", (8, 8)).save(img)
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"image": str(img), "text": "catgirl, black hair, red hoodie", "source": "danbooru"},
        {"image": str(img), "text": "knight in armor", "source": "gelbooru"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cast = tmp_path / "cast.json"
    cast.write_text(
        json.dumps({"actors": [{"id": "A", "wardrobe": ["red hoodie"]}, {"id": "B", "wardrobe": ["blue dress"]}]}),
        encoding="utf-8",
    )
    args = build_sample_parser().parse_args(
        ["--ckpt", "x.pt", "--prompt", "catgirl", "--cast-json", str(cast), "--corpus-jsonl", str(corpus)]
    )
    args.corpus_source = "danbooru"
    apply_cast_and_corpus_to_namespace(args, work_dir=str(tmp_path))
    assert "A wears red hoodie" in args.prompt and "B wears blue dress" in args.prompt
    assert "catgirl, black hair" in args.prompt and "knight" not in args.prompt
    assert "B wearing red hoodie" in args.negative_prompt
    assert json.loads(open(args.box_layout, encoding="utf-8").read())["regions"]
    assert args.anti_bleed is True
    assert args.moodboard_images == str(img)


def test_perfect_gen_argv_includes_inventions() -> None:
    argv = build_sample_argv("portrait of a girl", invention_stack="auto")
    assert "--invention-stack" in argv
