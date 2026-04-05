"""Tests for pipelines.book_comic.book_training_helpers."""

from pathlib import Path
from types import SimpleNamespace

from pipelines.book_comic import book_training_helpers as bth


def _args(**overrides):
    base = dict(
        book_train_preset="balanced",
        data_path="data/train",
        manifest_jsonl="",
        results_dir="results/book_train",
        model="",
        image_size=0,
        global_batch_size=0,
        lr=0.0,
        passes=-1,
        max_steps=-1,
        ar_profile="auto",
        num_ar_blocks=-1,
        ar_block_order="",
        dry_run=False,
        no_compile=False,
        no_xformers=False,
        num_workers=-1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_preset_for_book_train_production():
    p = bth.preset_for_book_train("production")
    assert p.train_shortcomings_mitigation == "all"
    assert p.train_art_guidance_mode == "all"
    assert p.train_anatomy_guidance == "strong"
    assert p.use_hierarchical_captions is True


def test_build_train_command_includes_book_guidance():
    args = _args(book_train_preset="balanced", manifest_jsonl="data/manifest.jsonl", dry_run=True)
    settings = bth.resolve_book_train_settings(args)
    cmd = bth.build_train_command(
        root=Path("c:/repo"),
        python_exe="python",
        args=args,
        settings=settings,
        passthrough_train_args=["--seed", "123"],
    )
    joined = " ".join(cmd)
    assert "--train-shortcomings-mitigation auto" in joined
    assert "--train-art-guidance-mode auto" in joined
    assert "--train-style-guidance-mode auto" in joined
    assert "--use-hierarchical-captions" in cmd
    assert "--num-ar-blocks" in cmd
    assert "--ar-block-order" in cmd
    assert "--dry-run" in cmd
    assert "--seed" in cmd


def test_native_manifest_preflight_without_rust_tools(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"image_path":"a.png","caption":"hello"}\n', encoding="utf-8")

    monkeypatch.setattr(bth, "run_rust_jsonl_validate", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no rust")))
    monkeypatch.setattr(bth, "run_rust_jsonl_stats", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no rust")))

    info = bth.run_native_manifest_preflight(manifest)
    assert info["exists"] is True
    assert "fnv1a64=" in info["fingerprint"] or "linecrc" in info["fingerprint"]
    assert info["rust_validate_ok"] is None
    assert info["rust_stats_ok"] is None


def test_build_hf_export_command():
    cmd = bth.build_hf_export_command(
        root=Path("c:/repo"),
        python_exe="python",
        dataset="org/bookset",
        out_dir=Path("data/out"),
        manifest_name="m.jsonl",
        max_samples=123,
        streaming=True,
        shuffle_seed=7,
    )
    joined = " ".join(cmd)
    assert "hf_export_to_sdx_manifest.py" in joined
    assert "--dataset org/bookset" in joined
    assert "--manifest-name m.jsonl" in joined
    assert "--max-samples 123" in joined
    assert "--streaming" in cmd
    assert "--shuffle-seed" in cmd


def test_build_caption_normalize_command():
    cmd = bth.build_caption_normalize_command(
        root=Path("c:/repo"),
        python_exe="python",
        inp_manifest=Path("in.jsonl"),
        out_manifest=Path("out.jsonl"),
        shortcomings_mitigation="all",
        shortcomings_2d=True,
        art_guidance_mode="auto",
        art_guidance_photography=False,
        anatomy_guidance="strong",
        style_guidance_mode="all",
        style_guidance_artists=False,
    )
    joined = " ".join(cmd)
    assert "normalize_captions.py" in joined
    assert "--shortcomings-mitigation all" in joined
    assert "--shortcomings-2d" in cmd
    assert "--no-art-guidance-photography" in cmd
    assert "--style-guidance-mode all" in joined
    assert "--no-style-guidance-artists" in cmd


def test_resolve_train_humanization_pack_strong():
    out = bth.resolve_train_humanization_pack("strong")
    assert out["shortcomings_mitigation"] == "all"
    assert out["anatomy_guidance"] == "strong"
    assert out["style_guidance_mode"] == "all"


def test_resolve_book_ar_profile_and_explicit_overrides():
    auto = bth.resolve_book_ar_profile("auto")
    assert auto == {}
    z = bth.resolve_book_ar_profile("zorder")
    assert z["num_ar_blocks"] == 2
    assert z["ar_block_order"] == "zorder"

    settings = bth.resolve_book_train_settings(_args(ar_profile="layout", num_ar_blocks=4, ar_block_order="zorder"))
    assert settings.num_ar_blocks == 4
    assert settings.ar_block_order == "zorder"
