from utils.superior.eval_report import build_markdown_report


def test_eval_report_includes_hf_section(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "results.json").write_text("[]", encoding="utf-8")
    (bench / "leaderboard.json").write_text("[]", encoding="utf-8")
    md = build_markdown_report(bench)
    assert "# SDX Benchmark Report" in md
    assert "## Leaderboard" in md


def test_eval_report_includes_text_encoder_stack(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "results.json").write_text("[]", encoding="utf-8")
    (bench / "leaderboard.json").write_text("[]", encoding="utf-8")
    md = build_markdown_report(bench)
    assert "## Text encoder stack" in md or "## HF pretrained status" in md
