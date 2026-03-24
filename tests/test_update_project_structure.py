"""Tests for scripts.tools.repo.update_project_structure."""

from pathlib import Path

import pytest
from scripts.tools.repo import update_project_structure as ups


def test_build_tree_includes_train_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "mini"
    root.mkdir()
    (root / "train.py").write_text("# x", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "cli.py").write_text("# x", encoding="utf-8")

    lines = ups.build_tree_lines(root, max_depth=3, skip_extra_dirs=set())
    text = "\n".join(lines)
    assert "train.py" in text
    assert "cli.py" in text


def test_script_writes_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out.md"
    monkeypatch.setattr(ups, "REPO_ROOT", tmp_path)
    (tmp_path / "train.py").write_text("#", encoding="utf-8")

    import sys

    monkeypatch.setattr(sys, "argv", ["update_project_structure.py", "--out", str(out), "--max-depth", "2"])
    code = ups.main()
    assert code == 0
    body = out.read_text(encoding="utf-8")
    assert "Auto-generated" in body
    assert "train.py" in body
