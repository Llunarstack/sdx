"""``python -m scripts.tools`` resolves every tool script with a ``__main__`` guard."""

from __future__ import annotations

import subprocess
import sys

from scripts.tools import __main__ as tools_main


def test_documented_commands_resolve():
    cmds = tools_main.commands()
    for name in ("ckpt_info", "data_quality", "preference_flywheel", "perfect_gen", "feedback", "corpus_retrieve"):
        assert cmds[name][0].is_file(), name
    assert "scorers" not in cmds  # library module, no CLI


def test_training_commands_use_training_dir():
    cmds = tools_main.commands()
    for name in ("noise_schedule_export", "mine_preference_pairs", "train_diffusion_dpo", "train_kd_distill"):
        assert cmds[name][0].parent.name == "training", f"{name} -> {cmds[name]}"


def test_merged_superior_scripts_keep_old_names():
    cmds = tools_main.commands()
    assert cmds["superior_curate"] == (cmds["superior"][0], ["curate"])
    assert cmds["run_flywheel"] == (cmds["superior"][0], ["flywheel"])
    assert cmds["agentic_evolve"] == (cmds["agentic"][0], ["evolve"])
    out = subprocess.run(
        [sys.executable, "-m", "scripts.tools", "superior_generate", "--ckpt", "x.pt", "--prompt", "p", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "sample.py" in out and "--prompt" in out
