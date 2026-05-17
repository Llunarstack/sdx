"""Every ``python -m scripts.tools`` command must resolve to an existing script."""

from __future__ import annotations

from scripts.tools import __main__ as tools_main


def test_tools_dispatcher_targets_exist():
    missing = [
        name
        for name, path in tools_main._CANONICAL.items()
        if not path.is_file()
    ]
    assert not missing, f"Broken dispatcher paths: {missing}"


def test_training_commands_use_training_dir():
    for name in (
        "noise_schedule_export",
        "mine_preference_pairs",
        "train_diffusion_dpo",
        "train_kd_distill",
    ):
        path = tools_main._CANONICAL[name]
        assert path.parent.name == "training", f"{name} -> {path}"
