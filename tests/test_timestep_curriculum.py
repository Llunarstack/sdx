"""Tests for ``utils.training.timestep_curriculum``."""

from __future__ import annotations

from types import SimpleNamespace

from utils.training.timestep_curriculum import (
    parse_timestep_curriculum_schedule,
    resolve_timestep_kwargs_for_step,
)


def test_parse_curriculum_segments() -> None:
    phases = parse_timestep_curriculum_schedule("0:high_noise|40000:logit_normal:0:1.5")
    assert len(phases) == 2
    assert phases[0].start_step == 0
    assert phases[0].mode == "high_noise"
    assert phases[1].mode == "logit_normal"
    assert phases[1].logit_mean == 0.0
    assert phases[1].logit_std == 1.5


def test_resolve_phase_by_global_step() -> None:
    cfg = SimpleNamespace(
        timestep_curriculum_schedule="0:high_noise|10:uniform",
        timestep_sample_mode="logit_normal",
        timestep_logit_mean=0.0,
        timestep_logit_std=1.0,
    )
    assert resolve_timestep_kwargs_for_step(cfg, 0)["mode"] == "high_noise"
    assert resolve_timestep_kwargs_for_step(cfg, 9)["mode"] == "high_noise"
    assert resolve_timestep_kwargs_for_step(cfg, 10)["mode"] == "uniform"


def test_empty_schedule_uses_flat_config() -> None:
    cfg = SimpleNamespace(
        timestep_curriculum_schedule="",
        timestep_sample_mode="low_noise",
        timestep_logit_mean=0.0,
        timestep_logit_std=1.0,
    )
    kw = resolve_timestep_kwargs_for_step(cfg, 999)
    assert kw["mode"] == "low_noise"
