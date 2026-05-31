"""Wave 13: TCFG, SLG, DBCache fingerprint, guidance probe."""

from __future__ import annotations

import torch
from utils.generation.guidance_probe import GuidanceProbe
from utils.generation.guidance_stack import combine_guided_prediction
from utils.generation.slg_guidance import parse_skip_blocks, slg_combine
from utils.generation.tcfg import tcfg_combine, tcfg_damp_unconditional
from utils.superior.block_cache import BlockDiTCache


def test_tcfg_damp_changes_uncond() -> None:
    cond = torch.randn(1, 4, 8, 8)
    uncond = cond + torch.randn_like(cond) * 0.5
    damped = tcfg_damp_unconditional(cond, uncond, damping=1.0)
    assert not torch.allclose(damped, uncond)


def test_tcfg_combine_differs_from_cfg() -> None:
    cond = torch.randn(1, 4, 4, 4)
    uncond = torch.randn(1, 4, 4, 4)
    out = tcfg_combine(cond, uncond, cfg_scale=7.0, damping=0.9)
    plain = uncond + 7.0 * (cond - uncond)
    assert not torch.allclose(out, plain)


def test_combine_guided_tcfg() -> None:
    cond = torch.randn(1, 2, 2, 2)
    uncond = torch.zeros_like(cond)
    out = combine_guided_prediction(cond, uncond, uncond, cfg_scale=5.0, tcfg_damping=0.8, zeresfdg_strength=0.0)
    assert out.shape == cond.shape


def test_slg_combine() -> None:
    cond = torch.ones(1, 2, 2, 2)
    uncond = torch.zeros(1, 2, 2, 2)
    skip = cond * 0.5
    out = slg_combine(cond, uncond, skip, cfg_scale=7.0, slg_scale=2.0)
    assert out.shape == cond.shape


def test_parse_skip_blocks_auto() -> None:
    blocks = parse_skip_blocks("auto", depth=12)
    assert len(blocks) >= 1


def test_guidance_probe_rerank() -> None:
    probe = GuidanceProbe(tau_steps=2)
    c = torch.randn(2, 4, 4, 4)
    u = torch.zeros_like(c)
    probe.note(c, u, step=0)
    probe.note(c * 2, u, step=1)
    order = probe.rerank_indices()
    assert len(order) == 2


def test_block_cache_cfg_split_fingerprint() -> None:
    cache = BlockDiTCache()
    t = torch.randn(4, 256)
    x = torch.randn(4, 16, 64)
    fp_full = cache.fingerprint_from_tensors(t, x, cfg_split=False)
    fp_split = cache.fingerprint_from_tensors(t, x, cfg_split=True)
    assert fp_split.shape == fp_full.shape
