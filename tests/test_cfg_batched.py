"""Batched CFG helper (shape/math smoke)."""

from __future__ import annotations

import torch
from utils.generation.cfg_batched import batched_cfg_forward, merge_cfg_model_kwargs


class _TinyCfgModel(torch.nn.Module):
    def forward(self, x, t, encoder_hidden_states=None, **kwargs):
        del t, kwargs
        bias = encoder_hidden_states.mean(dim=(1, 2)).view(x.shape[0], 1, 1, 1)
        return x + bias.expand_as(x)


def test_batched_cfg_matches_two_forwards():
    model = _TinyCfgModel()
    x = torch.randn(2, 4, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    emb_c = torch.randn(2, 3, 16)
    emb_u = torch.randn(2, 3, 16)
    mk_c = {"encoder_hidden_states": emb_c}
    mk_u = {"encoder_hidden_states": emb_u}
    batched = batched_cfg_forward(model, x, t, model_kwargs_cond=mk_c, model_kwargs_uncond=mk_u, cfg_scale=7.5)
    oc = model(x, t, **mk_c)
    ou = model(x, t, **mk_u)
    sequential = ou + 7.5 * (oc - ou)
    assert torch.allclose(batched, sequential, atol=1e-5)


def test_merge_cfg_kwargs_doubles_batch():
    mk_c = {"encoder_hidden_states": torch.randn(2, 3, 4)}
    mk_u = {"encoder_hidden_states": torch.randn(2, 3, 4)}
    merged = merge_cfg_model_kwargs(mk_c, mk_u)
    assert merged["encoder_hidden_states"].shape[0] == 4
