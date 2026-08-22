"""Trainable LoRA/DoRA injection: freezing, grad flow, and adapter round-trip."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from models.lora_train import (  # noqa: E402
    TrainableLoRALinear,
    inject_trainable_lora,
    lora_state_dict,
)


class _TinyDiT(nn.Module):
    """Stand-in with the module-name substrings the default targets match."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(16, 16)  # not a target substring -> stays frozen-trainable base

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 48)
        self.proj = nn.Linear(16, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        _ = self.qkv(x)
        x = self.proj(x)
        x = self.fc2(torch.relu(self.fc1(x)))
        return x


def test_injection_wraps_targets_and_freezes_base():
    m = _TinyDiT()
    trainable, wrapped = inject_trainable_lora(m, rank=4, alpha=8.0)
    # 2 blocks x (qkv, proj, fc1, fc2) = 8 wrapped Linears.
    assert wrapped == 8
    # Base weights frozen; only adapter params require grad.
    for name, p in m.named_parameters():
        if "lora_down" in name or "lora_up" in name:
            assert p.requires_grad, name
        else:
            assert not p.requires_grad, name
    assert len(trainable) == 2 * 4 * 2  # 8 layers x (down, up)


def test_forward_and_backward_updates_only_adapters():
    torch.manual_seed(0)
    m = _TinyDiT()
    inject_trainable_lora(m, rank=4, alpha=8.0)
    base_before = m.blocks[0].proj.linear.weight.clone()

    x = torch.randn(2, 16)
    loss = m(x).pow(2).mean()
    loss.backward()
    # Adapter got grad; frozen base did not.
    assert m.blocks[0].proj.lora_down.grad is not None
    assert m.blocks[0].proj.linear.weight.grad is None
    torch.testing.assert_close(m.blocks[0].proj.linear.weight, base_before)


def test_lora_identity_at_init():
    # up is zero-init, so the wrapped layer == base layer at step 0.
    torch.manual_seed(1)
    lin = nn.Linear(16, 16)
    wrapped = TrainableLoRALinear(lin, rank=4, alpha=8.0)
    x = torch.randn(3, 16)
    torch.testing.assert_close(wrapped(x), lin(x))


def test_dora_has_magnitude_and_runs():
    lin = nn.Linear(16, 16)
    w = TrainableLoRALinear(lin, rank=4, alpha=8.0, use_dora=True)
    assert w.dora_magnitude is not None and w.dora_magnitude.requires_grad
    out = w(torch.randn(2, 16))
    assert out.shape == (2, 16)


def test_adapter_state_dict_keys_match_inference_loader():
    from models.lora import _extract_adapters

    m = _TinyDiT()
    inject_trainable_lora(m, rank=4, alpha=8.0, use_dora=True)
    sd = lora_state_dict(m)
    # Keys use the .lora_down/.lora_up/.alpha/.dora_magnitude_vector convention.
    assert any(k.endswith(".lora_down.weight") for k in sd)
    assert any(k.endswith(".lora_up.weight") for k in sd)
    assert any(k.endswith(".dora_magnitude_vector") for k in sd)
    # The inference loader parses them into adapters with down+up populated.
    adapters = _extract_adapters(sd)
    assert adapters
    a = next(iter(adapters.values()))
    assert a.down is not None and a.up is not None


def test_no_target_match_reports_zero():
    solo = nn.Linear(8, 8)  # name "" has no target substring
    _, wrapped = inject_trainable_lora(solo, rank=2, alpha=2.0)
    assert wrapped == 0
