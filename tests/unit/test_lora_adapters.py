import torch
import torch.nn as nn

from models.lora import MultiLoRALinear, apply_loras


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


class _ToyStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def _state_lora_down_up(base: str = "proj"):
    return {
        f"{base}.lora_down.weight": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # rank=1
        f"{base}.lora_up.weight": torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
        f"{base}.alpha": torch.tensor(1.0),
    }


def _state_lora_ab_with_dora(base: str = "proj"):
    return {
        f"{base}.lora_A.weight": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # rank=1
        f"{base}.lora_B.weight": torch.tensor([[1.0], [1.0], [1.0], [1.0]]),
        f"{base}.lora_alpha": torch.tensor(1.0),
        f"{base}.dora_magnitude_vector": torch.tensor([1.0, 0.5, 0.5, 1.0]),
    }


def test_apply_loras_wraps_linear_and_changes_output():
    m = _Toy()
    x = torch.ones(2, 4)
    y0 = m(x)
    _, n = apply_loras(m, [(_state_lora_down_up(), 1.0)])
    assert n == 1
    assert isinstance(m.proj, MultiLoRALinear)
    y1 = m(x)
    assert not torch.allclose(y0, y1)


def test_apply_loras_supports_lora_ab_and_dora():
    m = _Toy()
    x = torch.ones(1, 4)
    _, n = apply_loras(m, [(_state_lora_ab_with_dora(), 1.0)])
    assert n == 1
    y = m(x)
    assert torch.isfinite(y).all()


def test_multi_lora_scale_normalization_caps_total():
    m = _Toy()
    s1 = _state_lora_down_up()
    s2 = _state_lora_down_up()
    _, n = apply_loras(
        m,
        [(s1, 2.0), (s2, 2.0)],
        normalize_scales=True,
        max_total_scale=1.0,
    )
    assert n == 1
    assert isinstance(m.proj, MultiLoRALinear)
    # internal scales should be capped to total abs <= 1.0
    assert sum(abs(float(s)) for s in m.proj._scales) <= 1.00001


def test_multi_lora_role_budget_caps_each_role():
    m = _Toy()
    s1 = _state_lora_down_up()
    s2 = _state_lora_down_up()
    _, n = apply_loras(
        m,
        [(s1, 2.0, "character"), (s2, 2.0, "style")],
        normalize_scales=True,
        max_total_scale=10.0,
        role_budgets={"character": 1.2, "style": 0.5},
    )
    assert n == 1
    assert isinstance(m.proj, MultiLoRALinear)
    ch_total = sum(abs(float(s)) for s, r in zip(m.proj._scales, m.proj._roles) if r == "character")
    st_total = sum(abs(float(s)) for s, r in zip(m.proj._scales, m.proj._roles) if r == "style")
    assert ch_total <= 1.20001
    assert st_total <= 0.50001


def test_stage_policy_character_focus_emphasizes_early_character_layers():
    m = _ToyStack()
    _, n = apply_loras(
        m,
        [(_state_lora_down_up("blocks.0"), 1.0, "character"), (_state_lora_down_up("blocks.2"), 1.0, "character")],
        normalize_scales=False,
        stage_policy="character_focus",
    )
    assert n == 2
    assert isinstance(m.blocks[0], MultiLoRALinear)
    assert isinstance(m.blocks[2], MultiLoRALinear)
    s_early = float(m.blocks[0]._scales[0])
    s_late = float(m.blocks[2]._scales[0])
    assert s_early > s_late


def test_custom_role_stage_weights_override_policy():
    m = _ToyStack()
    _, n = apply_loras(
        m,
        [(_state_lora_down_up("blocks.0"), 1.0, "character"), (_state_lora_down_up("blocks.2"), 1.0, "character")],
        normalize_scales=False,
        stage_policy="character_focus",
        role_stage_weights={"character": (0.5, 1.0, 1.5)},
    )
    assert n == 2
    s_early = float(m.blocks[0]._scales[0])
    s_late = float(m.blocks[2]._scales[0])
    assert s_late > s_early

