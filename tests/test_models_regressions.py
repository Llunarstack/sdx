"""Regression tests for bugs fixed in ``models/``."""

import copy

import torch
import torch.nn.functional as F
from models.dit_text import DiT_Text
from models.linear_attention import LocalWindowAttention
from models.lora import apply_loras
from models.lora_train import TrainableLoRALinear
from models.prompt_adherence import PromptParser
from models.rope2d import build_2d_rope_freqs


def test_token_routing_with_register_tokens_backpropagates():
    # The register-token gate used to be written in place into the sigmoid output, breaking backward.
    model = DiT_Text(
        input_size=8,
        patch_size=2,
        hidden_size=32,
        depth=1,
        num_heads=4,
        text_dim=16,
        num_register_tokens=2,
        token_routing_enabled=True,
        use_xformers=False,
    ).train()
    out = model(torch.randn(2, 4, 8, 8), torch.tensor([10, 500]), encoder_hidden_states=torch.randn(2, 3, 16))
    out.sum().backward()
    assert model.blocks[0].token_router[1].weight.grad is not None


def test_trained_dora_adapter_loads_with_same_output():
    # Inference used to multiply the low-rank delta by the DoRA magnitude instead of renormalizing W + dW.
    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(8, 6))
    trained = TrainableLoRALinear(copy.deepcopy(base[0]), rank=2, alpha=4.0, use_dora=True)
    with torch.no_grad():
        trained.lora_up.normal_()
        trained.dora_magnitude.mul_(1.7)
    x = torch.randn(3, 8)
    _, touched = apply_loras(base, [(trained.adapter_state("0"), 1.0)], normalize_scales=False)
    assert touched == 1
    assert torch.allclose(base(x), trained(x), atol=1e-5)


def test_prompt_parser_two_word_negation_and_attribute_positions():
    parsed = PromptParser().parse("a man free of armor wearing a red and blue cloak")
    armor, cloak = parsed.triples
    # "free of" / "devoid of" never matched the single-word negation regex.
    assert armor.negated and parsed.negation_indices == [2]
    # token_indices assumed attributes sit right before the subject, picking "and" instead of "red".
    assert cloak.token_indices == [7, 9, 10]


def test_rope2d_half_precision_tables_match_float32():
    # Angles used to be computed in the requested dtype; bf16 angles near 60 rad were ~0.1 off in cos.
    ref_cos, _ = build_2d_rope_freqs(64, 64, 64)
    bf16_cos, _ = build_2d_rope_freqs(64, 64, 64, dtype=torch.bfloat16)
    assert bf16_cos.dtype == torch.bfloat16
    assert (bf16_cos.float() - ref_cos).abs().max() < 0.01


def test_local_window_attention_masks_padding():
    # Zero-padded keys of partial windows used to take part in the softmax and dilute real attention.
    torch.manual_seed(0)
    attn = LocalWindowAttention(32, 4, window_size=8, num_global_tokens=0, qk_norm=False)
    torch.nn.init.normal_(attn.out_proj.weight, std=0.2)
    x = torch.randn(2, 25, 32)  # 5x5 grid inside one padded 8x8 window == plain full attention
    q, k, v = (t.reshape(2, 25, 4, 8).transpose(1, 2) for t in attn.qkv(x).chunk(3, dim=-1))
    ref = attn.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 25, 32))
    assert torch.allclose(attn(x), ref, atol=1e-5)
