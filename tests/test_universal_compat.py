import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
from utils.compat.adapter_bridge import bridge_apply, collect_foreign_deltas, map_sdx_targets  # noqa: E402
from utils.compat.asset_sniffer import sniff_asset  # noqa: E402
from utils.compat.embedding_bridge import load_textual_inversion, resize_vectors  # noqa: E402
from utils.compat.latent_probe import LatentBridge, probe_vae  # noqa: E402
from utils.compat.lycoris_math import (  # noqa: E402
    delta_from_loha,
    delta_from_lokr,
    delta_from_lora,
    project_delta,
    svd_factorize,
)

# ---------------------------------------------------------------------------
# Sniffer
# ---------------------------------------------------------------------------


def test_sniff_kohya_sdxl_lora():
    state = {
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": torch.zeros(8, 320),
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": torch.zeros(320, 8),
        "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.zeros(8, 1280),
    }
    rep = sniff_asset(state)
    assert rep.kind == "adapter"
    assert rep.adapter_algo == "lora"
    assert rep.family == "sdxl"


def test_sniff_loha_and_flux_adapters():
    loha = {
        "lora_unet_x.hada_w1_a": torch.zeros(64, 8),
        "lora_unet_x.hada_w1_b": torch.zeros(8, 64),
        "lora_unet_x.hada_w2_a": torch.zeros(64, 8),
        "lora_unet_x.hada_w2_b": torch.zeros(8, 64),
    }
    assert sniff_asset(loha).adapter_algo == "loha"
    flux = {
        "transformer.double_blocks.0.img_attn.qkv.lora_A.weight": torch.zeros(8, 3072),
        "transformer.double_blocks.0.img_attn.qkv.lora_B.weight": torch.zeros(9216, 8),
    }
    rep = sniff_asset(flux)
    assert rep.kind == "adapter"
    assert rep.family == "flux"


def test_sniff_checkpoint_families():
    sd15 = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": torch.zeros(320, 768),
    }
    assert sniff_asset(sd15).family == "sd15"
    sd2 = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": torch.zeros(320, 1024),
    }
    assert sniff_asset(sd2).family == "sd2"
    flux = {"double_blocks.0.img_attn.qkv.weight": torch.zeros(9216, 3072)}
    assert sniff_asset(flux).family == "flux"


def test_sniff_vae_embedding_upscaler():
    vae = {
        "encoder.down_blocks.0.resnets.0.conv1.weight": torch.zeros(128, 128, 3, 3),
        "decoder.conv_in.weight": torch.zeros(512, 4, 3, 3),
    }
    assert sniff_asset(vae).kind == "vae"
    ti = {"string_to_param": {"*": torch.zeros(4, 768)}}
    assert sniff_asset(ti).kind == "embedding"
    esrgan = {"body.0.rdb1.conv1.weight": torch.zeros(32, 64, 3, 3)}
    rep = sniff_asset(esrgan)
    assert rep.kind == "upscaler"
    assert rep.family == "esrgan"


# ---------------------------------------------------------------------------
# Adapter math
# ---------------------------------------------------------------------------


def test_lora_delta_matches_manual():
    down, up = torch.randn(4, 16), torch.randn(16, 4)
    delta = delta_from_lora(down, up, alpha=4.0)
    assert torch.allclose(delta, (up @ down) * (4.0 / 4), atol=1e-6)


def test_loha_delta_matches_manual():
    a1, b1 = torch.randn(8, 2), torch.randn(2, 8)
    a2, b2 = torch.randn(8, 2), torch.randn(2, 8)
    manual = (a1 @ b1) * (a2 @ b2)
    assert torch.allclose(delta_from_loha(a1, b1, a2, b2), manual, atol=1e-5)  # no alpha -> scale 1
    assert torch.allclose(delta_from_loha(a1, b1, a2, b2, alpha=1.0), manual * 0.5, atol=1e-5)  # alpha/rank


def test_lokr_delta_is_kronecker():
    w1, w2 = torch.randn(2, 2), torch.randn(3, 4)
    delta = delta_from_lokr(w1, w2)
    assert delta.shape == (6, 8)
    assert torch.allclose(delta, torch.kron(w1, w2) * (1.0), atol=1e-5)


def test_svd_factorize_roundtrip():
    delta = torch.randn(16, 4) @ torch.randn(4, 24)  # true rank 4
    down, up = svd_factorize(delta, rank=4)
    assert torch.allclose(up @ down, delta, atol=1e-4)


def test_project_delta_hits_target_shape():
    delta = torch.randn(320, 320)
    out = project_delta(delta, (256, 512), rank=8)
    assert out.shape == (256, 512)


# ---------------------------------------------------------------------------
# Bridge onto a toy sdx-style DiT
# ---------------------------------------------------------------------------


class _ToyAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)


class _ToyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = _ToyAttn(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.mlp.fc1, self.mlp.fc2 = self.mlp[0], self.mlp[2]


class _ToyDiT(nn.Module):
    def __init__(self, d=64, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock(d) for _ in range(depth)])


def _foreign_sd_lora() -> dict:
    # kohya-style SD LoRA hitting q and mlp at two depths
    state = {}
    for blk, dim in (("input_blocks_1", 320), ("output_blocks_9", 320)):
        base_q = f"lora_unet_{blk}_1_transformer_blocks_0_attn1_to_q"
        state[f"{base_q}.lora_down.weight"] = torch.randn(4, dim) * 0.01
        state[f"{base_q}.lora_up.weight"] = torch.randn(dim, 4) * 0.01
        base_ff = f"lora_unet_{blk}_1_transformer_blocks_0_ff_net_0_proj"
        state[f"{base_ff}.lora_down.weight"] = torch.randn(4, dim) * 0.01
        state[f"{base_ff}.lora_up.weight"] = torch.randn(dim * 4, 4) * 0.01
    state["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"] = torch.randn(4, 768)
    state["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight"] = torch.randn(3072, 4)
    return state


def test_collect_foreign_deltas_tags_roles_and_skips_te():
    from utils.compat.adapter_bridge import BridgeReport

    rep = BridgeReport()
    deltas = collect_foreign_deltas(_foreign_sd_lora(), report=rep)
    assert rep.skipped_text_encoder == 1
    roles = sorted({d.role for d in deltas})
    assert roles == ["mlp_in", "q"]


def test_map_sdx_targets_finds_fused_qkv_and_mlp():
    targets = map_sdx_targets(_ToyDiT())
    assert len(targets["qkv"]) == 4
    assert len(targets["mlp_in"]) == 4
    assert any(t for t in targets["q"])  # q maps onto fused qkv too


def test_bridge_apply_changes_output_and_reports_coverage():
    torch.manual_seed(0)
    model = _ToyDiT()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        before = [blk.attn.qkv(x) for blk in model.blocks]
    report = bridge_apply(model, _foreign_sd_lora(), scale=1.0, rank=4)
    assert report.foreign_layers == 4
    assert report.mapped == 4
    assert report.coverage == 1.0
    with torch.no_grad():
        after = [blk.attn.qkv(x) if hasattr(blk.attn.qkv, "linear") else None for blk in model.blocks]
    changed = any(a is not None and not torch.allclose(a, b) for a, b in zip(after, before))
    assert changed


def test_bridge_q_delta_lands_in_q_rows_of_fused_qkv():
    torch.manual_seed(0)
    model = _ToyDiT(d=64, depth=1)
    state = {
        "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q.lora_down.weight": torch.randn(4, 320) * 0.05,
        "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q.lora_up.weight": torch.randn(320, 4) * 0.05,
    }
    bridge_apply(model, state, scale=1.0, rank=4)
    wrapper = model.blocks[0].attn.qkv
    ad = wrapper._adapter_params[0]
    delta = (ad["up"] @ ad["down"]).detach()
    assert delta[:64].abs().sum() > 0  # q rows carry the adaptation
    assert delta[64:].abs().sum() < 1e-4  # k/v rows untouched


# ---------------------------------------------------------------------------
# CivitAI base-model families
# ---------------------------------------------------------------------------


def test_sniff_all_popular_civitai_checkpoint_families():
    cases = {
        "sdxl": {"add_embedding.linear_1.weight": torch.zeros(1280, 2816)},  # + Pony/Illustrious/NoobAI
        "sd3": {"model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight": torch.zeros(4608, 1536)},
        "auraflow": {
            "joint_transformer_blocks.0.attn.to_q.weight": torch.zeros(3072, 3072),
            "register_tokens": torch.zeros(8, 3072),
        },
        "hunyuan_dit": {"blocks.0.attn1.Wqkv.weight": torch.zeros(4224, 1408)},
        "cascade": {"clip_txt_pooled_mapper.weight": torch.zeros(2048, 1280)},
        "lumina": {
            "cap_embedder.0.weight": torch.zeros(2304),
            "layers.0.attention.qkv.weight": torch.zeros(6912, 2304),
        },
        "hidream": {"double_stream_blocks.0.block.ff_i.shared_experts.w1.weight": torch.zeros(768, 2560)},
        "qwen_image": {
            "transformer_blocks.0.img_mod.1.weight": torch.zeros(18432, 3072),
            "transformer_blocks.0.attn.to_q.weight": torch.zeros(3072, 3072),
        },
        "pixart": {
            "transformer_blocks.0.attn1.to_q.weight": torch.zeros(1152, 1152),
            "adaln_single.linear.weight": torch.zeros(6912, 1152),
        },
    }
    for family, state in cases.items():
        rep = sniff_asset(state)
        assert rep.kind == "checkpoint", f"{family}: kind={rep.kind}"
        assert rep.family == family, f"expected {family}, got {rep.family}"


def test_sniff_flux_diffusers_layout_and_chroma_variant():
    flux_diffusers = {
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(3072, 3072),
        "single_transformer_blocks.0.proj_mlp.weight": torch.zeros(12288, 3072),
        "context_embedder.weight": torch.zeros(3072, 4096),
    }
    assert sniff_asset(flux_diffusers).family == "flux"
    chroma = {
        "double_blocks.0.img_attn.qkv.weight": torch.zeros(9216, 3072),
        "distilled_guidance_layer.layers.0.weight": torch.zeros(5120, 64),
    }
    rep = sniff_asset(chroma)
    assert rep.family == "flux"
    assert rep.details.get("variant") == "chroma"


def test_sniff_kolors_variant_and_sd3_diffusers():
    kolors = {
        "add_embedding.linear_1.weight": torch.zeros(1280, 2816),
        "encoder_hid_proj.weight": torch.zeros(2048, 4096),
    }
    rep = sniff_asset(kolors)
    assert rep.family == "sdxl"
    assert rep.details.get("variant") == "kolors"
    sd3_diffusers = {
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(1536, 1536),
        "context_embedder.weight": torch.zeros(1536, 4096),
    }
    assert sniff_asset(sd3_diffusers).family == "sd3"


def test_sniff_adapter_families_for_new_ecosystems():
    hunyuan_lora = {
        "lora_unet_blocks_0_attn1_Wqkv.lora_down.weight": torch.zeros(8, 1408),
        "lora_unet_blocks_0_attn1_Wqkv.lora_up.weight": torch.zeros(4224, 8),
    }
    assert sniff_asset(hunyuan_lora).family == "hunyuan_dit"
    qwen_lora = {
        "transformer.transformer_blocks.0.img_mlp.net.0.proj.lora_A.weight": torch.zeros(8, 3072),
        "transformer.transformer_blocks.0.img_mlp.net.0.proj.lora_B.weight": torch.zeros(12288, 8),
    }
    assert sniff_asset(qwen_lora).family == "qwen_image"
    auraflow_lora = {
        "transformer.joint_transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(8, 3072),
        "transformer.joint_transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(3072, 8),
    }
    assert sniff_asset(auraflow_lora).family == "auraflow"


def test_flux_single_block_fused_linears_split_into_roles():
    d, mlp = 64, 256
    state = {
        # linear1 fuses qkv (3d) + mlp-in rows
        "lora_unet_single_blocks_0_linear1.lora_down.weight": torch.randn(4, d) * 0.05,
        "lora_unet_single_blocks_0_linear1.lora_up.weight": torch.randn(3 * d + mlp, 4) * 0.05,
        # linear2 fuses attn-out + mlp-out columns
        "lora_unet_single_blocks_0_linear2.lora_down.weight": torch.randn(4, d + mlp) * 0.05,
        "lora_unet_single_blocks_0_linear2.lora_up.weight": torch.randn(d, 4) * 0.05,
    }
    deltas = collect_foreign_deltas(state)
    roles = sorted(fd.role for fd in deltas)
    assert roles == ["attn_out", "mlp_in", "mlp_out", "qkv"]
    shapes = {fd.role: tuple(fd.delta.shape) for fd in deltas}
    assert shapes["qkv"] == (3 * d, d)
    assert shapes["mlp_in"] == (mlp, d)
    assert shapes["attn_out"] == (d, d)
    assert shapes["mlp_out"] == (d, mlp)


def test_bridge_apply_flux_style_lora_end_to_end():
    torch.manual_seed(0)
    model = _ToyDiT(d=64, depth=2)
    state = {
        "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": torch.randn(4, 3072) * 0.02,
        "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": torch.randn(9216, 4) * 0.02,
        "lora_unet_single_blocks_1_linear1.lora_down.weight": torch.randn(4, 3072) * 0.02,
        "lora_unet_single_blocks_1_linear1.lora_up.weight": torch.randn(21504, 4) * 0.02,
    }
    report = bridge_apply(model, state, scale=1.0, rank=4)
    assert report.foreign_layers == 2
    assert report.mapped == 3  # fused qkv + (qkv, mlp_in) from the split
    assert report.skipped_role == 0


# ---------------------------------------------------------------------------
# Embeddings + VAE
# ---------------------------------------------------------------------------


def test_load_ti_formats_and_resize_preserves_norm():
    a1111 = load_textual_inversion({"string_to_param": {"*": torch.randn(3, 768)}})
    assert a1111.vectors["clip"].shape == (3, 768)
    dual = load_textual_inversion({"clip_l": torch.randn(2, 768), "clip_g": torch.randn(2, 1280)})
    assert set(dual.vectors) == {"clip_l", "clip_g"}
    v = torch.randn(3, 768)
    out = resize_vectors(v, 1280)
    assert out.shape == (3, 1280)
    assert torch.allclose(out.norm(dim=1), v.norm(dim=1), atol=1e-4)


def test_probe_vae_reads_channels_and_bridge_identity():
    probe = probe_vae({"decoder.conv_in.weight": torch.zeros(512, 4, 3, 3)})
    assert probe.latent_channels == 4
    assert "kl-f8" in probe.family_guess
    bridge = LatentBridge(4, 4)
    z = torch.randn(1, 4, 8, 8)
    assert torch.allclose(bridge(z), z)  # identity at init
    bridge16 = LatentBridge(4, 16)
    assert bridge16(z).shape == (1, 16, 8, 8)
