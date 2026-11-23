"""Training data conditioning: aspect-ratio buckets, bucket batches, caption tag front-loading, caption dropout.

Also holds what used to be: test_training_data, test_training_objectives, test_training_wiring, test_ar_curriculum, test_device_perf, test_error_handling_utils, test_preference_rl, test_srpo_reward, test_timestep_curriculum.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from data import ResolutionBucketBatchSampler, Text2ImageDataset
from data.caption_utils import front_load_tags
from data.manifest_utils import negative_caption_from_row, pick_training_manifest
from data.t2i_dataset import nearest_aspect_bucket
from diffusion import create_diffusion
from diffusion.flow_matching import spectral_sfp_per_sample_loss
from diffusion.schedules import get_beta_schedule, get_timestep_loss_weight
from models.dit_text import TextEmbedder
from PIL import Image
from utils.training.monitoring import get_model_info
from utils.training.objectives import (
    pair_noise_to_latents,
    parse_ar_order_mix,
    parse_timestep_curriculum_schedule,
    resolve_ar_for_step,
    resolve_timestep_kwargs_for_step,
)
from utils.training.perf import (
    dataloader_perf_kwargs,
    encode_text_multi_group,
    optimal_num_workers,
    parallel_train_torchrun_example,
    resolve_training_num_workers,
    training_perf_hints,
)
from utils.training.preference import (
    GRPOGuardConfig,
    SRPOConfig,
    flow_sde_step,
    grpo_guard_weighted_loss,
    grpo_weighted_loss,
    semantic_relative_reward,
    timestep_reward_mask,
)

# ==================================================================================================
# test_training_data: Training data conditioning: aspect-ratio buckets, bucket batches, caption tag front-loading, caption dropout.
# ==================================================================================================

BUCKETS = [(64, 64), (64, 96), (96, 64), (128, 128)]  # (H, W)


def test_nearest_aspect_bucket_keeps_the_frame():
    assert BUCKETS[nearest_aspect_bucket(600, 900, BUCKETS)] == (96, 64)  # portrait -> portrait
    assert BUCKETS[nearest_aspect_bucket(900, 600, BUCKETS)] == (64, 96)  # landscape -> landscape
    assert BUCKETS[nearest_aspect_bucket(512, 512, BUCKETS)] == (128, 128)  # largest square it fills
    assert BUCKETS[nearest_aspect_bucket(100, 100, BUCKETS)] == (64, 64)  # no upscaling when a smaller one fits


def test_bucket_batches_are_single_bucket_and_fallback_keeps_the_shape(tmp_path):
    rng = np.random.default_rng(0)
    sizes = [(64, 96), (96, 64), (128, 128)] * 4  # (W, H)
    for i, (w, h) in enumerate(sizes):
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(tmp_path / f"im{i}.png")
        (tmp_path / f"im{i}.txt").write_text(f"a barn, thing {i}")
    ds = Text2ImageDataset(str(tmp_path), image_size=64, resolution_buckets=BUCKETS)
    sampler = ResolutionBucketBatchSampler(ds, 2, generator=torch.Generator().manual_seed(0))
    batches = list(sampler)
    assert len(batches) == len(sampler) == 6
    for batch in batches:
        assert len({ds._bucket_assign[i] for i in batch}) == 1
    for i, s in enumerate(ds.samples):
        with Image.open(s["path"]) as im:
            w, h = im.size
        assert ds._bucket_assign[i] == nearest_aspect_bucket(w, h, BUCKETS)

    # An unreadable file is replaced by a neighbour cropped to the broken index's bucket, so the batch still stacks.
    broken = next(i for i in range(len(ds)) if ds._bucket_assign[(i + 1) % len(ds)] != ds._bucket_assign[i])
    with open(ds.samples[broken]["path"], "wb") as f:
        f.write(b"not an image")
    assert tuple(ds[broken]["pixel_values"].shape[-2:]) == BUCKETS[ds._bucket_assign[broken]]


def test_front_load_tags_moves_whole_tags_once():
    vocab = ["detailed", "masterpiece"]
    out = front_load_tags("a barn, Detailed, ultra detailed, masterpiece, detailed", vocab)
    assert out == "Detailed, masterpiece, a barn, ultra detailed"
    assert front_load_tags("a barn at dusk", vocab) == "a barn at dusk"


def test_text_embedder_drops_captions_per_sample():
    torch.manual_seed(0)
    emb = TextEmbedder(8, 16, dropout_prob=0.5)
    out = emb(torch.randn(256, 3, 8) + 1.0, train=True)
    dropped = (out.abs().sum(dim=(1, 2)) == 0).float().mean().item()
    assert 0.3 < dropped < 0.7
    assert (emb(torch.randn(4, 3, 8), train=False).abs().sum(dim=(1, 2)) > 0).all()


# ==================================================================================================
# test_training_objectives: Training-objective correctness: loss weights, noise offset, spectral loss, zero-terminal SNR, OT pairing, EMA.
# ==================================================================================================


def _abar_snr(n=64):
    d = create_diffusion(num_timesteps=1000)
    t = torch.linspace(0, 999, n).long()
    return d.alpha_cumprod[t], d.snr[t]


@pytest.mark.parametrize("scheme", ["min_snr", "min_snr_soft", "edm", "v", "eps"])
def test_weights_are_one_objective_in_every_prediction_space(scheme):
    """w_eps*|eps_err|^2 == w_v*|v_err|^2 == w_x0*|x0_err|^2 for the same denoising error."""
    abar, snr = _abar_snr()
    kw = dict(snr=snr, alpha_cumprod=abar, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5)
    x0_err = torch.randn(abar.shape[0])
    a, s = abar.sqrt(), (1 - abar).sqrt()
    eps_err = -a / s * x0_err  # x0_hat = (x_t - s*eps_hat)/a
    v_err = -x0_err / s  # x0_hat = a*x_t - s*v_hat
    l_x0 = get_timestep_loss_weight(scheme, prediction_type="x0", **kw) * x0_err**2
    l_eps = get_timestep_loss_weight(scheme, prediction_type="epsilon", **kw) * eps_err**2
    l_v = get_timestep_loss_weight(scheme, prediction_type="v", **kw) * v_err**2
    torch.testing.assert_close(l_eps, l_x0, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(l_v, l_x0, rtol=1e-3, atol=1e-6)


def test_min_snr_matches_hang_et_al_per_target():
    abar, snr = _abar_snr()
    kw = dict(snr=snr, alpha_cumprod=abar, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5)
    capped = snr.clamp(max=5.0)
    torch.testing.assert_close(get_timestep_loss_weight("min_snr", prediction_type="epsilon", **kw), capped / snr)
    torch.testing.assert_close(get_timestep_loss_weight("min_snr", prediction_type="v", **kw), capped / (snr + 1))
    torch.testing.assert_close(get_timestep_loss_weight("min_snr", prediction_type="x0", **kw), capped)


def test_edm_uses_ve_sigma_and_zero_snr_is_finite():
    abar, snr = _abar_snr()
    sig = ((1.0 - abar) / abar).sqrt()
    w = get_timestep_loss_weight(
        "edm", snr=snr, alpha_cumprod=abar, min_snr_gamma=0, loss_weighting_sigma_data=0.5, prediction_type="x0"
    )
    torch.testing.assert_close(w, (sig**2 + 0.25) / (sig * 0.5) ** 2, rtol=1e-4, atol=0)
    zero = get_timestep_loss_weight(
        "min_snr", snr=torch.zeros(2), alpha_cumprod=torch.zeros(2), min_snr_gamma=5.0, loss_weighting_sigma_data=0.5
    )
    assert torch.equal(zero, torch.ones(2))


class _Oracle(torch.nn.Module):
    """Predicts the exact eps of ``x_t`` for a known ``x0`` (zero loss iff the target is the noise used in ``x_t``)."""

    def __init__(self, diffusion, x0):
        super().__init__()
        self.d, self.x0 = diffusion, x0

    def forward(self, x_t, t, **kw):
        a = self.d.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        s = self.d.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return (x_t - a * self.x0) / s


def test_noise_offset_is_part_of_the_target():
    torch.manual_seed(0)
    d = create_diffusion(num_timesteps=1000)
    x0 = torch.randn(4, 4, 8, 8)
    t = torch.randint(0, 1000, (4,))
    loss = d.per_sample_training_losses(_Oracle(d, x0), x0, t, noise_offset=0.3, loss_weighting="unit")
    assert float(loss.max()) < 1e-6


def test_spectral_loss_has_mse_scale():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16)
    t = torch.tensor([10, 900])
    flat = spectral_sfp_per_sample_loss(pred, target, t, 1000, low_sigma=1e6, high_sigma=1e6)
    torch.testing.assert_close(flat, (pred - target).pow(2).mean(dim=(1, 2, 3)), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("base", ["linear", "squaredcos_cap_v2"])
def test_zero_terminal_snr_schedule(base):
    plain = np.cumprod(1 - get_beta_schedule(base, 1000))
    betas = get_beta_schedule(base + "_ztsnr", 1000)
    abar = np.cumprod(1 - betas)
    assert ((betas > 0) & (betas < 1)).all()
    assert np.all(np.diff(abar) < 0)
    assert abar[0] == pytest.approx(plain[0])
    assert abar[-1] <= 2.0**-24 * (1 + 1e-9)
    assert abar[-1] < plain[-1] or plain[-1] < 1e-7  # cosine schedules already end near zero


def test_ot_pairing_permutes_noise_and_lowers_cost():
    torch.manual_seed(0)
    x, n = torch.randn(8, 4, 8, 8), torch.randn(8, 4, 8, 8)
    flat = n.reshape(8, -1)
    exact = pair_noise_to_latents(x, n, reg=0.05, mode="hungarian")
    soft = pair_noise_to_latents(x, n, reg=0.05, mode="soft")
    for out in (exact, soft):
        rows = torch.cdist(out.reshape(8, -1), flat).argmin(dim=1)
        assert sorted(rows.tolist()) == list(range(8))  # same i.i.d. draws, only reordered
    cost = lambda e: float(((x - e) ** 2).sum())  # noqa: E731
    assert cost(exact) <= cost(n)
    assert cost(soft) == pytest.approx(cost(exact), rel=1e-3)


def test_ema_warmup():
    from training.loop import ema_decay_at

    assert ema_decay_at(0, 0.9999) == pytest.approx(0.1)
    assert ema_decay_at(100, 0.9999) == pytest.approx(101 / 110)
    assert ema_decay_at(10**7, 0.9999) == 0.9999


def test_enhanced_trainer_x0_estimate_inverts_add_noise():
    from training.enhanced_trainer import EnhancedTrainer

    trainer = EnhancedTrainer.__new__(EnhancedTrainer)
    trainer.diffusion = create_diffusion(num_timesteps=1000)
    x0, eps = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    t = torch.tensor([5, 400, 900])
    torch.testing.assert_close(trainer.predict_x0(trainer.add_noise(x0, eps, t), eps, t), x0, rtol=1e-3, atol=1e-3)


# ==================================================================================================
# test_training_wiring: Verify scrape → enrich → control → train field wiring (image pipeline only).
# ==================================================================================================


def test_pick_training_manifest_prefers_enriched(tmp_path: Path):
    combined = tmp_path / "combined" / "manifest.jsonl"
    enriched = tmp_path / "enriched" / "manifest.jsonl"
    combined.parent.mkdir(parents=True)
    enriched.parent.mkdir(parents=True)
    combined.write_text(json.dumps({"image_path": "a.png", "caption": "raw"}) + "\n", encoding="utf-8")
    enriched.write_text(
        json.dumps({"image_path": "a.png", "caption": "researched prompt", "tag_sources": ["creative_rag"]}) + "\n",
        encoding="utf-8",
    )
    assert pick_training_manifest(combined, enriched) == enriched


def test_pick_training_manifest_falls_back_to_combined(tmp_path: Path):
    combined = tmp_path / "combined" / "manifest.jsonl"
    enriched = tmp_path / "enriched" / "manifest.jsonl"
    combined.parent.mkdir(parents=True)
    enriched.parent.mkdir(parents=True)
    combined.write_text(json.dumps({"image_path": "a.png", "caption": "raw"}) + "\n", encoding="utf-8")
    enriched.write_text("", encoding="utf-8")
    assert pick_training_manifest(combined, enriched) == combined


def test_negative_caption_from_research_fields():
    row = {"negative_prompt_hint": "low quality, blurry"}
    assert negative_caption_from_row(row) == "low quality, blurry"
    row2 = {"negative_caption": "bad anatomy", "negative_prompt_hint": "ignored"}
    assert negative_caption_from_row(row2) == "bad anatomy"


def test_dataset_reads_negative_prompt_hint(tmp_path: Path):
    import numpy as np
    from data.t2i_dataset import Text2ImageDataset
    from PIL import Image

    root = tmp_path / "data"
    root.mkdir(parents=True)
    img = root / "img.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img)
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_path": str(img),
                "caption": "1girl, solo",
                "negative_prompt_hint": "worst quality, blurry",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ds = Text2ImageDataset(str(manifest), image_size=32)
    item = ds[0]
    assert "blurry" in item["negative_caption"].lower()


def test_control_manifest_row_shape(tmp_path: Path):
    """Control preprocess output must include fields train.py reads."""
    from setup.manifests import _extract_one

    data_root = tmp_path / "sdx"
    img_dir = data_root / "danbooru" / "images"
    img_dir.mkdir(parents=True)
    import numpy as np
    from PIL import Image

    img = img_dir / "abc.png"
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img)
    row = {
        "image_path": "danbooru/images/abc.png",
        "caption": "researched diffusion prompt",
        "negative_caption": "blurry",
        "md5": "abc",
    }
    out = _extract_one(data_root, row, control_type="canny", controls_dir=data_root / "controls")
    assert out is not None
    assert out.get("control_image")
    assert out.get("control_type") == "canny"
    assert out.get("caption") == "researched diffusion prompt"
    assert out.get("negative_caption") == "blurry"


def test_pipeline_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "train.py",
        "setup/download.py",
        "setup/manifests.py",
        "setup/indexes.py",
        "scripts/run_pipeline.py",
        "runpod/download.sh",
        "runpod/train.sh",
        "runpod/sample.sh",
    ]
    missing = [p for p in required if not (root / p).is_file()]
    assert not missing, f"missing pipeline files: {missing}"


def test_train_cli_has_manifest_arg():
    from training.cli import build_train_arg_parser

    p = build_train_arg_parser()
    args = p.parse_args(["--manifest-jsonl", "x.jsonl", "--data-path", "/data"])
    assert args.manifest_jsonl == "x.jsonl"


def test_enriched_row_fields_survive_merge_research():
    from setup.manifests import _merge_research_row
    from utils.captioning import PromptResearchResult

    row = {
        "caption": "1girl, solo",
        "character_tags": ["miku"],
        "copyright_tags": ["vocaloid"],
    }
    researched = PromptResearchResult(
        diffusion_prompt="idol on stage, spotlight",
        negative_prompt="low quality",
        sources=["creative_rag"],
    )
    merged = _merge_research_row(row, researched)
    assert "miku" in merged["caption"].lower()
    assert merged["negative_caption"] == "low quality"
    assert merged["booru_caption"] == "1girl, solo"


# ==================================================================================================
# test_ar_curriculum: Tests for utils.training.objectives runtime resolution.
# ==================================================================================================


def test_parse_ar_order_mix_filters_and_dedupes():
    out = parse_ar_order_mix("raster,zorder,snake,spiral,zorder,invalid")
    assert out == ["raster", "zorder", "snake", "spiral"]


def test_resolve_ar_for_step_linear_and_mix():
    # linear ramp 0 -> 4 over steps; order cycles deterministic.
    b0, o0 = resolve_ar_for_step(
        0,
        base_blocks=2,
        base_order="raster",
        curriculum_mode="linear",
        ramp_start=0,
        ramp_end=100,
        curriculum_start_blocks=0,
        curriculum_target_blocks=4,
        order_mix="raster,snake",
    )
    b1, o1 = resolve_ar_for_step(
        60,
        base_blocks=2,
        base_order="raster",
        curriculum_mode="linear",
        ramp_start=0,
        ramp_end=100,
        curriculum_start_blocks=0,
        curriculum_target_blocks=4,
        order_mix="raster,snake",
    )
    assert b0 == 0 and o0 in ("raster", "snake")
    assert b1 in (2, 4) and o1 in ("raster", "snake")


def test_resolve_ar_for_step_none_uses_base():
    b, o = resolve_ar_for_step(
        999,
        base_blocks=2,
        base_order="zorder",
        curriculum_mode="none",
    )
    assert b == 2
    assert o == "zorder"


# ==================================================================================================
# test_device_perf: Device / perf helpers (no GPU required for string/dataclass checks).
# ==================================================================================================


def test_parallel_train_torchrun_example_shape() -> None:
    s = parallel_train_torchrun_example(3, extra="--data-path x")
    assert "--standalone" in s and "--nproc_per_node=3" in s
    assert "--data-path x" in s and s.strip().startswith("python")


def test_training_perf_hints_keys() -> None:
    h = training_perf_hints(num_workers=4, world_size=2)
    assert "world_size=2" in h.ddp_note
    assert "expandable_segments" in h.vram_fragmentation_note
    assert "num_workers=4" in h.datloader_note


def test_dataloader_perf_kwargs_prefetch() -> None:
    kw = dataloader_perf_kwargs(num_workers=4, prefetch_factor=3)
    assert kw["num_workers"] == 4
    assert kw["prefetch_factor"] == 3
    assert kw["pin_memory"] is False or kw["pin_memory"] is True
    kw0 = dataloader_perf_kwargs(num_workers=0, prefetch_factor=2)
    assert "prefetch_factor" not in kw0


def test_resolve_training_num_workers_auto() -> None:
    assert resolve_training_num_workers(-1, 10_000, 32, auto=False) >= 2
    assert resolve_training_num_workers(6, 10_000, 32, auto=False) == 6


@pytest.mark.parametrize("cpu_count, expected", [(1, 0), (2, 0), (3, 1), (8, 6)])
def test_auto_workers_respect_cpu_capacity(monkeypatch, cpu_count: int, expected: int) -> None:
    monkeypatch.setattr("utils.training.perf.os.cpu_count", lambda: cpu_count)
    assert optimal_num_workers(10_000, 32) == expected
    assert optimal_num_workers(10_000, 32, max_workers=1, min_workers=4) == min(expected, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_conditioning_kwargs_match_latent_dtype(dtype: torch.dtype) -> None:
    from training.loop import conditioning_kwargs

    latents = torch.zeros((2, 4, 8, 8), dtype=dtype)
    cfg = SimpleNamespace(creativity_embed_dim=8, creativity_max=1.0)
    result = conditioning_kwargs(
        cfg,
        {"control_image": torch.zeros((2, 3, 8, 8))},
        latents,
        torch.device("cpu"),
        encoder_hidden=torch.zeros((2, 4, 8), dtype=dtype),
        encoder_hidden_neg=None,
        style_embedding=None,
        token_weights=None,
        creativity_jitter_std=0.1,
    )
    assert result["control_image"].dtype == dtype
    assert result["creativity"].dtype == dtype


def test_encode_text_multi_group() -> None:
    calls: list = []

    def _fake_encode(caps: list) -> list:
        calls.append(list(caps))
        return [[len(c)] for c in caps]

    g0, g1, g2 = encode_text_multi_group([["p1", "p2"], None, ["s1"]], _fake_encode)
    assert calls == [["p1", "p2", "s1"]]
    assert g0 == [[2], [2]]
    assert g1 is None
    assert g2 == [[2]]


# ==================================================================================================
# test_error_handling_utils: Tests for ``utils.training.monitoring`` helpers.
# ==================================================================================================


def test_get_model_info_linear() -> None:
    m = torch.nn.Linear(4, 2, bias=True)
    info = get_model_info(m)
    assert info["total_parameters"] == 10  # 4*2 + 2
    assert info["trainable_parameters"] == 10
    assert "model_size_mb" in info
    assert info["device"] == torch.device("cpu")


# ==================================================================================================
# test_preference_rl: Preference-RL weighting (GRPO / GRPO-Guard / SRPO) and the Flow-GRPO SDE rollout step.
# ==================================================================================================


def _grad_per_sample(loss_fn) -> torch.Tensor:
    per = torch.ones(4, requires_grad=True)
    loss_fn(per).backward()
    return per.grad


def test_grpo_fits_high_reward_rollouts_hardest():
    adv = torch.tensor([2.0, 0.5, -0.5, -2.0])
    g = _grad_per_sample(lambda per: grpo_weighted_loss(per, adv))
    assert torch.all(g[:-1] > g[1:])


def test_grpo_needs_the_group():
    with pytest.raises(ValueError):
        grpo_weighted_loss(torch.ones(1), torch.ones(1))


def test_grpo_guard_weights_increase_with_advantage():
    adv = torch.tensor([2.0, 0.5, -0.5, -2.0])
    cfg = GRPOGuardConfig(grad_reweight=False)
    g = _grad_per_sample(lambda per: grpo_guard_weighted_loss(per, adv, torch.full((4,), 0.5), config=cfg))
    assert torch.all(g[:-1] >= g[1:]) and g[0] > g[-1]


def test_flow_sde_keeps_the_flow_marginals():
    """With the exact velocity of 1-D Gaussian data, the SDE must still land on the data distribution."""
    torch.manual_seed(0)
    data_std = 0.5

    def velocity(x, s):  # E[eps - x0 | x_s] for x0 ~ N(0, data_std^2)
        var = (1 - s) ** 2 * data_std**2 + s**2
        return x * (s - (1 - s) * data_std**2) / var

    x = torch.randn(20000)
    grid = torch.linspace(1.0, 0.0, 201).tolist()
    for s_cur, s_next in zip(grid[:-1], grid[1:]):
        v = velocity(x, s_cur)
        x = flow_sde_step(x, v, s_cur, s_next, noise_level=0.7) if s_next > 0 else x - s_cur * v
    assert float(x.std()) == pytest.approx(data_std, rel=0.05)


# ==================================================================================================
# test_srpo_reward: SRPO realism reward + late-timestep mask (``utils.training.preference``, wired via ``train_flow_grpo --srpo``).
# ==================================================================================================


def test_semantic_relative_reward_cancels_global_bias():
    def biased_score(images, text):
        return torch.full((images.shape[0],), 10.0) + (1.0 if "realistic" in text else 0.0)

    cfg = SRPOConfig(control_positive="realistic", control_negative="plastic")
    r = semantic_relative_reward(biased_score, torch.zeros(3, 3, 8, 8), "portrait", config=cfg)
    assert torch.allclose(r, torch.ones(3))


def test_semantic_relative_reward_works_on_float_scores():
    cfg = SRPOConfig(control_positive="natural", control_negative="waxy")
    assert semantic_relative_reward(lambda img, text: 0.7 if "natural" in text else 0.4, None, "", config=cfg) == (
        pytest.approx(0.3)
    )


def test_timestep_mask_zeroes_late_steps():
    mask = timestep_reward_mask(torch.tensor([0.1, 0.5, 0.8, 0.95]), cutoff=0.75)
    assert mask.tolist() == [1.0, 1.0, 0.0, 0.0]


# ==================================================================================================
# test_timestep_curriculum: Tests for ``utils.training.objectives``.
# ==================================================================================================


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


# --- Resumable checkpoints, non-finite guard, run manifest, baseline defaults ---


def _tiny_run(ckpt_path: Path, total: int, *, stop_at: int | None = None, resume: bool = False):
    """train.py's resume path in miniature: fresh objects, checkpoint restore, sampler skip, RNG restore."""
    import copy
    import logging
    import random as pyrandom

    from torch.utils.data import DataLoader, TensorDataset
    from training.loop import (
        ResumableSampler,
        repeat_epochs,
        restore_trainer_state,
        resume_position,
        trainer_state,
        update_ema,
    )
    from utils.modeling.checkpoints import atomic_torch_save, load_checkpoint_file, set_rng_state

    torch.manual_seed(0)
    pyrandom.seed(0)
    np.random.seed(0)
    model = torch.nn.Linear(4, 2)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    g = torch.Generator().manual_seed(123)
    data = TensorDataset(torch.randn(10, 4, generator=g), torch.randn(10, 2, generator=g))
    sampler = ResumableSampler(data, num_replicas=1, rank=0, shuffle=True, seed=7)
    loader = DataLoader(data, batch_size=3, sampler=sampler, drop_last=True, generator=torch.Generator().manual_seed(7))
    spe = len(loader)
    steps, epoch = 0, 0
    if resume:
        ckpt = load_checkpoint_file(ckpt_path)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        restore_trainer_state(ckpt, opt, scaler, logging.getLogger("test"))
        steps = ckpt["steps"]
        epoch, batch = resume_position(ckpt, spe)
        sampler.skip = batch * 3
        set_rng_state(ckpt["rng"])
    for x, y in repeat_epochs(loader, sampler, epoch):
        noise = torch.randn_like(x) * 0.1 + pyrandom.random() * 0.01 + float(np.random.rand()) * 0.01
        loss = ((model(x + noise) - y) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        update_ema(ema, model, 0.9)
        steps += 1
        if steps == stop_at:
            state = trainer_state(opt, scaler, steps=steps, steps_per_epoch=spe)
            atomic_torch_save({"model": model.state_dict(), "ema": ema.state_dict(), **state}, ckpt_path)
            return None
        if steps == total:
            return model, ema


def test_resume_matches_uninterrupted_run(tmp_path):
    straight = _tiny_run(tmp_path / "unused.pt", 8)
    ckpt = tmp_path / "last.pt"
    _tiny_run(ckpt, 8, stop_at=4)  # stops mid-epoch (3 batches per epoch)
    resumed = _tiny_run(ckpt, 8, resume=True)
    for a, b in zip(straight, resumed, strict=True):
        for pa, pb in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
            assert torch.equal(pa, pb)


def test_atomic_save_keeps_previous_checkpoint_when_write_crashes(tmp_path, monkeypatch):
    from utils.modeling.checkpoints import atomic_torch_save

    path = tmp_path / "last.pt"
    atomic_torch_save({"steps": 1}, path)

    def crash(obj, f):
        f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", crash)
    with pytest.raises(OSError):
        atomic_torch_save({"steps": 2}, path)
    monkeypatch.undo()
    assert torch.load(path)["steps"] == 1
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]


def test_old_checkpoint_resumes_with_warnings(caplog):
    import logging

    from training.loop import restore_trainer_state, resume_position

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    old = {"model": model.state_dict(), "ema": model.state_dict(), "steps": 7}  # pre-resume-state checkpoint
    with caplog.at_level(logging.WARNING):
        restore_trainer_state(old, opt, scaler, logging.getLogger("test"))
    text = caplog.text
    assert "no optimizer state" in text and "no GradScaler state" in text and "no RNG state" in text
    assert resume_position(old, 3) == (2, 1)


def test_bucket_sampler_rebuilds_any_epoch_and_skips():
    ds = SimpleNamespace(resolution_buckets=[(64, 64)], _bucket_assign=[0] * 6 + [1] * 4)
    s = ResolutionBucketBatchSampler(ds, 2, generator=torch.Generator().manual_seed(3))
    s.set_epoch(2)
    full = list(s)
    list(s)  # the generator moved on; set_epoch rebuilds the same order
    s.set_epoch(2)
    s.skip = 2
    assert list(s) == full[2:]
    assert len(list(s)) == len(full)  # skip applies to one pass only


def test_nonfinite_guard_skips_then_aborts():
    from training.loop import NonFiniteGuard

    guard = NonFiniteGuard(3)
    assert guard.ok(torch.tensor(1.0), torch.tensor(2.0), 1)
    assert not guard.ok(torch.tensor(float("nan")), torch.tensor(2.0), 2)
    assert not guard.ok(torch.tensor(1.0), torch.tensor(float("inf")), 3)
    assert guard.ok(1.0, 1.0, 4) and guard.consecutive == 0 and guard.skipped == 2
    guard.ok(float("nan"), 1.0, 5)
    guard.ok(float("nan"), 1.0, 6)
    with pytest.raises(FloatingPointError, match="3 consecutive"):
        guard.ok(float("nan"), 1.0, 7)
    assert NonFiniteGuard(0).ok(float("nan"), 1.0, 1) is False  # 0 = never abort


def test_fixed_seed_repeats_draws_and_restores_training_rng():
    from training.loop import fixed_seed

    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    with fixed_seed(1):
        a = torch.rand(2)
    with fixed_seed(1):
        b = torch.rand(2)
    assert torch.equal(a, b)
    assert torch.equal(torch.rand(3), expected)


def test_run_manifest_records_environment_and_dataset(tmp_path):
    import logging

    from config.train_config import TrainConfig
    from training.loop import dataset_fingerprint, write_run_manifest

    data = tmp_path / "data"
    data.mkdir()
    (data / "a.png").write_bytes(b"x")
    (data / "a.txt").write_text("a cat", encoding="utf-8")
    write_run_manifest(tmp_path, TrainConfig(data_path=str(data)), logging.getLogger("test"))
    m = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["packages"]["torch"] and m["dataset"]["files"] == 2 and m["config"]["data_path"] == str(data)
    assert {"git", "argv", "system", "seed", "cudnn_version"} <= m.keys()
    before = dataset_fingerprint(data)["sha256"]
    (data / "a.txt").write_text("a black cat", encoding="utf-8")
    assert dataset_fingerprint(data)["sha256"] != before


def test_experimental_training_features_default_off():
    from training.cli import build_train_arg_parser, build_train_config_from_args

    cfg = build_train_config_from_args(build_train_arg_parser().parse_args(["--data-path", "x"]))
    off = (
        "refinement_prob img2img_prob edit_conditioning mdm_mask_ratio mdm_mask_schedule moe_num_experts "
        "moe_balance_loss_weight num_ar_blocks ar_order_mix style_embed_dim control_cond_dim creativity_embed_dim "
        "creativity_jitter_std size_embed_dim patch_se curriculum_difficulty_steps curriculum_caption_steps repa_weight "
        "ssm_every_n num_register_tokens use_rope token_routing_enabled drop_path_rate layerscale_init qk_norm "
        "lora_train noise_offset spectral_sfp_loss ot_noise_pair_reg flow_matching_training bridge_aux_weight "
        "timestep_curriculum_schedule caption_dropout_schedule resolution_buckets boost_adherence_caption "
        "caption_unicode_normalize train_prompt_emphasis attn_grounding_loss_weight attn_token_coverage_loss_weight "
        "prompt_reinject_every_n prompt_reinject_alpha prompt_timestep_schedule_enabled use_hierarchical_captions "
        "hierarchical_caption_drop_global_p hierarchical_caption_drop_local_p foveated_train_prob "
        "train_originality_augment_prob save_polyak"
    ).split()
    assert [k for k in off if getattr(cfg, k)] == []
    none_modes = (
        "ar_curriculum_mode train_shortcomings_mitigation train_art_guidance_mode train_anatomy_guidance "
        "train_style_guidance_mode"
    ).split()
    assert [k for k in none_modes if getattr(cfg, k) != "none"] == []
    assert (cfg.kv_merge_factor, cfg.token_keep_ratio, cfg.timestep_sample_mode) == (1, 1.0, "uniform")
    assert (cfg.beta_schedule, cfg.autoencoder_type, cfg.text_encoder_mode) == ("linear", "kl", "t5")


# ==================================================================================================
# Training readiness: objective math, timestep sampling, caption dropout, gradient reach, latent scaling,
# dataset audit, loader skip counting and the readiness gate (training/readiness.py, data/audit.py).
# ==================================================================================================


def _train_cfg(*argv: str):
    from training.cli import build_train_arg_parser, build_train_config_from_args

    return build_train_config_from_args(build_train_arg_parser().parse_args(list(argv)))


class _FlowOracle(torch.nn.Module):
    """Exact velocity of the straight path through a known ``x0``: ``(x_t - x0) / s``."""

    def __init__(self, x0, T):
        super().__init__()
        self.x0, self.T = x0, T

    def forward(self, x_t, t, **kw):
        s = (t.float() / (self.T - 1)).view(-1, 1, 1, 1).clamp_min(1e-6)
        return (x_t - self.x0) / s


def test_flow_interpolation_target_and_sampler_agree():
    from diffusion.flow_matching import flow_matching_per_sample_losses

    torch.manual_seed(0)
    T, x0, eps = 1000, torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    seen = {}

    class _Probe(torch.nn.Module):
        def forward(self, x_t, t, **kw):
            seen[int(t[0])] = x_t
            return eps - x0

    for t in (0, 400, 999):  # x_t is the straight line from x0 (t=0) to eps (t=T-1); its slope is the target
        loss = flow_matching_per_sample_losses(_Probe(), x0, eps, T, {}, t_idx=torch.full((3,), t))
        assert float(loss.max()) == 0.0
    torch.testing.assert_close(seen[0], x0)
    torch.testing.assert_close(seen[999], eps)
    torch.testing.assert_close((seen[999] - seen[400]) / (599 / 999), eps - x0, rtol=1e-4, atol=1e-4)
    # The sampler integrates the same convention: an exact velocity field lands on x0.
    d = create_diffusion(num_timesteps=T)
    out = d.sample_loop(
        _FlowOracle(x0[:1], T),
        (1, 4, 8, 8),
        model_kwargs_cond={},
        cfg_scale=1.0,
        num_inference_steps=20,
        device="cpu",
        flow_matching_sample=True,
        timestep_schedule="euler",
    )
    torch.testing.assert_close(out, x0[:1], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("pred", ["epsilon", "v", "x0"])
def test_vp_q_sample_target_and_inversion_are_consistent(pred):
    torch.manual_seed(0)
    d = create_diffusion(num_timesteps=1000, prediction_type=pred)
    x0, eps = torch.randn(4, 4, 8, 8), torch.randn(4, 4, 8, 8)
    t = torch.tensor([0, 250, 600, 999])
    x_t = d.q_sample(x0, t, noise=eps)
    a, s = d.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1), d.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
    torch.testing.assert_close(x_t, a * x0 + s * eps)
    target = d.training_target(x0, eps, t)
    x0_hat, eps_hat = d._predict_x0_and_noise(target, x_t, t)
    torch.testing.assert_close(x0_hat[1:3], x0[1:3], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(eps_hat[1:3], eps[1:3], rtol=1e-3, atol=1e-3)

    class _Perfect(torch.nn.Module):
        def forward(self, x, tt, **kw):
            return target

    loss = d.per_sample_training_losses(_Perfect(), x0, t, noise=eps, loss_weighting="min_snr")
    assert float(loss.max()) == 0.0


def test_training_timesteps_follow_configured_distribution():
    from training.loop import flow_shift_for_latents, step_timesteps

    torch.manual_seed(0)
    T, n = 1000, 40000
    cfg = SimpleNamespace(timestep_sample_mode="logit_normal", timestep_logit_mean=0.5, timestep_logit_std=0.7)
    t = step_timesteps(cfg, T, n, torch.device("cpu")).float()
    assert t.min() >= 0 and t.max() <= T - 1
    u = (t / (T - 1)).clamp(1e-3, 1 - 1e-3)
    logit = torch.log(u / (1 - u))
    assert abs(float(logit.mean()) - 0.5) < 0.02 and abs(float(logit.std()) - 0.7) < 0.02
    uni = step_timesteps(SimpleNamespace(timestep_sample_mode="uniform"), T, n, torch.device("cpu")).float()
    assert abs(float(uni.mean()) - (T - 1) / 2) < 10 and abs(float(uni.std()) - T / 12**0.5) < 5
    flow = SimpleNamespace(image_size=256, flow_shift=3.0)
    assert flow_shift_for_latents(flow, torch.zeros(1, 4, 32, 32)) == pytest.approx(3.0)
    assert flow_shift_for_latents(flow, torch.zeros(1, 4, 64, 64)) == pytest.approx(6.0)  # sqrt(4x tokens)


def test_caption_dropout_rate_and_null_prompt():
    from training.loop import drop_captions

    torch.manual_seed(0)
    caps = [f"cap {i}" for i in range(20000)]
    layouts = [{"i": i} for i in range(20000)]
    out, lay = drop_captions(caps, 0.1, layouts)
    dropped = [c == "" for c in out]
    assert abs(sum(dropped) / len(caps) - 0.1) < 0.008
    assert all((c == "") == (spec is None) for c, spec in zip(out, lay, strict=True))
    assert all(c in ("", orig) for c, orig in zip(out, caps, strict=True))
    assert drop_captions(caps, 0.0) == (caps, None)
    # The dropped prompt is the sampler's CFG uncond: sample.py encodes the (default empty) negative prompt.
    from utils.generation.sample_cli_parser import build_sample_parser

    assert build_sample_parser().get_default("negative_prompt") == ""


def test_gradients_reach_every_trainable_parameter():
    from config.train_config import get_dit_build_kwargs
    from models.dit_text import DiT_Text
    from training.loop import conditioning_kwargs, diffusion_step_loss, step_timesteps, training_noise

    torch.manual_seed(0)
    cfg = _train_cfg("--data-path", "x", "--image-size", "64")
    kw = {**get_dit_build_kwargs(cfg, class_dropout_prob=0.0), "text_dim": 32}
    model = DiT_Text(**kw, depth=2, hidden_size=64, num_heads=2, patch_size=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    d = create_diffusion(num_timesteps=1000)
    lat, enc = torch.randn(4, 4, 8, 8), torch.randn(4, 7, 32)
    seen = {n: 0.0 for n, p in model.named_parameters() if p.requires_grad}
    for step in range(4):  # adaLN-Zero gates are zero at init, so upstream grads appear after the first steps
        mk = conditioning_kwargs(
            cfg, {}, lat, "cpu", encoder_hidden=enc, encoder_hidden_neg=None, style_embedding=None, token_weights=None
        )
        t = step_timesteps(cfg, 1000, 4, torch.device("cpu"), global_step=step)
        loss = diffusion_step_loss(
            cfg, d, model, lat, t, mk, None, training_noise(cfg, lat), refinement_prob=0.0, refinement_max_t=0
        )
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is None or torch.isfinite(p.grad).all(), n
                seen[n] += 0.0 if p.grad is None else float(p.grad.abs().sum())
        opt.step()
    assert [n for n, v in seen.items() if v == 0.0] == []


def test_latent_scale_applied_on_encode_and_inverted_on_decode():
    from training.loop import encode_images_vae
    from utils.generation.guidance import decode_latent_preview_rgb
    from utils.modeling.encoders import ShiftedAutoencoder

    class _IdVAE(torch.nn.Module):  # latent == image, decode == latent
        def encode(self, x):
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda g=None: x.clone()))

        def decode(self, z):
            return SimpleNamespace(sample=z)

    x = torch.rand(1, 3, 8, 8) * 2 - 1
    z = encode_images_vae(x, _IdVAE(), 0.13)
    torch.testing.assert_close(z, x * 0.13)
    img = decode_latent_preview_rgb(z, vae=_IdVAE(), latent_scale=0.13, ae_type="kl", rae_bridge=None)
    torch.testing.assert_close(img, (x[0] * 0.5 + 0.5).clamp(0, 1))
    shifted = ShiftedAutoencoder(_IdVAE(), 0.2)  # FLUX/SD3 VAEs: encode subtracts the shift, decode adds it back
    torch.testing.assert_close(encode_images_vae(x, shifted, 0.5), (x - 0.2) * 0.5)
    torch.testing.assert_close(shifted.decode(encode_images_vae(x, shifted, 0.5) / 0.5).sample, x)


def _png(path: Path, size=(96, 96), mode="RGB", seed=0, caption: str | None = "a red cube on a wooden table"):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], len(mode)), dtype=np.uint8)
    Image.fromarray(arr.squeeze(), mode=mode).save(path)
    if caption is not None:
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")


def test_dataset_audit_finds_planted_problems(tmp_path):
    from data.audit import audit_dataset

    d = tmp_path / "data"
    d.mkdir()
    for i in range(6):
        _png(d / f"ok{i}.png", seed=i)
    (d / "dup.png").write_bytes((d / "ok0.png").read_bytes())
    (d / "dup.txt").write_text("a copy", encoding="utf-8")
    arr = np.asarray(Image.open(d / "ok1.png")).astype(np.int16)
    Image.fromarray(np.clip(arr + 3, 0, 255).astype(np.uint8)).save(d / "near.png")  # same picture, re-encoded
    (d / "near.txt").write_text("almost a copy", encoding="utf-8")
    (d / "corrupt.png").write_bytes((d / "ok2.png").read_bytes()[:200])
    (d / "corrupt.txt").write_text("broken file", encoding="utf-8")
    _png(d / "alpha.png", mode="RGBA", seed=7)
    _png(d / "small.png", size=(20, 20), seed=8)
    _png(d / "wide.png", size=(400, 80), seed=9)
    _png(d / "nocap.png", seed=10, caption=None)
    _png(d / "short.png", seed=11, caption="cat")
    cfg = _train_cfg("--data-path", str(d), "--image-size", "64")
    rep = audit_dataset(cfg, max_bad_frac=0.5)
    c = rep["counts"]
    assert c["missing_caption"] == 1 and c["corrupt"] == 1 and c["loader_failed"] == 1
    assert c["exact_duplicate_rows"] == 1 and c["near_duplicate_rows"] == 1
    assert c["alpha_flattened"] == 1 and c["upscaled_small_image"] == 1 and c["extreme_aspect_crop"] == 1
    assert c["short_caption"] == 1
    assert rep["ok"] and rep["rows_total"] == 14
    assert not audit_dataset(cfg, max_bad_frac=0.01)["ok"]  # 2 of 14 rows are dropped or unreadable


def test_loader_counts_silently_replaced_samples(tmp_path):
    from data import collate_t2i

    _png(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes(b"not an image")
    (tmp_path / "b.txt").write_text("broken", encoding="utf-8")
    ds = Text2ImageDataset(str(tmp_path), image_size=32)
    items = [ds[i] for i in range(len(ds))]
    assert collate_t2i(items)["load_skips"] == 1
    assert "load_skips" not in collate_t2i([i for i in items if "load_skips" not in i])


def test_readiness_gate_blocks_long_runs_without_ready_report(tmp_path, monkeypatch):
    from training import readiness

    d = tmp_path / "data"
    d.mkdir()
    _png(d / "a.png")
    monkeypatch.setattr(readiness, "git_info", lambda: {"commit": "abc", "dirty": False})
    base = ("--data-path", str(d), "--results-dir", str(tmp_path / "res"))
    readiness.enforce_readiness_gate(_train_cfg(*base, "--max-steps", "500"), min_steps=2000, skip=False)
    readiness.enforce_readiness_gate(_train_cfg(*base, "--dry-run"), min_steps=2000, skip=False)
    long_cfg = _train_cfg(*base, "--max-steps", "50000")
    with pytest.raises(SystemExit, match="readiness check"):
        readiness.enforce_readiness_gate(long_cfg, min_steps=2000, skip=False)
    readiness.enforce_readiness_gate(long_cfg, min_steps=2000, skip=True)
    from data.audit import dataset_fingerprint

    path = readiness.report_path(long_cfg, readiness.config_hash(long_cfg), dataset_fingerprint(long_cfg))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ready": True, "git": {"commit": "abc"}}), encoding="utf-8")
    readiness.enforce_readiness_gate(long_cfg, min_steps=2000, skip=False)
    # Run length does not change the identity; the objective does.
    assert readiness.config_hash(_train_cfg(*base, "--max-steps", "9")) == readiness.config_hash(long_cfg)
    assert readiness.config_hash(_train_cfg(*base, "--lr", "3e-3")) != readiness.config_hash(long_cfg)
    monkeypatch.setattr(readiness, "git_info", lambda: {"commit": "def", "dirty": False})
    with pytest.raises(SystemExit, match="another git commit"):
        readiness.enforce_readiness_gate(long_cfg, min_steps=2000, skip=False)


def test_readiness_check_end_to_end_on_cpu(tmp_path):
    pytest.importorskip("diffusers")
    pytest.importorskip("tokenizers")
    from training.readiness import run_readiness

    d = tmp_path / "data"
    d.mkdir()
    for i in range(4):
        _png(d / f"img{i}.png", size=(64, 64), seed=i, caption=f"a red cube number {i} on a table")
    cfg = _train_cfg("--data-path", str(d), "--image-size", "64", "--results-dir", str(tmp_path / "res"))
    rep = run_readiness(cfg, device="cpu", overfit_steps=120, overfit_samples=4)
    status = {c["name"]: c["status"] for c in rep["checks"]}
    assert rep["ready"], status
    assert status["throughput"] == "SKIP" and Path(rep["path"]).is_file()
