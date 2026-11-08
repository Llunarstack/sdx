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
