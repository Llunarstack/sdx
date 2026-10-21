"""Training data conditioning: aspect-ratio buckets, bucket batches, caption tag front-loading, caption dropout.

Also holds what used to be: test_training_data, test_training_objectives, test_training_wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from data import ResolutionBucketBatchSampler, Text2ImageDataset
from data.caption_utils import front_load_tags
from data.manifest_utils import negative_caption_from_row, pick_training_manifest
from data.t2i_dataset import nearest_aspect_bucket
from diffusion import create_diffusion
from diffusion.schedules import get_beta_schedule
from diffusion.spectral_sfp import spectral_sfp_per_sample_loss
from diffusion.timestep_sampling import get_timestep_loss_weight
from models.dit_text import TextEmbedder
from PIL import Image
from utils.training.ot_noise_pairing import pair_noise_to_latents

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
    from train import ema_decay_at

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
    from setup.preprocess_control_maps import _extract_one

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
        "setup/download_datasets.py",
        "setup/merge_manifests.py",
        "setup/enrich_manifest_captions.py",
        "setup/build_rag_corpus.py",
        "setup/preprocess_control_maps.py",
        "setup/build_artist_index.py",
        "scripts/run_pipeline.py",
        "runpod/download.sh",
        "runpod/train.sh",
        "runpod/sample.sh",
    ]
    missing = [p for p in required if not (root / p).is_file()]
    assert not missing, f"missing pipeline files: {missing}"


def test_train_cli_has_manifest_arg():
    from training.train_cli_parser import build_train_arg_parser

    p = build_train_arg_parser()
    args = p.parse_args(["--manifest-jsonl", "x.jsonl", "--data-path", "/data"])
    assert args.manifest_jsonl == "x.jsonl"


def test_enriched_row_fields_survive_merge_research():
    from setup.enrich_manifest_captions import _merge_research_row
    from utils.caption.prompt_research import PromptResearchResult

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
