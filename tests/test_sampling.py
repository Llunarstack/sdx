"""
Tests for ``diffusion.sampling`` (holy-grail recipe) and the latent helpers in ``diffusion.sampling_utils``.

Verifies per-step plans, presets, runtime guard clamps, CADS and latent refinement helpers.

Run with:
    pytest tests/test_sampling.py -v
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Blueprint / step plan
# ---------------------------------------------------------------------------


class TestHolyGrailBlueprint:
    def test_step_plan_at_start(self):
        from diffusion.sampling import HolyGrailRecipe, build_holy_grail_step_plan

        recipe = HolyGrailRecipe(base_cfg=7.5)
        plan = build_holy_grail_step_plan(recipe=recipe, step_index=0, total_steps=50)
        assert plan.cfg_scale > 0
        assert plan.control_scale > 0
        assert plan.adapter_scale > 0

    def test_step_plan_at_end(self):
        from diffusion.sampling import HolyGrailRecipe, build_holy_grail_step_plan

        recipe = HolyGrailRecipe(base_cfg=7.5)
        plan = build_holy_grail_step_plan(recipe=recipe, step_index=49, total_steps=50)
        assert plan.cfg_scale > 0

    def test_cfg_increases_over_steps(self):
        """With cfg_early_ratio < cfg_late_ratio, CFG should increase over steps."""
        from diffusion.sampling import HolyGrailRecipe, build_holy_grail_step_plan

        recipe = HolyGrailRecipe(base_cfg=7.5, cfg_early_ratio=0.7, cfg_late_ratio=1.0)
        plan_start = build_holy_grail_step_plan(recipe=recipe, step_index=0, total_steps=50)
        plan_end = build_holy_grail_step_plan(recipe=recipe, step_index=49, total_steps=50)
        assert plan_end.cfg_scale > plan_start.cfg_scale

    def test_single_step(self):
        from diffusion.sampling import HolyGrailRecipe, build_holy_grail_step_plan

        recipe = HolyGrailRecipe(base_cfg=7.5)
        plan = build_holy_grail_step_plan(recipe=recipe, step_index=0, total_steps=1)
        assert plan.cfg_scale > 0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestHolyGrailPresets:
    def test_list_presets(self):
        from diffusion.sampling import list_holy_grail_presets

        presets = list_holy_grail_presets()
        assert "balanced" in presets
        assert "anime" in presets
        assert "photoreal" in presets

    def test_get_preset(self):
        from diffusion.sampling import get_holy_grail_preset

        p = get_holy_grail_preset("anime")
        assert p.name == "anime"
        assert p.cfg_early_ratio > 0

    def test_get_unknown_preset_raises(self):
        from diffusion.sampling import get_holy_grail_preset

        with pytest.raises(KeyError):
            get_holy_grail_preset("nonexistent_preset_xyz")

    def test_preset_values_sane(self):
        from diffusion.sampling import HOLY_GRAIL_PRESETS

        for name, p in HOLY_GRAIL_PRESETS.items():
            assert 0 < p.cfg_early_ratio <= 2.0, f"{name}: bad cfg_early_ratio"
            assert 0 < p.cfg_late_ratio <= 2.0, f"{name}: bad cfg_late_ratio"
            assert p.cads_strength >= 0, f"{name}: negative cads_strength"
            assert 0 < p.clamp_floor, f"{name}: non-positive clamp_floor"


# ---------------------------------------------------------------------------
# Runtime guard
# ---------------------------------------------------------------------------


class TestRuntimeGuard:
    def test_sanitize_clamps_cfg_ratios(self):
        from diffusion.sampling import sanitize_holy_grail_kwargs

        out = sanitize_holy_grail_kwargs({"holy_grail_cfg_early_ratio": 99.0})
        assert out["holy_grail_cfg_early_ratio"] <= 1.4

    def test_sanitize_clamps_negative(self):
        from diffusion.sampling import sanitize_holy_grail_kwargs

        out = sanitize_holy_grail_kwargs({"holy_grail_cads_strength": -1.0})
        assert out["holy_grail_cads_strength"] == 0.0

    def test_sanitize_consistency_constraint(self):
        """cads_min_strength must not exceed cads_strength after sanitize."""
        from diffusion.sampling import sanitize_holy_grail_kwargs

        out = sanitize_holy_grail_kwargs(
            {
                "holy_grail_cads_strength": 0.05,
                "holy_grail_cads_min_strength": 0.1,  # violates constraint
            }
        )
        assert out["holy_grail_cads_min_strength"] <= out["holy_grail_cads_strength"]

    def test_sanitize_returns_copy(self):
        from diffusion.sampling import sanitize_holy_grail_kwargs

        original = {"holy_grail_cfg_early_ratio": 0.72}
        out = sanitize_holy_grail_kwargs(original)
        assert out is not original


# ---------------------------------------------------------------------------
# Condition annealing
# ---------------------------------------------------------------------------


class TestConditionAnnealing:
    def test_cads_noise_std_at_start(self):
        from diffusion.sampling import cads_noise_std

        std = cads_noise_std(progress=0.0, base_strength=0.1)
        assert abs(std - 0.1) < 1e-6

    def test_cads_noise_std_at_end(self):
        from diffusion.sampling import cads_noise_std

        std = cads_noise_std(progress=1.0, base_strength=0.1, min_strength=0.0)
        assert std == 0.0

    def test_apply_condition_noise_shape(self):
        from diffusion.sampling import apply_condition_noise

        cond = torch.randn(2, 77, 4096)
        noisy = apply_condition_noise(cond, std=0.05)
        assert noisy.shape == cond.shape

    def test_apply_condition_noise_zero_std_noop(self):
        from diffusion.sampling import apply_condition_noise

        cond = torch.randn(2, 77, 4096)
        out = apply_condition_noise(cond, std=0.0)
        torch.testing.assert_close(out, cond)


# ---------------------------------------------------------------------------
# Latent refiner
# ---------------------------------------------------------------------------


class TestLatentRefiner:
    def test_unsharp_mask_shape(self):
        from diffusion.sampling_utils import unsharp_mask_latent

        x = torch.randn(2, 4, 16, 16)
        out = unsharp_mask_latent(x, sigma=0.6, amount=0.2)
        assert out.shape == x.shape

    def test_unsharp_mask_zero_amount_noop(self):
        from diffusion.sampling_utils import unsharp_mask_latent

        x = torch.randn(2, 4, 16, 16)
        out = unsharp_mask_latent(x, sigma=0.6, amount=0.0)
        torch.testing.assert_close(out, x)

    def test_dynamic_threshold_clip_only_keeps_inliers(self):
        """``rescale=False`` (holy-grail final clamp) clips outliers but never shrinks the latent."""
        from diffusion.sampling_utils import dynamic_threshold

        x = torch.randn(2, 4, 16, 16) * 3
        out = dynamic_threshold(x, 0.99, 1.0, rescale=False)
        bound = torch.quantile(x.flatten(1).abs(), 0.99, dim=1).view(2, 1, 1, 1)
        inside = x.abs() <= bound
        torch.testing.assert_close(out[inside], x[inside])
        assert (out.abs() <= bound + 1e-6).all()


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


class TestRecommender:
    def test_anime_prompt(self):
        from diffusion.sampling import recommend_holy_grail_preset

        assert recommend_holy_grail_preset(prompt="anime girl, manga style") == "anime"

    def test_photoreal_prompt(self):
        from diffusion.sampling import recommend_holy_grail_preset

        assert recommend_holy_grail_preset(prompt="photoreal portrait, dslr") == "photoreal"

    def test_illustration_prompt(self):
        from diffusion.sampling import recommend_holy_grail_preset

        assert recommend_holy_grail_preset(prompt="concept art illustration") == "illustration"

    def test_default_balanced(self):
        from diffusion.sampling import recommend_holy_grail_preset

        result = recommend_holy_grail_preset(prompt="a dog in a field")
        assert result in ("balanced", "photoreal", "anime", "illustration", "aggressive")
