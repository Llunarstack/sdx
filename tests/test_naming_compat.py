from diffusion.losses.timestep_loss_weight import get_timestep_loss_weight as get_timestep_loss_weight_canonical
from models.dit_text_variants import DiT_P_2_Text
from models.superior_vit import SuperiorViT
from utils.architecture.ar_block_conditioning import normalize_num_ar_blocks as normalize_num_ar_blocks_canonical
from vit_quality.checkpoint_utils import load_vit_quality_checkpoint


def test_diffusion_holy_grail_shim_reexports_sampling_extras():
    from diffusion.holy_grail import list_holy_grail_presets  # noqa: PLC0415
    from diffusion.sampling_extras import list_holy_grail_presets as list_se  # noqa: PLC0415

    assert list_holy_grail_presets() == list_se()


def test_ar_block_conditioning_canonical_available():
    assert normalize_num_ar_blocks_canonical(2) == 2
    assert normalize_num_ar_blocks_canonical(7) == -1


def test_diffusion_timestep_loss_weight_canonical():
    assert callable(get_timestep_loss_weight_canonical)


def test_model_canonical_aliases_expose_symbols():
    assert callable(DiT_P_2_Text)
    assert callable(SuperiorViT)


def test_vit_quality_canonical_package_exports_checkpoint_loader():
    assert callable(load_vit_quality_checkpoint)


def test_config_defaults_prompt_style_mediums_surface():
    """Smoke: canonical config modules used by sample/prompt tools stay importable."""
    from config.defaults import (  # noqa: PLC0415
        art_mediums,
        model_presets,
        prompt_domains,
        style_guidance,
    )

    assert isinstance(getattr(prompt_domains, "WATERMARK_NEGATIVE_STRONG", None), str)
    assert callable(getattr(art_mediums, "guidance_fragments", None))
    assert callable(getattr(style_guidance, "style_guidance_fragments", None))
    assert callable(getattr(model_presets, "apply_preset_to_args", None))

