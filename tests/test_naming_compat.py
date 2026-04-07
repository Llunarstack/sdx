from diffusion.losses.timestep_loss_weight import get_timestep_loss_weight as get_timestep_loss_weight_canonical
from diffusion.timestep_loss_weight import get_timestep_loss_weight as get_timestep_loss_weight_legacy
from models.dit_text_variants import DiT_P_2_Text
from models.superior_vit import SuperiorViT
from utils.architecture.ar_block_conditioning import normalize_num_ar_blocks as normalize_num_ar_blocks_canonical
from vit_quality.checkpoint_utils import load_vit_quality_checkpoint


def test_ar_block_conditioning_canonical_available():
    assert normalize_num_ar_blocks_canonical(2) == 2
    assert normalize_num_ar_blocks_canonical(7) == -1


def test_diffusion_timestep_weight_legacy_and_canonical_available():
    assert callable(get_timestep_loss_weight_canonical)
    assert callable(get_timestep_loss_weight_legacy)


def test_model_canonical_aliases_expose_symbols():
    assert callable(DiT_P_2_Text)
    assert callable(SuperiorViT)


def test_vit_quality_canonical_package_exports_checkpoint_loader():
    assert callable(load_vit_quality_checkpoint)

