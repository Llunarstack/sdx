"""RAELatentBridge smoke tests."""

import torch
from models.rae_latent_bridge import RAELatentBridge


def test_rae_bridge_shapes_and_cycle():
    b = RAELatentBridge(rae_channels=16, dit_channels=4)
    z = torch.randn(2, 16, 8, 8)
    z4 = b.rae_to_dit(z)
    assert z4.shape == (2, 4, 8, 8)
    z_back = b.dit_to_rae(z4)
    assert z_back.shape == z.shape
    cyc = b.cycle_loss(z)
    assert cyc.ndim == 0 and cyc.item() >= 0
