"""Resolve the diffusers autoencoder class for a config ``autoencoder_type``.

``AutoencoderRAE`` only exists in diffusers>=0.37.0; importing it unconditionally
alongside ``AutoencoderKL`` breaks the default KL path on older installs, so the
RAE import is deferred until "rae" is actually requested.
"""

from __future__ import annotations


def get_autoencoder_class(ae_type: str):
    """Return ``AutoencoderKL`` for "kl" (default) or ``AutoencoderRAE`` for "rae"."""
    if str(ae_type or "kl").strip().lower() == "rae":
        try:
            from diffusers import AutoencoderRAE
        except ImportError as e:
            raise ImportError(
                "autoencoder_type='rae' requires diffusers>=0.37.0 (AutoencoderRAE). "
                "Upgrade with: pip install -U 'diffusers>=0.37.0'"
            ) from e
        return AutoencoderRAE
    from diffusers import AutoencoderKL

    return AutoencoderKL
