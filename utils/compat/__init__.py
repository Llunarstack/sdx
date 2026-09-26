"""
Universal ecosystem **compat** — bridge foreign image-gen assets into sdx.

Pipeline: :func:`sniff_asset` identifies any file (checkpoint / adapter /
embedding / VAE / text encoder / upscaler) across every popular CivitAI
image-gen ecosystem — SD1.5, SD2, SDXL and its derivatives (Pony, Illustrious,
NoobAI, Playground, Kolors), SD3/3.5, Flux (incl. Chroma), AuraFlow, PixArt,
Hunyuan-DiT, Lumina, HiDream, Qwen-Image, Stable Cascade, plus generic DiTs —
then the matching bridge converts it:

- adapters (LoRA/DoRA/LoCon/LoHa/LoKr) → ``adapter_bridge.bridge_apply``
- textual inversions → ``embedding_bridge.load_textual_inversion``
- VAEs → ``latent_probe.probe_vae`` (+ ``LatentBridge`` when shapes differ)
- upscalers/face restore → already served by ``utils.modeling.hf_upscale``

Cross-architecture transfer is principled-approximate: deltas keep their role
and depth, not their exact effect. Every apply returns a coverage report.
"""

from __future__ import annotations

from utils.compat.adapter_bridge import BridgeReport, bridge_apply
from utils.compat.asset_sniffer import AssetReport, sniff_asset
from utils.compat.embedding_bridge import EmbeddingAsset, load_textual_inversion, resize_vectors
from utils.compat.latent_probe import LatentBridge, VAEProbe, probe_vae

__all__ = [
    "AssetReport",
    "BridgeReport",
    "EmbeddingAsset",
    "LatentBridge",
    "VAEProbe",
    "bridge_apply",
    "load_textual_inversion",
    "probe_vae",
    "resize_vectors",
    "sniff_asset",
]
