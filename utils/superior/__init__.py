"""
SDX **Superior Stack**: composable quality and grounding modules on top of DiT + diffusion.

Retrieval, multi-metric ranking, self-correction, block caches, preference mining and DPO loops.
Import the submodule you need (``utils.superior.inference_pipeline`` ...); the package itself stays
empty so the sampler's cache imports do not pull in the training/orchestration modules.
"""
