"""Frontier: generation ideas outside the usual prompt-latent-decode box (concept map in ``frontier/README.md``)."""

from .engine import FrontierEngine, FrontierPlan, analyze_prompt

__all__ = ["FrontierEngine", "FrontierPlan", "analyze_prompt"]
