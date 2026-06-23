"""
Frontier: generation ideas outside the usual prompt–latent–decode box.

See ``frontier/README.md`` for the concept map.
"""

__version__ = "11.0.0"

from .engine import FrontierEngine, FrontierPlan, analyze_prompt

__all__ = ["FrontierEngine", "FrontierPlan", "analyze_prompt"]
