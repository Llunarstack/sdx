"""
Merged registry: core + extended design domains.

Tier keys per domain: lite, standard, strong.
"""

from __future__ import annotations

from typing import Dict

from .registry_core import DOMAIN_NEGATIVES_CORE, DOMAIN_POSITIVES_CORE
from .registry_extra import DOMAIN_NEGATIVES_EXTRA, DOMAIN_POSITIVES_EXTRA

DOMAIN_POSITIVES: Dict[str, Dict[str, str]] = {**DOMAIN_POSITIVES_CORE, **DOMAIN_POSITIVES_EXTRA}
DOMAIN_NEGATIVES: Dict[str, Dict[str, str]] = {**DOMAIN_NEGATIVES_CORE, **DOMAIN_NEGATIVES_EXTRA}

__all__ = ["DOMAIN_POSITIVES", "DOMAIN_NEGATIVES"]
