"""Competitor-weakness evaluation harness for sdx.

Grades generated images against a fixed prompt suite (prompt_suite.py) using
graceful scorers (scorers.py), so progress on the failure modes that leading
text-to-image models share (see docs/COMPETITIVE_ANALYSIS.md) is measured, not
guessed. Model-agnostic: score sdx output or a competitor's, the same way.
"""

from scripts.tools.eval.prompt_suite import SUITE, SUITE_VERSION, SuiteItem

__all__ = ["SUITE", "SUITE_VERSION", "SuiteItem"]
