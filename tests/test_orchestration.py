"""Tests for utils/orchestration.py (pipeline role registry)."""

from utils.orchestration import DESIGNER, VERIFIER, REASONER, pipeline_roles


def test_pipeline_roles_order_and_names():
    roles = pipeline_roles()
    assert len(roles) == 3
    assert roles[0] is DESIGNER
    assert roles[1] is VERIFIER
    assert roles[2] is REASONER
    assert {r.name for r in roles} == {"designer", "verifier", "reasoner"}
    for r in roles:
        assert r.description
        assert r.sdx_module_hint
