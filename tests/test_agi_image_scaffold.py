"""Smoke tests for research.agi_image (torch-free)."""

from __future__ import annotations

import pytest


def test_import_research_agi_surface():
    from research.agi_image import GoalSpec

    assert GoalSpec(title="t", narrative="dog").narrative == "dog"


@pytest.mark.parametrize(
    "path",
    [
        "research.agi_image.intents",
        "research.agi_image.planning",
        "research.agi_image.world",
        "research.agi_image.evaluation",
        "research.agi_image.reasoning",
        "research.agi_image.alignment",
        "research.agi_image.schemas",
        "research.agi_image.tooling",
        "research.agi_image.benchmarks",
        "research.agi_image.integrations",
        "research.agi_image.memory",
        "research.agi_image.reflection",
    ],
)
def test_subpackages_import(path: str):
    __import__(path)


def test_generation_plan_manifest():
    from research.agi_image.planning import GenerationPlan, GenerationStep, GenerationStepKind

    p = GenerationPlan(goal_id="g1")
    p.add(
        GenerationStep(
            kind=GenerationStepKind.verify_visual,
            description="CLIP alignment gate",
            deps=[],
            kwargs={"metric": "clip"},
        )
    )
    manifest = p.to_manifest_dict()
    assert manifest["goal_id"] == "g1"
    assert manifest["steps"][0]["kind"] == "verify_visual"


def test_tool_registry_invokes_registered():
    from research.agi_image.tooling import ToolCall, ToolStubRegistry

    reg = ToolStubRegistry()

    reg.register(
        "noop",
        lambda _a: {"echo": True},
    )
    out = reg.invoke(ToolCall(name="noop", args={}))
    assert out.ok and out.payload.get("echo") is True


def test_sample_hints_from_goal():
    from research.agi_image.integrations.sample_hints import goal_spec_to_sample_hints
    from research.agi_image.intents import GoalSpec

    g = GoalSpec(title="Brand", narrative="minimal wordmark for a bakery")
    h = goal_spec_to_sample_hints(g)
    assert "visual-design-domain" in h["extras"]
