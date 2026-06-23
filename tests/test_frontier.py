"""Tests for frontier outside-the-box generation modules."""

from __future__ import annotations

import torch

from frontier import FrontierEngine, analyze_prompt
from frontier.logic import AbsenceExtractor, ContradictionScanner
from frontier.chaos import SerendipityInjector, EntropyBudgetAllocator
from frontier.memory import GenerationEchoMemory
from frontier.narrative import TemporalMomentAnalyzer, WitnessPerspectiveAnalyzer
from frontier.narrative.witness import WitnessPerspective


class TestContradictionScanner:
    def test_detects_time_conflict(self):
        hits = ContradictionScanner().scan("busy plaza at noon with a dramatic sunset")
        assert hits
        assert hits[0].category == "time"

    def test_suggest_rewrite_drops_side(self):
        out = ContradictionScanner().suggest_rewrite("empty street with a huge crowd", pick="left")
        assert "crowd" not in out.lower()


class TestAbsenceExtractor:
    def test_no_people(self):
        constraints = AbsenceExtractor().extract("wide shot, no people, clear sky")
        subjects = {c.subject for c in constraints}
        assert "people" in subjects

    def test_merge_negative(self):
        ext = AbsenceExtractor()
        c = ext.extract("no text on signs")
        neg = ext.merge_negative_prompt("blurry", c)
        assert "text" in neg.lower()


class TestWitness:
    def test_cctv_guard(self):
        frame = WitnessPerspectiveAnalyzer().analyze("security camera footage of a hallway")
        assert frame.perspective == WitnessPerspective.GUARD
        assert frame.confidence > 0


class TestMoment:
    def test_anticipation_curve(self):
        cue = TemporalMomentAnalyzer(num_steps=10).analyze("statue about to fall")
        assert cue.phase.value == "anticipation"
        assert len(cue.step_emphasis) == 10
        assert cue.step_emphasis[0] > cue.step_emphasis[-1]


class TestChaos:
    def test_serendipity_peak_mid_schedule(self):
        curve = SerendipityInjector(num_steps=20).curve(dial=0.5)
        assert curve.peak_step > 2
        assert curve.peak_step < 18

    def test_entropy_budget_sums(self):
        b = EntropyBudgetAllocator(num_steps=12).allocate(total=1.0)
        assert abs(sum(b.per_step) - 1.0) < 1e-5

    def test_serendipity_noise_scale(self):
        inj = SerendipityInjector(num_steps=8)
        curve = inj.curve(0.4)
        n = torch.ones(2, 4)
        out = inj.apply_to_noise(n, curve.peak_step, curve)
        assert out.mean() > 1.0


class TestGenerationEcho:
    def test_remembers_failure_tags(self):
        mem = GenerationEchoMemory(capacity=8)
        mem.record_failure("red dragon on cliff", ["extra claws", "melted wings"])
        suffix = mem.negative_suffix("red dragon on cliff at dawn")
        assert "extra claws" in suffix


class TestFrontierEngine:
    def test_full_plan(self):
        plan = analyze_prompt("empty plaza at noon, sunset sky, CCTV view, about to rain")
        assert plan.contradictions
        assert plan.witness is not None
        assert plan.augmented_prompt

    def test_sample_kwargs(self):
        eng = FrontierEngine(serendipity_dial=0.2)
        plan = eng.analyze("candid street photo, no cars")
        kw = eng.sample_kwargs(plan, base_negative="ugly")
        assert "prompt" in kw
        assert kw["frontier_serendipity_scales"]
        assert "cars" in kw["negative_prompt"].lower() or "ugly" in kw["negative_prompt"].lower()
