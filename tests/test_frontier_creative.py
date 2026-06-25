"""Tests for creative frontier — imagination, mutations, mood physics."""

from __future__ import annotations

from frontier.cinema import ShotGrammar
from frontier.constraint import CreativeConstraintEngine
from frontier.glitch import GlitchPlanner
from frontier.imagination import analyze_imagination, imagination_sample_kwargs
from frontier.mutation import PromptMutator
from frontier.paradox import ParadoxKeeper
from frontier.registry import idea_by_id
from frontier.surreal import DreamLogicPlanner
from frontier.synesthesia import SynesthesiaEngine
from frontier.vibe import MoodPhysics


class TestCreativeModules:
    def test_dream_boosts_serendipity(self):
        _, _, boost = DreamLogicPlanner().fragments("surreal melting dreamscape")
        assert boost > 0

    def test_paradox_suppresses_resolve(self):
        _, _, suppress = ParadoxKeeper().fragments("Escher impossible stairs")
        assert suppress

    def test_mutator_changes_prompt(self):
        muts = PromptMutator().mutate_batch("sunset landscape", seed=1, count=3)
        assert len(muts) >= 2
        assert any(m.mutated != m.original for m in muts)

    def test_random_constraint(self):
        pack = CreativeConstraintEngine().suggest_random(seed=3)
        assert pack.rules

    def test_synesthesia_knobs(self):
        ser, cfg = SynesthesiaEngine().diffusion_knobs("jazz club smoky portrait")
        assert ser > 0

    def test_shot_grammar_ots(self):
        pos, _ = ShotGrammar().fragments("over the shoulder dialogue scene")
        assert "shoulder" in pos.lower()

    def test_mood_physics_curve(self):
        plan = MoodPhysics().analyze("ominous dread forest")
        curve = MoodPhysics().step_emphasis_curve(plan, 10)
        assert len(curve) == 10

    def test_glitch_vhs(self):
        _, _, boost = GlitchPlanner().fragments("VHS tracking error aesthetic")
        assert boost > 0


class TestImaginationEngine:
    def test_full_plan(self):
        plan = analyze_imagination(
            "surreal Escher city, jazz mood, over the shoulder, VHS glitch, rusted gate, steampunk cyberpunk crowd",
            mutate_count=2,
            mutate_seed=42,
        )
        assert plan.creative_trace
        assert plan.serendipity_dial > 0.2
        kw = imagination_sample_kwargs(plan)
        assert kw["frontier_creative"]
        assert plan.mutations

    def test_registry(self):
        assert idea_by_id("imagination_engine").status == "implemented"


class TestNewCreativeModules:
    def test_fusion(self):
        from frontier.fusion import GenreCollisionEngine

        pos, _ = GenreCollisionEngine().fragments("steampunk city with cyberpunk neon")
        assert "steampunk" in pos.lower() or "brass" in pos.lower()

    def test_crowd(self):
        from frontier.collective import CrowdGrammar

        pos, neg = CrowdGrammar().fragments("concert crowd at stadium")
        assert "duplicate" in neg.lower()

    def test_creative_refine_mutations(self):

        # dry-run: only test mutation list building via analyze_imagination
        from frontier.imagination import analyze_imagination

        plan = analyze_imagination("sunset dragon", mutate_count=3, mutate_seed=1)
        assert len(plan.mutations) >= 2


class TestPromptReplace:
    def test_replace_prompt_in_cmd(self):
        from utils.generation.sample_features import _replace_or_append_prompt

        cmd = ["python", "sample.py", "--prompt", "old", "--ckpt", "x.pt"]
        out = _replace_or_append_prompt(cmd, "new prompt")
        assert "--prompt" in out
        assert "new prompt" in out
        assert "old" not in out
