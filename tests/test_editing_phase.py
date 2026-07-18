"""Tests for the post-generation editing phase orchestrator."""

from __future__ import annotations

import numpy as np


class TestPromptPieces:
    def test_extract_tokens(self):
        from utils.generation.editing_phase import extract_prompt_tokens

        toks = extract_prompt_tokens("a lone samurai at sunset with red armor")
        assert "samurai" in toks
        assert "the" not in toks

    def test_expected_text(self):
        from utils.generation.editing_phase import expected_render_text

        assert "OPEN" in expected_render_text('a shop sign that says "OPEN"')
        assert expected_render_text("no quotes here") == []

    def test_piece_labels(self):
        from utils.generation.editing_phase import infer_piece_labels

        labs = infer_piece_labels("portrait of a woman with detailed hands in a forest")
        assert "subject" in labs
        assert "face" in labs or "hands" in labs
        assert "background" in labs


class TestMissingTokens:
    def test_gap_detection(self):
        from utils.generation.editing_phase import missing_tokens_heuristic

        miss = missing_tokens_heuristic(
            "samurai red armor cherry blossoms",
            "a person standing outside",
        )
        assert "samurai" in miss or "armor" in miss


class TestPlanEdits:
    def test_ocr_and_anatomy_plan(self):
        from utils.generation.editing_phase import Diagnosis, EditingPhaseConfig, plan_edits

        d = Diagnosis(
            gate_passed=False,
            gate_failures=["clip"],
            clip_score=0.1,
            missing_tokens=["dragon"],
            expected_text=["CAFE"],
            needs_ocr=True,
            needs_anatomy=True,
            piece_labels=["subject", "face"],
        )
        acts = plan_edits(d, 'portrait with hands and sign "CAFE"', cfg=EditingPhaseConfig())
        kinds = {a.kind for a in acts}
        assert "ocr_fix" in kinds
        assert "inpaint_region" in kinds


class TestEditingPhaseDryRun:
    def test_loop_produces_pieces_and_report(self, tmp_path):
        from utils.generation.editing_phase import EditingPhaseConfig, run_editing_phase

        img = np.random.randint(40, 200, (128, 128, 3), dtype=np.uint8)
        cfg = EditingPhaseConfig(
            max_iters=2,
            min_clip=0.0,  # skip CLIP hub download in unit tests
            min_sharpness=0.99,  # force sharpness failure → actions
            dry_run=True,
            device="cpu",
            enable_art_post=True,
        )
        result = run_editing_phase(
            img,
            'a warrior with hands holding a sword, sign says "OPEN"',
            ckpt=None,
            config=cfg,
            work_dir=tmp_path,
            caption="a blurry person outdoors",
        )
        assert result.iterations >= 1
        assert result.piece_dir is not None
        assert (tmp_path / "pieces").is_dir()
        assert result.actions_applied or result.diagnosis_history
        assert result.stopped_reason in {"max_iters", "no_actions", "gates_passed"}


class TestOrchestrationRoles:
    def test_editor_role_present(self):
        from utils.generation.orchestration import pipeline_roles

        names = [r.name for r in pipeline_roles()]
        assert names == ["designer", "reasoner", "verifier", "editor"]
