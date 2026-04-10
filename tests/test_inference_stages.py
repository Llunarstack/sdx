from __future__ import annotations

import pytest
from utils.generation.inference_stages import (
    INFERENCE_PIPELINE_STAGES,
    inference_stage_index,
)


def test_inference_stages_unique_and_non_empty() -> None:
    assert len(INFERENCE_PIPELINE_STAGES) >= 4
    assert len(set(INFERENCE_PIPELINE_STAGES)) == len(INFERENCE_PIPELINE_STAGES)


def test_inference_stage_index() -> None:
    assert inference_stage_index("prompt") == 0
    assert inference_stage_index("image_output") == len(INFERENCE_PIPELINE_STAGES) - 1


def test_inference_stage_index_unknown() -> None:
    with pytest.raises(ValueError, match="unknown inference stage"):
        inference_stage_index("not_a_real_stage")
