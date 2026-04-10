"""Tests for ``utils.training.error_handling`` helpers (beyond checkpoint validation)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from utils.training.error_handling import get_model_info, retry_on_cuda_oom, safe_execute


def test_safe_execute_reraises_and_logs_with_logger(caplog: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("test_safe_execute")
    caplog.set_level(logging.ERROR, logger="test_safe_execute")

    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        safe_execute(boom, logger=log)

    assert any("boom" in r.getMessage() and "nope" in r.getMessage() for r in caplog.records)


def test_safe_execute_callable_without_function_name_uses_type_name(caplog: pytest.LogCaptureFixture) -> None:
    class Caller:
        def __call__(self) -> None:
            raise RuntimeError("x")

    log = logging.getLogger("test_safe_execute2")
    caplog.set_level(logging.ERROR, logger="test_safe_execute2")

    with pytest.raises(RuntimeError, match="x"):
        safe_execute(Caller(), logger=log)

    assert any("Caller" in r.getMessage() for r in caplog.records)


def test_get_model_info_linear() -> None:
    m = torch.nn.Linear(4, 2, bias=True)
    info = get_model_info(m)
    assert info["total_parameters"] == 10  # 4*2 + 2
    assert info["trainable_parameters"] == 10
    assert "model_size_mb" in info
    assert info["device"] == torch.device("cpu")


def test_retry_on_cuda_oom_reduces_batch_and_retries() -> None:
    calls: list[int] = []

    @retry_on_cuda_oom(max_retries=3, reduce_batch_size=True)
    def train_step(*, batch_size: int) -> str:
        calls.append(batch_size)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    with patch.object(torch.cuda, "empty_cache", MagicMock()):
        out = train_step(batch_size=8)

    assert out == "ok"
    assert calls == [8, 4]


def test_retry_on_cuda_oom_non_oom_reraises_immediately() -> None:
    @retry_on_cuda_oom(max_retries=3)
    def f() -> None:
        raise RuntimeError("something else")

    with pytest.raises(RuntimeError, match="something else"):
        f()
