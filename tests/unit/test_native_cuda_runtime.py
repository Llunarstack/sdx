from __future__ import annotations

import numpy as np

from sdx_native.rope_apply_native import apply_rope_interleaved_numpy, maybe_apply_rope_interleaved_cuda
from sdx_native.rmsnorm_native import maybe_rmsnorm_rows_cuda, rmsnorm_rows_numpy
from sdx_native.silu_gate_native import maybe_silu_gate_cuda, silu_gate_numpy


def test_maybe_rmsnorm_cuda_matches_numpy_when_available() -> None:
    x = np.random.randn(6, 16).astype(np.float32)
    y_ref = rmsnorm_rows_numpy(x)
    y_cuda = maybe_rmsnorm_rows_cuda(x)
    if y_cuda is None:
        return
    assert np.allclose(y_cuda, y_ref, atol=1e-4, rtol=1e-4)


def test_maybe_rope_cuda_matches_numpy_when_available() -> None:
    q = np.random.randn(8, 32).astype(np.float32)
    k = np.random.randn(8, 32).astype(np.float32)
    q_ref, k_ref = apply_rope_interleaved_numpy(q, k, theta_base=10000.0)
    out = maybe_apply_rope_interleaved_cuda(q, k, theta_base=10000.0)
    if out is None:
        return
    q_cuda, k_cuda = out
    assert np.allclose(q_cuda, q_ref, atol=1e-4, rtol=1e-4)
    assert np.allclose(k_cuda, k_ref, atol=1e-4, rtol=1e-4)


def test_maybe_silu_gate_cuda_matches_numpy_when_available() -> None:
    x = np.random.randn(128).astype(np.float32)
    g = np.random.randn(128).astype(np.float32)
    y_ref = silu_gate_numpy(x, g)
    y_cuda = maybe_silu_gate_cuda(x, g)
    if y_cuda is None:
        return
    assert np.allclose(y_cuda, y_ref, atol=1e-5, rtol=1e-5)
