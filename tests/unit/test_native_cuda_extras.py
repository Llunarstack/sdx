import numpy as np

from sdx_native.rope_apply_native import apply_rope_interleaved_numpy
from sdx_native.rmsnorm_native import rmsnorm_rows_numpy
from sdx_native.silu_gate_native import silu_gate_numpy


def test_rmsnorm_rows_numpy_unit_rms():
    x = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.5]], dtype=np.float32)
    y = rmsnorm_rows_numpy(x, eps=1e-6)
    rms = np.sqrt(np.mean(y * y, axis=1))
    assert np.allclose(rms, np.ones_like(rms), atol=1e-5)


def test_rope_apply_numpy_changes_values():
    q = np.arange(16, dtype=np.float32).reshape(2, 8)
    k = (np.arange(16, dtype=np.float32).reshape(2, 8) + 1.0).astype(np.float32)
    qo, ko = apply_rope_interleaved_numpy(q, k, theta_base=10000.0)
    assert qo.shape == q.shape
    assert ko.shape == k.shape
    assert not np.allclose(qo, q)
    assert not np.allclose(ko, k)


def test_silu_gate_numpy_matches_reference():
    x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=np.float32)
    g = np.array([1.0, 0.25, 2.0, -1.0, 0.5], dtype=np.float32)
    out = silu_gate_numpy(x, g)
    ref = (x * (1.0 / (1.0 + np.exp(-x))) * g).astype(np.float32)
    assert np.allclose(out, ref, atol=1e-6)
