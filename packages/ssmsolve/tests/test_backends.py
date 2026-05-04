"""Tests for ssmsolve backend process functions.

Each backend (pyfar, numba, rust) is tested independently by patching
_backend_solve_F/_backend_solve_C in ssmsolve.models. Both storage orders
('F' / 'C') and both precisions (float32 / float64) are covered.
"""

import numpy as np
import pytest
import ssmsolve.models as _m
from pyfar import Signal
from ssmsolve.models import StateSpaceModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(0)


def _make_system(n=8, m=3, p=4, T=64, dtype=np.float64, storage="F"):
    A = 0.8 * np.eye(n, dtype=dtype)
    B = _rng.random((n, m)).astype(dtype)
    C = _rng.random((p, n)).astype(dtype)
    sys = StateSpaceModel(A, B, C, sampling_rate=1, dtype=dtype, storage=storage)
    sig = Signal(_rng.random((m, T)).astype(dtype), sampling_rate=1)
    return sys, sig


def _pyfar_reference(sys, sig):
    """Reference output using the pyfar scipy-BLAS backend."""
    orig = _m._backend_solve
    try:
        _m._backend_solve = None
        sys.init_state()
        return sys.process(sig).time.copy()
    finally:
        _m._backend_solve = orig


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------

DTYPES = [np.float32, np.float64]
STORAGES = ["F", "C"]

DTYPE_IDS = ["float32", "float64"]
STORAGE_IDS = ["fortran", "c-order"]


# ---------------------------------------------------------------------------
# Pyfar (scipy BLAS) backend
# ---------------------------------------------------------------------------


class TestPyfarBackend:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("storage", STORAGES, ids=STORAGE_IDS)
    def test_output_shape(self, dtype, storage):
        sys, sig = _make_system(dtype=dtype, storage=storage)
        _m._backend_solve = None
        sys.init_state()
        out = sys.process(sig)
        assert out.time.shape == (4, 64)

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("storage", STORAGES, ids=STORAGE_IDS)
    def test_zero_input_zero_output(self, dtype, storage):
        sys, _ = _make_system(dtype=dtype, storage=storage)
        _m._backend_solve = None
        zero_sig = Signal(np.zeros((3, 64), dtype=dtype), sampling_rate=1)
        sys.init_state()
        out = sys.process(zero_sig)
        np.testing.assert_allclose(out.time, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Numba backend
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def numba_backend():
    pytest.importorskip("numba", reason="numba not installed")
    from ssmsolve.backends.numba import solve

    orig = _m._backend_solve
    _m._backend_solve = solve
    yield
    _m._backend_solve = orig


class TestNumbaBackend:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("storage", STORAGES, ids=STORAGE_IDS)
    def test_matches_pyfar(self, numba_backend, dtype, storage):
        sys, sig = _make_system(dtype=dtype, storage=storage)
        ref = _pyfar_reference(sys, sig)
        from ssmsolve.backends.numba import solve

        _m._backend_solve = solve
        sys.init_state()
        out = sys.process(sig).time
        atol = 1e-4 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(out, ref, rtol=0, atol=atol)

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("storage", STORAGES, ids=STORAGE_IDS)
    def test_zero_input_zero_output(self, numba_backend, dtype, storage):
        sys, _ = _make_system(dtype=dtype, storage=storage)
        zero_sig = Signal(np.zeros((3, 64), dtype=dtype), sampling_rate=1)
        sys.init_state()
        out = sys.process(zero_sig)
        np.testing.assert_allclose(out.time, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Rust backend
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def rust_backend():
    pytest.importorskip("ssmsolve_rs", reason="ssmsolve-rs not installed")
    from ssmsolve.backends.rust import solve

    orig = _m._backend_solve
    _m._backend_solve = solve
    yield
    _m._backend_solve = orig


class TestRustBackend:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("storage", STORAGES, ids=STORAGE_IDS)
    def test_matches_pyfar(self, rust_backend, dtype, storage):
        sys, sig = _make_system(dtype=dtype, storage=storage)
        ref = _pyfar_reference(sys, sig)
        from ssmsolve.backends.rust import solve

        _m._backend_solve = solve
        sys.init_state()
        out = sys.process(sig).time
        atol = 1e-4 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(out, ref, rtol=0, atol=atol)

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("storage", STORAGES, ids=STORAGE_IDS)
    def test_zero_input_zero_output(self, rust_backend, dtype, storage):
        sys, _ = _make_system(dtype=dtype, storage=storage)
        zero_sig = Signal(np.zeros((3, 64), dtype=dtype), sampling_rate=1)
        sys.init_state()
        out = sys.process(zero_sig)
        np.testing.assert_allclose(out.time, 0.0, atol=1e-6)
