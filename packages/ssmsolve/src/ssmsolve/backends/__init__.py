"""Runtime backend detection for ssmsolve.

Detection order: rust (ssmsolve-rs) → jit (numba) → pyfar (scipy BLAS fallback).

The active backend name is exposed as :data:`ssmsolve.BACKEND`.
"""

from __future__ import annotations

__all__ = ["get_solver", "BACKEND"]


def _try_rust():
    from ssmsolve.backends.rust import solve  # noqa: PLC0415

    return solve, "rust"


def _try_numba():
    from ssmsolve.backends.numba import solve  # noqa: PLC0415

    return solve, "numba"


def get_solver():
    """Return ``(solve_fn, backend_name)`` for the best available backend.

    Falls back to ``(None, "pyfar")`` when neither extra is installed;
    :class:`~ssmsolve.models.StateSpaceModel` will then delegate to
    :meth:`pyfar.classes.filter.StateSpaceModel._process` (scipy BLAS).

    ``solve_fn`` signature (when not ``None``)::

        solve(y, x, A, B, C, D, u) -> None

    All arrays are modified in-place (``y`` and ``x``).
    """
    for loader in (_try_rust, _try_numba):
        try:
            return loader()
        except ImportError:
            continue
    return None, "pyfar"


_solver, BACKEND = get_solver()
