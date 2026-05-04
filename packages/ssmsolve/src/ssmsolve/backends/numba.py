"""Numba JIT backend (jit extra)."""

from __future__ import annotations

from numba import jit


@jit(nopython=True, cache=True)
def _solve_inner(y, x, A, B, C, D, u):
    for i in range(y.shape[1]):
        y[:, i] = C @ x + D @ u[:, i]
        x = A @ x + B @ u[:, i]


def solve(y, x, A, B, C, D, u):
    """JIT-compiled state-space solver (numba)."""
    _solve_inner(y, x, A, B, C, D, u)
