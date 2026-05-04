"""Numba JIT backend (jit extra)."""

from __future__ import annotations

from numba import float32, float64, jit


@jit(
    [
        (float32[::1, :], float32[::1], float32[::1, :], float32[::1, :], float32[::1, :], float32[::1, :], float32[::1, :]),
        (float64[::1, :], float64[::1], float64[::1, :], float64[::1, :], float64[::1, :], float64[::1, :], float64[::1, :]),
    ],
    nopython=True,
    cache=True,
)
def solve_F(y, x, A, B, C, D, u):
    """JIT solver for Fortran-order (column-major) arrays."""
    for i in range(y.shape[1]):
        y[:, i] = C @ x + D @ u[:, i]
        x = A @ x + B @ u[:, i]


@jit(
    [
        (float32[:, ::1], float32[::1], float32[:, ::1], float32[:, ::1], float32[:, ::1], float32[:, ::1], float32[:, ::1]),
        (float64[:, ::1], float64[::1], float64[:, ::1], float64[:, ::1], float64[:, ::1], float64[:, ::1], float64[:, ::1]),
    ],
    nopython=True,
    cache=True,
)
def solve_C(y, x, A, B, C, D, u):
    """JIT solver for C-order (row-major) arrays."""
    for i in range(y.shape[1]):
        y[:, i] = C @ x + D @ u[:, i]
        x = A @ x + B @ u[:, i]


def solve(y, x, A, B, C, D, u):
    """Dispatch to solve_F or solve_C based on A's memory layout."""
    if A.flags["F_CONTIGUOUS"]:
        solve_F(y, x, A, B, C, D, u)
    else:
        solve_C(y, x, A, B, C, D, u)
