"""Rust/CBLAS backend (ssmsolve-rs extra)."""

from __future__ import annotations

import numpy as np
from ssmsolve_rs import solve_f32, solve_f64


def solve(y, x, A, B, C, D, u):
    """Dispatch to f32 or f64 CBLAS solver based on array dtype."""
    if A.dtype == np.dtype(np.float32):
        solve_f32(y, x, A, B, C, D, u)
    else:
        solve_f64(y, x, A, B, C, D, u)
