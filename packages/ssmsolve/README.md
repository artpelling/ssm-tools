# `ssmsolve`

Python package for time-domain simulation of discrete-time state-space models.

The solver backend is selected automatically at import time from whichever optional extra
is installed — no code changes needed when switching backends.

## Installation

```toml
# pyproject.toml
dependencies = [
    "ssmsolve @ git+https://github.com/artpelling/ssm-tools#subdirectory=packages/ssmsolve"
]
```

Install an optional extra for a faster solver:

```sh
pip install "ssmsolve[rust]"   # CBLAS via Rust (ssmsolve-rs)
pip install "ssmsolve[jit]"    # Numba JIT
pip install "ssmsolve[full]"   # all extras
```

Without an extra the pyfar BLAS solver (`scipy.linalg gemv`) is used as a fallback.

## Quick start

```python
import numpy as np
import ssmsolve
from pyfar import Signal
from ssmsolve.models import StateSpaceModel

print(ssmsolve.BACKEND)   # "rust" | "numba" | "pyfar"

# Build a system (n states, m inputs, p outputs)
A, B, C = np.eye(100) * 0.9, np.random.randn(100, 2), np.random.randn(4, 100)
sys = StateSpaceModel(A, B, C, sampling_rate=44100, dtype=np.float32)
sys.init_state()

# Process a signal — returns a pyfar.Signal
sig = Signal(np.random.randn(2, 4096), sampling_rate=44100)
out = sys.process(sig)
```

## Solver backends

| Backend | Extra | Solver | dtypes |
|---------|-------|--------|--------|
| `"rust"` | `ssmsolve[rust]` | Fortran BLAS `gemv` (F-order) / CBLAS `gemv` (C-order) via `ssmsolve-rs` | float32, float64 |
| `"numba"` | `ssmsolve[jit]` | Numba `@jit(nopython=True)` | float32, float64 |
| `"pyfar"` | *(fallback)* | `scipy.linalg gemv` (BLAS) | float32, float64 |

Detection order: `rust` → `numba` → `pyfar`. The active backend is exposed as
`ssmsolve.BACKEND` and can be checked at runtime.

All classes accept a `storage` parameter (`'F'` column-major or `'C'` row-major). The system
state `x` is updated in place across calls, enabling sequential chunk processing.

## Classes

| Class | Status |
|-------|--------|
| `StateSpaceModel` | available |
| `TriangularStateSpaceModel` | planned |
| `DiagonalStateSpaceModel` | planned |

