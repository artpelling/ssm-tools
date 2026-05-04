# `ssmsolve`

Python package for time-domain simulation of discrete-time state-space models, backed by
[`ssmsolve-rs`](https://github.com/artpelling/ssm-tools/tree/main/packages/ssmsolve-rs) (BLAS-accelerated Rust extension).

## Installation

```toml
dependencies = [
    "ssmsolve @ git+https://github.com/artpelling/ssm-tools#subdirectory=packages/ssmsolve"
]
```

## Quick start

```python
import numpy as np
from pyfar import Signal
from ssmsolve.models import StateSpaceModel

# Build a system (n states, m inputs, p outputs)
A, B, C = np.eye(100) * 0.9, np.random.randn(100, 2), np.random.randn(4, 100)
sys = StateSpaceModel(A, B, C, sampling_rate=44100, dtype=np.float32)
sys.init_state()

# Process a signal — returns a pyfar.Signal
sig = Signal(np.random.randn(2, 4096), sampling_rate=44100)
out = sys.process(sig)
```

## Solver backends

| Class | Backend | dtypes | Status |
|-------|---------|--------|--------|
| `pyfar.StateSpaceModel` | BLAS `gemv` via NumPy | float32, float64 | baseline |
| `StateSpaceModel` | CBLAS `gemv` via `ssmsolve-rs` | float32, float64 | available |
| `TriangularStateSpaceModel` | — | — | planned |
| `DiagonalStateSpaceModel` | — | — | planned |

All classes accept a `storage` parameter (`'F'` column-major or `'C'` row-major). The system
state `x` is updated in place across calls, enabling sequential chunk processing.
