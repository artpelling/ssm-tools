# `ssm-tools`: State-space System Modelling Tools

A collection of interoperable Python packages for working with state-space models in acoustics.

```mermaid
graph LR
    irdl["[irdl](https://github.com/artpelling/irdl)\nIR datasets"]
    across["[across](packages/across/README.md)\nModel reduction · ERA"]
    ssm["[ssm_tools](pyproject.toml)\nTime-domain solvers"]

    pyfar([pyfar])
    pymor([pyMOR])

    irdl -->|pyfar.Signal IR| across
    across -->|A, B, C, D| ssm
    pymor -->|ERAReductor| across
    pyfar -->|StateSpaceModel| ssm
```

## Packages

| Package | Description |
|---------|-------------|
| [`ssm_tools`](pyproject.toml) | Fast time-domain solvers for state-space models — the main package in this repo |
| [`across`](packages/across/README.md) | Reduced-order state-space models from impulse response data via ERA |
| [`irdl`](https://github.com/artpelling/irdl) | Downloads and processes impulse response datasets |

---

## `ssm_tools`

[`pyfar`](https://pyfar.org)-compatible state-space model classes with interchangeable solver backends. Latest benchmarks: [artpelling.github.io/ssm-tools-asv](https://artpelling.github.io/ssm-tools-asv).

The discrete-time recursion solved by all backends is:

```
y[:, i] = C @ x + D @ u[:, i]
x       = A @ x + B @ u[:, i]
```

### Solver backends

| Class | Backend | dtypes | Status |
|-------|---------|--------|--------|
| `pyfar.StateSpaceModel` | NumPy | float32, float64 | baseline |
| `StateSpaceModel` | Rust + BLAS (CBLAS `gemv`) | float32, float64 | available |
| `TriangularStateSpaceModel` | — | — | planned |
| `DiagonalStateSpaceModel` | — | — | planned |

All classes accept a `storage` parameter (`'F'` column-major or `'C'` row-major) that controls the memory layout of the system matrices. The system state `x` is updated in place across calls, so sequential chunk processing preserves state.

### Installation

Add to your `pyproject.toml`:

```toml
dependencies = [
    "ssm_tools @ git+https://github.com/artpelling/ssm-tools"
]
```

or install directly:

```sh
pip install git+https://github.com/artpelling/ssm-tools
```

> **Note:** `ssm_tools` contains a Rust extension that requires an LP64 CBLAS library at build time. On most systems this is resolved automatically. See [BLAS.md](BLAS.md) for details.

### Quick start

```python
import numpy as np
from pyfar import Signal
from ssm_tools.models import StateSpaceModel

# Build a system (n states, m inputs, p outputs)
A, B, C = np.eye(100) * 0.9, np.random.randn(100, 2), np.random.randn(4, 100)
sys = StateSpaceModel(A, B, C, sampling_rate=44100, dtype=np.float32)
sys.init_state()

# Process a signal — returns a pyfar.Signal
sig = Signal(np.random.randn(2, 4096), sampling_rate=44100)
out = sys.process(sig)
```

### Development setup

The Rust extension is built with [maturin](https://github.com/PyO3/maturin), managed via [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/artpelling/ssm-tools
cd ssm-tools
uv sync
uv run maturin develop --release
```
