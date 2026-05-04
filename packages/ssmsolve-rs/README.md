# `ssmsolve-rs`

High-performance BLAS-backed state-space solvers — the `[rust]` extra for
[`ssmsolve`](https://github.com/artpelling/ssm-tools/tree/main/packages/ssmsolve).

Implements the discrete-time state-space recursion:

```
y[:, i] = C @ x + D @ u[:, i]
x       = A @ x + B @ u[:, i]
```

using `cblas_sgemv` / `cblas_dgemv` for both `float32` and `float64` arrays.

## Installation

```sh
pip install "ssmsolve[rust]"
```

When installed, `ssmsolve` will automatically use this backend (`ssmsolve.BACKEND == "rust"`).
Requires a BLAS library at build time. See [BLAS.md](BLAS.md) for platform-specific instructions.

## Standalone usage

`ssmsolve-rs` can also be used directly without `ssmsolve`:

```sh
pip install ssmsolve-rs
```

```python
from ssmsolve_rs import solve_f32, solve_f64
```

## Building from source

```sh
pip install maturin
maturin develop --release
```

