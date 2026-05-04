# ssmsolve-rs

High-performance BLAS-backed state-space solvers — the Rust backend for [`ssmsolve`](https://github.com/artpelling/ssm-tools/tree/main/packages/ssmsolve).

Implements the discrete-time state-space recursion:

```
y[:, i] = C @ x + D @ u[:, i]
x       = A @ x + B @ u[:, i]
```

using `cblas_sgemv` / `cblas_dgemv` for both `float32` and `float64` arrays.

## Installation

```sh
pip install ssmsolve-rs
```

Requires a BLAS library at build time. See [BLAS.md](BLAS.md) for platform-specific instructions.

## Usage

`ssmsolve-rs` is used automatically by `ssmsolve` when installed. You can also call the solvers directly:

```python
from ssmsolve_rs import solve_f32, solve_f64
```

## Building from source

```sh
pip install maturin
maturin develop --release
```
