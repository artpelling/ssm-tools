# `ssm-tools`: State-space System Modelling Tools
A collection of interoperable Python packages to work with state-space models in acoustics.

## Ecosystem Packages
- [`ssm_tools`](pyproject.toml): Parent package for joint maintenance, implementing fast time domain solvers.
- [`across`](packages/across/README.md): Constructs reduced order state-space models from impulse response data.
- [`irdl`](https://github.com/artpelling/irdl): Downloads and processes impulse response datasets.

## `ssm_tools` package
The `ssm_tools` package contains [`pyfar`](https://pyfar.org)-compatible classes of differently structured state space models implementing fast time domain solvers. Latest benchmarks can be found [here](https://artpelling.github.io/ssm-tools-asv).

### Installation
Add
``` toml
dependencies = [
    "ssm_tools @ git+https://github.com/artpelling/ssm-tools"
]
```
to your `pyproject.toml` (or similarly to `requirements.txt`).

### BLAS backend

The Rust extension links against a BLAS implementation selected by the [`blas-src`](https://crates.io/crates/blas-src) crate. The default is **system OpenBLAS** (`openblas-src` with the `system` feature). To switch backends, edit `Cargo.toml`:

| Backend | Change `blas-src` feature to | Also add |
|---|---|---|
| System OpenBLAS *(default)* | `openblas` | `openblas-src = { …, features = ["system"] }` |
| Compiled OpenBLAS | `openblas` | *(remove the `openblas-src` override)* |
| Netlib reference | `netlib` | `netlib-src = { … }` |
| Apple Accelerate | `accelerate` | *(macOS only, no extra crate needed)* |
| Intel MKL | `intel-mkl` | `intel-mkl-src = { … }` |

For example, to use compiled OpenBLAS instead of the system library:
```toml
blas-src = { version = "0.14", default-features = false, features = ["openblas"] }
# remove the openblas-src line
```
