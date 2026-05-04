# State-space System Modelling Tools

A collection of interoperable Python packages for working with state-space models in acoustics.

Any discrete-time LTI system can be formulated in state-space. The system's action from input $u$ to output $y$ is governed by the so-called state equations:

$$
\begin{aligned}
\mathbf{x}[k+1] &= \mathbf{A}\mathbf{x}[k] + \mathbf{B}\mathbf{u}[k] \\
\mathbf{y}[k] &= \mathbf{C}\mathbf{x}[k] + \mathbf{D}\mathbf{u}[k]
\end{aligned}
$$

## Ecosystem

| Package | Description |
|---------|-------------|
| [`irdl`](https://artpelling.github.io/irdl/) | Downloads and processes impulse response datasets |
| [`across`](packages/across/README.md) | Reduced-order state-space models from impulse response data via ERA |
| [`ssmsolvers`](https://github.com/artpelling/ssmsolvers) | BLAS-accelerated Rust solvers for discrete-time state-space recursion |
| [`ssmsolve`](packages/ssmsolve/README.md) | Python state-space model classes backed by `ssmsolvers` |

### Workflow

```mermaid
graph LR
    subgraph s1["Download IR datasets"]
        irdl["irdl"]
    end

    pymor([pyMOR])
    numba([Numba])
    rocket-fft([rocket-fft])

    subgraph s2["Reduced-order modelling"]
        across["across"]
        pyfar([pyfar])
    end

    subgraph s3["Online computation"]
        direction TB
        ssmsolvers["ssmsolvers"]
        ssmsolve["ssmsolve"]
    end

    irdl -->|pyfar.Signal| across
    pymor -->|ERAReductor\nRandomizedERAReductor| across
    numba --> |JIT-compilation| across
    rocket-fft --> |fast FFT| across
    across -->|A, B, C, D| pyfar
    pyfar -->|StateSpaceModel| ssmsolve
    ssmsolvers -->|solve_f32\nsolve_f64| ssmsolve

    click across "packages/across/README.md"
    click ssmsolve "packages/ssmsolve/README.md"
    click ssmsolvers "https://github.com/artpelling/ssmsolvers"
    click irdl "https://github.com/artpelling/irdl"
    click numba "https://numba.pydata.org"
    click rocket-fft "https://github.com/styfenschaer/rocket-fft"
    click pyfar "https://pyfar.org"
    click pymor "https://pymor.org"
```

