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
| [`across`](packages/across/) | Reduced-order state-space models from impulse response data via ERA |
| [`ssmsolve`](packages/ssmsolve/) | Python state-space model classes with pluggable solver backends |
| [`ssmsolve-rs`](packages/ssmsolve-rs/) | BLAS-accelerated Rust solvers for discrete-time state-space recursion |

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
        ssmsolve-rs["ssmsolve-rs\n(optional)"]
        ssmsolve["ssmsolve"]
    end

    irdl -->|pyfar.Signal| across
    pymor -->|ERAReductor\nRandomizedERAReductor| across
    numba -->|JIT-compilation| across
    rocket-fft -->|fast FFT| across
    across -->|A, B, C, D| pyfar
    pyfar -->|StateSpaceModel| ssmsolve
    numba -.->|"[jit] extra"| ssmsolve
    ssmsolve-rs -.->|"[rust] extra"| ssmsolve

    click across "packages/across/"
    click ssmsolve "packages/ssmsolve/"
    click ssmsolve-rs "packages/ssmsolve-rs/"
    click irdl "https://github.com/artpelling/irdl"
    click numba "https://numba.pydata.org"
    click rocket-fft "https://github.com/styfenschaer/rocket-fft"
    click pyfar "https://pyfar.org"
    click pymor "https://pymor.org"
```

