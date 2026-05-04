# `across`: Acoustic Reduced Order State-space Systems

Python package for reduced-order modelling of acoustic state-space systems with the
Eigensystem Realization Algorithm (ERA).

The package wraps [`pymor`](https://pymor.org) reductors for use with [`pyfar`](https://pyfar.org)
signals, removing boilerplate and ensuring compatibility.

## Installation

```toml
# pyproject.toml
dependencies = [
    "across @ git+https://github.com/artpelling/ssm-tools#subdirectory=packages/across"
]
```

## Example

Create a reduced-order state-space system from impulse response data:

```python
from across import ERA

era = ERA(ir)
ssm = era.reduce(50)
```

For large or dense IRs, randomized matrix approximations offer better scalability:

```python
from across import RandomizedERA

era = RandomizedERA(ir)
ssm = era.reduce(50)
```

## References

- [Pelling et al., MSSP 2025](https://doi.org/10.1016/j.ymssp.2025.113613)
- [Pelling & Sarradj, Acoustics 2021](https://doi.org/10.3390/acoustics3030037)
- [Martinsson & Tropp, SIAM 2020](https://epubs.siam.org/doi/abs/10.1137/20M1327616)

