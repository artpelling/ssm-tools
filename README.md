# `ssm-tools`: State-space System Modelling Tools
A collection of interoperable Python packages to work with state-space models in acoustics.

## Ecosystem Packages
- [`ssm_tools`](pyproject.toml): Parent package for joint maintenance, implementing fast time domain solvers.
- [`across`](packages/across/README.md): Constructs reduced order state-space models from impulse response data.
- [`irdl`](packages/irdl/README.md): Downloads and processes impulse response datasets.

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
