# `across`: Acoustic Reduced Order State-space Systems
Python package for reduced order modelling of acoustics state space systems with the Eigensystem Realization Algorithm (ERA). The package wraps [`pymor`](https://pymor.org) reductors for use with [`pyfar`](https://pyfar.org) signals to remove boilerplate code and assure compatibility.

## Installation
Add
``` toml
dependencies = [
   "across @ git+https://github.com/artpelling/ssm-tools#subdirectory=packages/across"
]
```
to your `pyproject.toml` (or similarly to `requirements.txt`). 

## Example
To create a reduced order state-space system from impulse response data, simply pass the data to the `ERA` reductor class and call its `reduce` method (works well for short IRs).
``` python
from across import ERA

era = ERA(ir)
ssm = ERA.reduce(50)
```

## Outlook
For more complex IRs, classical ERA can lead to substantial computations. In this case, using randomized matrix approximations might be preferred. It is intended to add this to `across` in the near future. A working implementation can be found in the [`adaptive-era` repo](https://github.com/artpelling/adaptive-era). For more, consult:
- https://doi.org/10.3390/acoustics3030037
- https://arxiv.org/abs/2506.08870
- https://epubs.siam.org/doi/abs/10.1137/20M1327616
